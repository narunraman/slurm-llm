"""High-level orchestration for remote LLM job management."""

import fcntl
import hashlib
import logging
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml
from rich.console import Console

from .config import ConfigLoader, RemoteLLMError
from .job import JobManager, JobInfo
from .ssh_manager import SSHConnectionManager
from .scheduler import ClusterScheduler
from .health import (
    _check_job_errors,
    detect_vllm_port,
    wait_for_vllm_health,
    wait_for_job_running,
    query_model_info,
)

logger = logging.getLogger(__name__)
console = Console()


@contextmanager
def _model_readiness_lock(model_name: str, cluster_name: str, instance: int):
    """Serialize preparation of one model on this launcher host."""
    lock_key = f"{model_name}\0{cluster_name}\0{instance}".encode("utf-8")
    digest = hashlib.sha256(lock_key).hexdigest()[:20]
    lock_path = (
        Path(tempfile.gettempdir())
        / f"slurm-llm-model-{os.getuid()}-{digest}.lock"
    )

    with open(lock_path, "a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            console.print(
                f"  [dim]→ Another process is preparing {model_name}; waiting...[/dim]"
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class OrchestratorConfig:
    """Configuration for job orchestration behavior."""

    health_check_interval: int = 5  # Initial interval for health checks
    max_health_check_interval: int = 30  # Max interval for exponential backoff
    show_progress: bool = True  # Show progress bars
    quiet: bool = False  # Suppress INFO logs (only show warnings and errors)
    exclude_clusters: list[str] | None = None  # Clusters to exclude from selection


class JobOrchestrator:
    """
    High-level orchestration for remote LLM job lifecycle management.

    Coordinates:
    - Cluster selection via ClusterScheduler
    - SSH connection management via SSHConnectionManager
    - Job submission and tracking via JobManager
    - Health checking and readiness verification

    Example:
        orchestrator = JobOrchestrator()
        jobs = orchestrator.ensure_models_ready(["llama-3.1-8b-instruct", "llama-3.3-70b-instruct"])
    """

    def __init__(
        self, config_dir: str | None = None, config: OrchestratorConfig | None = None
    ):
        """
        Initialize job orchestrator.

        Args:
            config_dir: Path to config directory (default: ~/.config/slurm-llm)
            config: OrchestratorConfig for timeouts and behavior
        """
        self.config = config or OrchestratorConfig()
        self.config_loader = ConfigLoader(config_dir)
        self.job_manager = JobManager()
        self.conn_manager = SSHConnectionManager(self.config_loader)
        self.scheduler = ClusterScheduler(self.config_loader, self.conn_manager)

        if self.config.quiet:
            logging.getLogger("slurm_llm").setLevel(logging.WARNING)

    def ensure_models_ready(
        self,
        model_names: list[str],
        cluster_map: dict[str, str] | None = None,
        time_override: str | None = None,
        instance_map: dict[str, int] | None = None,
    ) -> dict[str, JobInfo]:
        """
        Ensure multiple models are running and healthy.

        For each model:
        1. Check for existing running job
        2. Submit new job if needed
        3. Wait for SLURM RUNNING status
        4. Detect vLLM port
        5. Wait for vLLM health

        Args:
            model_names: List of model names to prepare
            cluster_map: Optional mapping of model_name -> cluster_name overrides
            time_override: Optional SLURM time limit override (e.g., "4:00:00" for 4 hours)
            instance_map: Optional mapping of model_name -> instance ID

        Returns:
            dict mapping model_name -> JobInfo for all ready models

        Raises:
            Exception: If any model fails to become ready
        """
        unique_models = list(dict.fromkeys(model_names))

        console.print(
            f"\n[bold cyan]Preparing {len(unique_models)} model(s)...[/bold cyan]"
        )

        ready_jobs = {}
        for i, model_name in enumerate(unique_models, 1):
            console.print(
                f"\n[bold cyan]Model {i}/{len(unique_models)}: {model_name}[/bold cyan]"
            )

            try:
                instance = instance_map.get(model_name, 0) if instance_map else 0
                job = self.ensure_model_ready(
                    model_name,
                    cluster_map=cluster_map,
                    time_override=time_override,
                    exclude_clusters=self.config.exclude_clusters,
                    instance=instance,
                )
                ready_jobs[model_name] = job
                console.print(f"  [green]✓ Ready: {job.node}:{job.local_port}[/green]")

            except Exception as e:
                console.print(f"  [red]✗ Failed: {e}[/red]")
                raise

        # Return ready jobs (caller will display summary)
        return ready_jobs

    def ensure_model_ready(
        self,
        model_name: str,
        cluster_name: str | None = None,
        cluster_map: dict[str, str] | None = None,
        time_override: str | None = None,
        exclude_clusters: list[str] | None = None,
        instance: int = 0,
    ) -> JobInfo:
        """Ensure one model is ready without racing another local process."""
        model_config = self.config_loader.require_model(model_name)
        lock_model = model_config.id
        lock_cluster = cluster_name or (cluster_map or {}).get(model_name) or "auto"

        with _model_readiness_lock(lock_model, lock_cluster, instance):
            # The full discovery check deliberately happens after lock acquisition.
            # A process that waited will therefore see and reuse the first process's
            # PENDING or RUNNING job instead of submitting a duplicate.
            return self._ensure_model_ready_unlocked(
                model_name,
                cluster_name=cluster_name,
                cluster_map=cluster_map,
                time_override=time_override,
                exclude_clusters=exclude_clusters,
                instance=instance,
            )

    def _ensure_model_ready_unlocked(
        self,
        model_name: str,
        cluster_name: str | None = None,
        cluster_map: dict[str, str] | None = None,
        time_override: str | None = None,
        exclude_clusters: list[str] | None = None,
        instance: int = 0,
    ) -> JobInfo:
        """
        Ensure a single model is running and healthy.

        Full 5-step workflow:
        1. Check for existing running job
        2. Submit new job if needed (with smart cluster/GPU selection)
        3. Wait for SLURM RUNNING status
        4. Detect vLLM port from logs
        5. Wait for vLLM health check

        Args:
            model_name: Model name to prepare
            cluster_name: Optional explicit cluster override
            cluster_map: Optional mapping of model_name -> cluster_name
            time_override: Optional SLURM time limit override (e.g., "4:00:00" for 4 hours)
            instance: Instance ID (default 0)

        Returns:
            JobInfo for the ready model

        Raises:
            ValueError: If model not found in registry
            TimeoutError: If job doesn't reach running or healthy state
            Exception: Other errors during submission or health checks
        """
        # Step 1: Check ALL clusters for existing running job
        console.print("  [blue]→ Checking for existing job...[/blue]")

        clusters = self.config_loader.load_clusters()
        ssh_clients = {}

        # Determine which clusters to check for existing jobs
        if cluster_name:
            # Explicit cluster override: only check that cluster
            clusters_to_check = {cluster_name}
        elif cluster_map and model_name in cluster_map:
            # Cluster specified via cluster_map for this model
            clusters_to_check = {cluster_map[model_name]}
        else:
            # Check ALL available clusters for existing jobs
            exclude_set = set(exclude_clusters or [])
            clusters_to_check = {c for c in clusters.keys() if c not in exclude_set}

        # Connect to clusters in parallel
        def try_connect(name):
            try:
                client = self.conn_manager.get_client(name)
                console.print(f"    [dim]✓ Connected to {name}[/dim]")
                return name, client
            except Exception as e:
                logger.debug(f"Could not connect to cluster {name}: {e}")
                console.print(f"    [yellow]⚠ {e}[/yellow]")
                return name, None

        valid_clusters = [c for c in clusters_to_check if c in clusters]
        if len(valid_clusters) > 1:
            console.print(
                f"    [dim]Connecting to {len(valid_clusters)} clusters...[/dim]"
            )

        with ThreadPoolExecutor(max_workers=len(valid_clusters) or 1) as executor:
            futures = [executor.submit(try_connect, name) for name in valid_clusters]
            for future in as_completed(futures):
                name, client = future.result()
                if client:
                    ssh_clients[name] = client

        existing_job = self.job_manager.find_running_job_by_model(
            model_name, self.config_loader, ssh_clients=ssh_clients, refresh=True, instance=instance
        )

        if existing_job:
            status_display = existing_job.status
            console.print(
                f"  [green]✓ Found existing job: {existing_job.job_id} ({status_display}) on {existing_job.cluster_name}[/green]"
            )

            ssh_client = self.conn_manager.get_client(existing_job.cluster_name)

            # If job is PENDING, wait for it to start running first
            if existing_job.status in {"PENDING", "CONFIGURING"}:
                console.print("  [blue]→ Waiting for job to start running...[/blue]")
                try:
                    success = wait_for_job_running(
                        existing_job,
                        ssh_client,
                        self.job_manager,
                        estimated_wait=3600,  # Default estimate for pending job
                        show_progress=self.config.show_progress,
                    )
                    if not success:
                        raise Exception(f"Job {existing_job.job_id} failed to reach RUNNING status")
                except Exception as e:
                    console.print(f"  [yellow]⚠ Existing job failed: {e}[/yellow]")
                    console.print("  [yellow]→ Will submit new job[/yellow]")
                    existing_job = None  # Fall through to submit new job

            if existing_job and existing_job.status == "RUNNING":
                # Wait for job to be healthy with full timeout
                # Port detection will happen as part of health check process
                console.print("  [blue]→ Waiting for vLLM to be ready...[/blue]")

                try:
                    # Detect port if not already stored
                    if not existing_job.local_port:
                        existing_job.local_port = detect_vllm_port(existing_job, ssh_client)
                        self.job_manager.save_job(existing_job)

                    # Wait for health (polls until ready or job dies, no timeout)
                    wait_for_vllm_health(
                        existing_job,
                        ssh_client,
                        show_progress=self.config.show_progress,
                    )

                    # Query model info to get max_model_length (only if not already set)
                    if (
                        not existing_job.max_model_len
                        and existing_job.node
                        and existing_job.local_port
                    ):
                        model_info = query_model_info(
                            ssh_client, existing_job.node, existing_job.local_port
                        )
                        if model_info and "max_model_len" in model_info:
                            existing_job.max_model_len = model_info["max_model_len"]
                            self.job_manager.save_job(existing_job)
                            console.print(
                                f"  [dim]→ Max model length: {existing_job.max_model_len}[/dim]"
                            )

                    return existing_job
                except Exception as e:
                    console.print(f"  [yellow]⚠ Existing job not healthy: {e}[/yellow]")
                    console.print("  [yellow]→ Will submit new job[/yellow]")

        # Step 2: No healthy job - select best cluster and submit
        console.print("  [blue]→ No healthy job found, selecting cluster...[/blue]")

        model_config = self.config_loader.require_model(model_name)
        resource_config = self.config_loader.resource_for(model_config)
        if time_override:
            resource_config.time = time_override

        selected_cluster, estimated_wait = self.scheduler.select_cluster_for_model(
            model_config,
            resource_config,
            model_name=model_name,
            explicit_cluster=cluster_name,
            cluster_map=cluster_map,
            exclude_clusters=exclude_clusters,
        )
        wait_display = f"{estimated_wait // 60}min" if estimated_wait is not None else "unknown"
        console.print(f"  [dim]→ Selected: {selected_cluster} (est. wait: {wait_display})[/dim]")

        console.print("  [blue]→ Submitting new job...[/blue]")
        job = self._submit_job(model_name, selected_cluster, time_override=time_override, instance=instance)

        console.print(
            f"  [green]✓ Job submitted: {job.job_id} on {job.cluster_name}[/green]"
        )

        # Step 3: Wait for SLURM RUNNING
        ssh_client = self.conn_manager.get_client(job.cluster_name)

        try:
            success = wait_for_job_running(
                job,
                ssh_client,
                self.job_manager,
                estimated_wait=estimated_wait,
                show_progress=self.config.show_progress,
            )

            if not success:
                raise RemoteLLMError(
                    f"Job {job.job_id} ended before it started running (status: {job.status}). "
                    f"{_check_job_errors(ssh_client, job.job_id, job)}"
                )
        except TimeoutError as e:
            console.print(f"  [red]✗ Timeout waiting for job allocation: {e}[/red]")
            raise

        # Step 4: Detect vLLM port
        job.local_port = detect_vllm_port(
            job, ssh_client, show_progress=self.config.show_progress
        )

        self.job_manager.save_job(job)

        console.print(f"  [green]✓ Port detected: {job.local_port}[/green]")

        # Step 5: Wait for vLLM health (polls until ready or job dies, no timeout)
        try:
            wait_for_vllm_health(
                job,
                ssh_client,
                initial_interval=self.config.health_check_interval,
                max_interval=self.config.max_health_check_interval,
                show_progress=self.config.show_progress,
            )

            # Query model info to get max_model_length
            if job.node and job.local_port:
                model_info = query_model_info(ssh_client, job.node, job.local_port)
                if model_info and "max_model_len" in model_info:
                    job.max_model_len = model_info["max_model_len"]
                    self.job_manager.save_job(job)
                    console.print(
                        f"  [dim]→ Max model length: {job.max_model_len}[/dim]"
                    )

        except RuntimeError as e:
            error_str = str(e)
            if not sys.stdin.isatty():
                raise
            if "KV cache memory error" in error_str:
                console.print("  [red]✗ KV cache memory error detected[/red]")
                if self._handle_kv_cache_error(model_name, error_str):
                    raise RemoteLLMError("max_model_len updated in model_registry.yaml; run the command again") from None
            elif "Out of Memory" in error_str or "out of GPU memory" in error_str:
                console.print("  [red]✗ CUDA Out of Memory error detected[/red]")
                cluster_config = self.config_loader.load_clusters().get(job.cluster_name)
                if cluster_config and self._handle_oom_error(model_name, job, cluster_config, error_str):
                    raise RemoteLLMError("model_memory_gb updated in model_registry.yaml; run the command again") from None
            raise

        except TimeoutError as e:
            console.print(f"  [red]✗ Timeout waiting for vLLM health: {e}[/red]")
            raise

        # Return ready job (caller will print success message)
        return job

    def _submit_job(
        self,
        model_name: str,
        cluster_name: str,
        time_override: str | None = None,
        instance: int = 0,
    ) -> JobInfo:
        """Submit a vLLM job for the model on the chosen cluster, sized to its GPUs."""
        model_config = self.config_loader.require_model(model_name)
        resource_config = self.config_loader.resource_for(model_config)
        if time_override:
            resource_config.time = time_override
        cluster_config = self.config_loader.get_cluster(cluster_name)
        gpu_spec = self.scheduler.match_gpu_for_model(cluster_config, model_config, resource_config)
        if gpu_spec:
            console.print(
                f"    [dim]GPU allocation: {gpu_spec['num_gpus']} x {gpu_spec['type']} "
                f"({gpu_spec['memory']}GB each)[/dim]"
            )
        job_id = self.job_manager.submit_job(
            model_config=model_config,
            resource_config=resource_config,
            cluster_config=cluster_config,
            ssh_client=self.conn_manager.get_client(cluster_name),
            gpu_spec=gpu_spec,
            instance=instance,
        )
        return self.job_manager.get_job(job_id)

    def _update_model_registry(self, model_name: str, **updates) -> bool:
        """
        Update model registry with new values.

        Args:
            model_name: Name of the model to update
            **updates: Key-value pairs to update (e.g., max_model_len=128896, model_memory_gb=40)

        Returns:
            True if successfully updated, False otherwise
        """
        registry_path = self.config_loader.config_dir / "model_registry.yaml"
        if not registry_path.exists():
            # start from the packaged registry that is in use
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.config_loader.config_file("model_registry.yaml"), registry_path)

        try:
            with open(registry_path, "r") as f:
                registry = yaml.safe_load(f) or {}

            models = registry.get("models", [])
            model_config = self.config_loader.resolve_model(model_name)
            model_updated = False

            for model_data in models:
                if not isinstance(model_data, dict):
                    continue
                if (
                    model_data.get("short_name") == model_name
                    or model_data.get("id") == model_name
                    or (model_config and model_data.get("id") == model_config.id)
                ):
                    # Apply all updates
                    for key, value in updates.items():
                        model_data[key] = value
                    model_updated = True
                    display_name = model_data.get(
                        "short_name", model_data.get("id", "unknown")
                    )

                    # Show what was updated
                    update_strs = [f"{k} = {v}" for k, v in updates.items()]
                    console.print(
                        f"\n[green]✓ Updated {display_name}: {', '.join(update_strs)}[/green]"
                    )
                    break

            if not model_updated:
                console.print(
                    f"[yellow]Model '{model_name}' not found in registry[/yellow]"
                )
                return False

            with open(registry_path, "w") as f:
                yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

            console.print(f"[green]✓ Updated {registry_path}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Failed to update model registry: {e}[/red]")
            return False

    def _handle_kv_cache_error(self, model_name: str, error_message: str = "") -> bool:
        """
        Handle KV cache error interactively by offering to set max_model_len.

        Args:
            model_name: Name of the model that failed
            error_message: The full error message string

        Returns:
            True if model registry was updated, False otherwise
        """
        # Extract suggested max_model_len from error message
        suggested_max_len = None
        match = re.search(r"Suggested max_model_len: (\d+)", error_message)
        if match:
            suggested_max_len = int(match.group(1))

        # Get current max_model_len from registry
        model_config = self.config_loader.resolve_model(model_name)
        current_max_len = model_config.max_model_len if model_config else None

        # Ask if user wants to update
        console.print(
            "\n[bold yellow]💡 Would you like to set max_model_len to reduce the context window?[/bold yellow]"
        )

        if current_max_len:
            console.print(
                f"[dim]Current max_model_len in registry: {current_max_len}[/dim]"
            )
        else:
            console.print(
                "[dim]Current max_model_len in registry: Not set (uses model default)[/dim]"
            )

        if suggested_max_len:
            console.print(f"[cyan]vLLM suggests: {suggested_max_len}[/cyan]")

        from rich.prompt import Confirm, Prompt

        update = Confirm.ask(
            "\nSet max_model_len to limit context window?", default=False
        )

        if not update:
            return False

        # Prompt for max_model_len
        console.print("\n[bold cyan]Enter the maximum sequence length:[/bold cyan]")
        console.print(
            "[dim]Lower values use less GPU memory but limit context window[/dim]"
        )

        default_value = (
            str(suggested_max_len)
            if suggested_max_len
            else (str(current_max_len) if current_max_len else "32768")
        )

        max_len_str = Prompt.ask("Max sequence length", default=default_value)

        try:
            max_len = int(max_len_str)
        except ValueError:
            console.print("[red]Invalid input. max_model_len not updated.[/red]")
            return False

        # Update registry using shared function
        return self._update_model_registry(model_name, max_model_len=max_len)

    def _handle_oom_error(
        self, model_name: str, job: JobInfo, cluster_config, error_message: str = ""
    ) -> bool:
        """
        Handle OOM error interactively by offering to update model memory requirements.

        Args:
            model_name: Name of the model that failed
            job: JobInfo containing GPU allocation details
            cluster_config: ClusterConfig with available GPU types

        Returns:
            True if model registry was updated, False otherwise
        """
        # Show error details first for context
        if error_message:
            # Extract just the key parts without the full traceback
            if "Check logs:" in error_message:
                log_info = error_message.split("Check logs:")[1].split("\n")[0].strip()
                console.print(f"\n  [dim]Check logs: {log_info}[/dim]")

        # Get current memory requirement from registry using ConfigLoader
        # This handles all alias matching properly
        current_memory_gb = None
        model_config = self.config_loader.resolve_model(model_name)
        if model_config:
            current_memory_gb = model_config.model_memory_gb

        # Show context about what happened
        console.print("\n[bold yellow]⚠️  Out of Memory Error Detected[/bold yellow]")

        if current_memory_gb:
            console.print(
                f"[dim]Current requirement in registry: {current_memory_gb}GB[/dim]"
            )
        else:
            console.print(
                "[dim]Current requirement in registry: Not set (using size-based estimate)[/dim]"
            )

        # Show current GPU allocation that failed
        if job.gpu_info:
            console.print(
                f"[dim]GPU that failed: {job.gpu_info.get('num_gpus', 1)} x "
                f"{job.gpu_info.get('type', 'unknown')} ({job.gpu_info.get('memory', '?')}GB each)[/dim]"
            )

        from rich.prompt import Confirm, Prompt

        update = Confirm.ask(
            "\n[bold yellow]Update model memory requirements?[/bold yellow]",
            default=True,
        )

        if not update:
            return False

        # Get available GPU types from cluster
        gpu_options = []
        if cluster_config.gpu_types:
            console.print(
                "\n[bold cyan]Available GPU types on this cluster:[/bold cyan]"
            )
            for i, gpu_type in enumerate(cluster_config.gpu_types, 1):
                gpu_memory = gpu_type.memory
                gpu_name = gpu_type.type  # Use 'type' attribute, not 'name'
                gpu_full_or_mig = "Full GPU" if gpu_type.is_full_gpu else "MIG"
                console.print(f"  {i}. {gpu_name} ({gpu_memory}GB) - {gpu_full_or_mig}")
                gpu_options.append(gpu_memory)

        if not gpu_options:
            console.print("[yellow]No GPU types configured for this cluster[/yellow]")
            return False

        # Smart default: suggest next tier up from failed GPU
        suggested_memory = None
        if job.gpu_info and job.gpu_info.get("memory"):
            # Suggest next tier up from failed GPU
            failed_memory = job.gpu_info.get("memory")
            larger_options = [m for m in gpu_options if m > failed_memory]
            if larger_options:
                suggested_memory = str(min(larger_options))
                console.print(
                    f"\n[bold cyan]💡 Suggested: {suggested_memory}GB (next tier up from {failed_memory}GB that failed)[/bold cyan]"
                )
            else:
                suggested_memory = str(max(gpu_options))
                console.print(
                    f"\n[yellow]Note: {failed_memory}GB is the largest GPU available on this cluster[/yellow]"
                )
        else:
            # Fallback to largest GPU
            suggested_memory = str(max(gpu_options))
            console.print(
                f"\n[bold cyan]💡 Suggested: {suggested_memory}GB (largest available)[/bold cyan]"
            )

        min_memory_str = Prompt.ask("Minimum GPU memory (GB)", default=suggested_memory)

        try:
            min_memory = int(min_memory_str)
        except ValueError:
            console.print("[red]Invalid input. Memory requirements not updated.[/red]")
            return False

        # Update registry using shared function
        return self._update_model_registry(model_name, model_memory_gb=min_memory)

    def close(self):
        """Close SSH connections."""
        self.conn_manager.close_all()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
