"""Client for remote vLLM models with automatic job management and tunneling."""

from __future__ import annotations

import logging
import math
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
from openai import OpenAI

from .config import ConfigLoader
from .job import JobInfo
from .orchestration import JobOrchestrator
from .tunnels import TunnelHandle, TunnelRegistry

logger = logging.getLogger(__name__)

__all__ = ["SlurmLLM", "TunnelHandle", "TunnelRegistry"]


def _model_ids_match(expected: str, actual: str) -> bool:
    """Compare model IDs while tolerating punctuation/case alias differences."""
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    return normalize(expected) == normalize(actual)


_BATCH_FORMAT = (
    'generate_batch expects a list of requests, each a dict with a "messages" list, e.g. '
    '[{"messages": [{"role": "user", "content": "Q1"}]}, {"messages": [{"role": "user", "content": "Q2"}]}]'
)


def _check_batch_requests(requests: Any) -> None:
    """Raise TypeError, explaining the expected format, unless `requests` is a list of request dicts."""
    if isinstance(requests, (str, bytes, dict)) or not isinstance(requests, (list, tuple)):
        raise TypeError(f"{_BATCH_FORMAT}; got {type(requests).__name__}")
    for idx, request in enumerate(requests):
        if isinstance(request, dict) and isinstance(request.get("messages"), list):
            continue
        if isinstance(request, dict) and "role" in request:
            hint = " (it looks like one message: wrap each conversation as {\"messages\": [...]})"
        elif isinstance(request, list):
            hint = " (a list of messages: wrap it as {\"messages\": [...]})"
        elif isinstance(request, str):
            hint = ' (a string: use {"messages": [{"role": "user", "content": ...}]})'
        elif isinstance(request, dict):
            hint = ' without a "messages" list'
        else:
            hint = ""
        raise TypeError(f"{_BATCH_FORMAT}; request {idx} is {type(request).__name__}{hint}")


class SlurmLLM:
    """Client for remote vLLM models with automatic job management and tunneling.

    Handles:
    - Job submission and health checking via JobOrchestrator
    - SSH tunnel creation to remote vLLM server
    - Model name resolution (alias -> served name)
    - Local mode: direct HTTP to vLLM nodes (no SSH tunneling)
    - Load balancing across multiple instances

    Examples:
        # Basic usage (from your machine, uses SSH tunneling)
        client = SlurmLLM(model="llama-3.1-8b-instruct")
        client.ensure_ready()  # Blocks until model is ready

        response = client.generate(
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7
        )
        print(response.choices[0].message.content)

        # Local mode (from cluster compute node, direct HTTP)
        client = SlurmLLM(
            model="qwen-7b",
            local=True,
            load_balance=True,
            job_ids=["12345", "12346"],
            nodes_dir="/path/to/remote_root/manifests/Qwen2.5-7B-Instruct/nodes"
        )
        client.ensure_ready()  # Discovers endpoints from job_ids

        # Batch processing across multiple instances
        responses = client.generate_batch([
            {"messages": [{"role": "user", "content": "Q1"}]},
            {"messages": [{"role": "user", "content": "Q2"}]},
        ])

        # Context manager (auto-cleanup)
        with SlurmLLM(model="llama-3.1-8b-instruct") as client:
            client.ensure_ready()
            response = client.generate(messages=[...])
    """

    def __init__(
        self,
        model: str,
        cluster: str | None = None,
        config_dir: str | None = None,
        local_port: int = 0,  # 0 = auto-assign free port
        walltime: str | None = None,  # SLURM time limit (e.g., "4:00:00")
        local: bool = False,  # Local mode: skip SSH, direct HTTP to nodes
        load_balance: bool = False,  # Use all available instances
        job_ids: list[str] | None = None,  # Explicit job IDs (local mode)
        nodes_dir: str | None = None,  # Path to nodes/ directory (local mode)
    ):
        """Initialize SlurmLLM client.

        Args:
            model: Model name with optional instance (e.g., "llama-70b" or "llama-70b:1")
            cluster: Force specific cluster (optional)
            config_dir: Config directory path (default: ~/.config/slurm-llm)
            local_port: Local port for SSH tunnel (default: 0 = auto-assign)
            walltime: SLURM time limit override (e.g., "4:00:00" for 4 hours)
            local: If True, skip SSH tunneling and connect directly to nodes.
                   Use this when running on a cluster compute node.
                   Requires nodes_dir to be set.
            load_balance: If True, discover and use all available instances.
                          Requests are distributed across instances.
            job_ids: List of SLURM job IDs to connect to (local mode).
                     If None, pool mode is used: all *.node files in nodes_dir
                     are discovered and new ones are picked up continuously.
            nodes_dir: Path to nodes/ directory containing {job_id}.node files.
                       Required when local=True.

        Raises:
            ValueError: If local=True but nodes_dir is not provided.
        """
        # Validate local mode requirements
        if local and not nodes_dir:
            raise ValueError(
                "nodes_dir is required when local=True. "
                "Pass the nodes_dir path from submit_model(). "
                "job_ids is optional; if omitted, all endpoints in nodes_dir are discovered (pool mode)."
            )
        # Parse model:instance syntax (e.g., "llama-70b:1")
        if ":" in model and model.rsplit(":", 1)[-1].isdigit():
            model_name, instance_str = model.rsplit(":", 1)
            self.model = model_name
            self.instance: int | None = int(instance_str)
        else:
            self.model = model
            self.instance = None  # None means "any instance or default"

        self.cluster = cluster
        self.local_port = local_port
        self.walltime = walltime
        self.local = local
        self.load_balance = load_balance
        self.job_ids = job_ids
        self.nodes_dir = nodes_dir

        # Resolve short name → vLLM served name (e.g. "qwen-3-32b" → "Qwen3-32B")
        self.served_model_name: str | None = None
        try:
            loader = ConfigLoader(config_dir)
            model_config = loader.resolve_model(self.model)
            if model_config:
                self.served_model_name = (
                    model_config.id.split("--")[-1]
                    if "--" in model_config.id
                    else model_config.id
                )
        except Exception:
            pass

        # In local mode, we don't need orchestrator for SSH
        if not local:
            self.orchestrator = JobOrchestrator(config_dir)
        else:
            self.orchestrator = None

        self._job: JobInfo | None = None
        self._ssh_client = None
        self._tunnel_handle: TunnelHandle | None = None
        self._openai_client: OpenAI | None = None
        self.max_model_len: int | None = None  # Set during ensure_ready()

        # For load balancing: list of (endpoint, client) tuples
        self._endpoints: list[tuple[str, OpenAI]] = []
        self._endpoint_index = 0  # Round-robin counter
        self._endpoints_lock = threading.Lock()  # Protects _endpoints and _endpoint_index

        # Tunnel registry (only used in non-local mode)
        if not local:
            self._tunnel_registry = TunnelRegistry.get_instance()
            # Clean up any stale tunnels from previous sessions
            cleaned = self._tunnel_registry.cleanup_stale_tunnels()
            if cleaned > 0:
                logger.info(
                    f"Cleaned up {cleaned} stale tunnel(s) from previous sessions"
                )
        else:
            self._tunnel_registry = None

    def ensure_ready(self) -> None:
        """Ensure model job is running and tunnel is established.

        In normal mode:
        1. Submits job if needed (or finds existing healthy job)
        2. Waits indefinitely for job to reach RUNNING status
        3. Waits for vLLM server to be healthy
        4. Creates SSH tunnel to remote server
        5. Initializes OpenAI client pointing to tunnel

        In local mode (local=True):
        1. Reads the jobs' endpoint files in nodes_dir
        2. Probes endpoints to find healthy instances
        3. Initializes OpenAI client(s) pointing directly to nodes

        Can be called multiple times safely (idempotent).
        """
        if self._openai_client or self._endpoints:
            return  # Already ready

        if self.local:
            self._ensure_ready_local()
        else:
            self._ensure_ready_remote()

    def _ensure_ready_local(self) -> None:
        """Local mode: discover endpoints from job_ids and connect directly.

        Discovery approach:
        1. Use job_ids provided at init time
        2. Check nodes/{job_id}.node files for node info
        3. Parse SLURM log for actual port (vLLM startup message)
        4. Health check before adding to pool
        5. If load_balance=True, start background polling for remaining endpoints
        """
        logger.info(f"Ensuring model '{self.model}' is ready (local mode)")

        assert self.nodes_dir is not None, "nodes_dir must be set for local mode"

        self._nodes_dir = Path(self.nodes_dir)
        self._pool_mode = self.job_ids is None
        self._discovered_job_ids: set[str] = set()

        if self.job_ids:
            jobs = [{"job_id": jid, "instance": i} for i, jid in enumerate(self.job_ids)]
        else:
            # Pool mode: discover all endpoints in nodes_dir
            node_files = sorted(self._nodes_dir.glob("*.node"))
            node_files = self._cleanup_stale_node_files(node_files)
            jobs = [{"job_id": f.stem, "instance": i} for i, f in enumerate(node_files)]
            logger.info(f"Pool mode: found {len(jobs)} node file(s) in {self._nodes_dir}")

        self._expected_jobs = len(jobs)

        logger.info(f"Discovering {len(jobs)} job(s){' (pool mode)' if self._pool_mode else ''}")

        # Discover endpoints
        discovered_endpoints = self._discover_endpoints_from_jobs(jobs)

        if not discovered_endpoints:
            # No endpoints ready yet - wait and poll
            logger.info(
                "No endpoints ready yet, waiting for instances to come online..."
            )
            discovered_endpoints = self._wait_for_endpoints(jobs, min_endpoints=1)

        if not discovered_endpoints:
            raise RuntimeError(
                f"Could not discover any healthy endpoints for '{self.model}'. "
                "Check that vLLM jobs are running and logs/endpoint files are accessible."
            )

        # Create OpenAI clients for each endpoint
        for ep in discovered_endpoints:
            self._add_endpoint(ep)

        # Set primary client (for single-request compatibility)
        self._openai_client = self._endpoints[0][1]

        # Verify at least one endpoint and get served model name
        for endpoint_str, client in self._endpoints:
            try:
                models_list = client.models.list()
                if models_list.data:
                    self.served_model_name = models_list.data[0].id
                    break
            except Exception as e:
                logger.warning(f"Could not verify endpoint {endpoint_str}: {e}")

        logger.info(f"Local mode ready with {len(self._endpoints)} endpoint(s)")

        # Start background polling if load_balance is enabled
        # In pool mode: always poll for new endpoints (new vLLM instances joining the pool)
        # In explicit mode: poll until all expected endpoints are discovered
        if self.load_balance and (self._pool_mode or len(self._discovered_job_ids) < self._expected_jobs):
            self._start_background_discovery(jobs)

    def _add_endpoint(self, ep: dict) -> None:
        """Add an endpoint to the pool."""
        endpoint_str = f"{ep['node']}:{ep['port']}"

        with self._endpoints_lock:
            # Check if already added
            for existing_str, _ in self._endpoints:
                if existing_str == endpoint_str:
                    return

            base_url = f"http://{ep['node']}:{ep['port']}/v1"
            client = OpenAI(
                base_url=base_url,
                api_key="dummy",
                timeout=1800.0,
            )
            self._endpoints.append((endpoint_str, client))
            self._discovered_job_ids.add(ep["job_id"])

        logger.info(f"Added endpoint: {endpoint_str} (job {ep['job_id']})")

    def _discover_endpoints_from_jobs(self, jobs: list[dict]) -> list[dict]:
        """Discover endpoints using node files and SLURM logs."""
        discovered = []

        for job_info in jobs:
            job_id = job_info["job_id"]
            instance = job_info.get("instance", 0)

            endpoint = self._discover_single_endpoint(job_id, instance)
            if endpoint:
                discovered.append(endpoint)

        return discovered

    def _discover_single_endpoint(self, job_id: str, instance: int = 0) -> dict | None:
        """Discover a single endpoint from node file and SLURM log.

        1. Read node file for node hostname
        2. Parse SLURM log for port
        3. Health check the endpoint
        """
        # Check node file for node info
        node_file = self._nodes_dir / f"{job_id}.node"

        if not node_file.exists():
            logger.debug(f"Node file not found: {node_file}")
            return None

        # Parse node file
        node = None
        try:
            with open(node_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("node="):
                        node = line.split("=", 1)[1]
                        break
        except Exception as e:
            logger.debug(f"Error reading node file {node_file}: {e}")
            return None

        if not node:
            logger.debug(f"No node found in node file {node_file}")
            return None

        # Find SLURM log and parse for port
        # nodes_dir is at {remote_root}/manifests/{model}/nodes
        # logs are at {remote_root}/slurm-{job_id}.out
        log_candidates = [
            self._nodes_dir.parent.parent.parent / f"slurm-{job_id}.out",  # remote_root
            self._nodes_dir.parent.parent / f"slurm-{job_id}.out",  # manifests level
            Path.home() / f"slurm-{job_id}.out",
        ]

        port = None
        for log_path in log_candidates:
            if log_path.exists():
                port = self._parse_port_from_log(log_path)
                if port:
                    break

        if not port:
            logger.debug(f"Could not find port for job {job_id} in logs")
            return None

        # Health check
        if not self._probe_endpoint(node, port):
            logger.debug(f"Endpoint {node}:{port} not healthy yet")
            return None

        # Verify the endpoint is serving the correct model
        if not self._verify_model_at_endpoint(node, port):
            logger.warning(
                f"Endpoint {node}:{port} (job {job_id}) serving wrong model, skipping"
            )
            return None

        return {
            "node": node,
            "port": port,
            "instance": instance,
            "job_id": job_id,
        }

    def _parse_port_from_log(self, log_path: Path) -> int | None:
        """Parse vLLM port from SLURM log file.

        Looks for the job template's "Port selection: ... actual=PORT" line, then
        "Starting vLLM [API ]server X on http://0.0.0.0:PORT"
        (vLLM dropped "API server N" -> "server" around v0.19, so match both).
        """
        try:
            with open(log_path, "r") as f:
                content = f.read()

            port_match = re.search(r"Port selection:.*\bactual=(\d+)", content)
            if port_match:
                return int(port_match.group(1))

            # Matches "Starting vLLM API server 0 on http://..." (vLLM <=0.11)
            # and "Starting vLLM server on http://..." (vLLM >=0.19).
            port_match = re.search(
                r"Starting vLLM (?:API )?server.*on http://[^:]+:(\d+)",
                content,
                re.IGNORECASE,
            )
            if port_match:
                return int(port_match.group(1))

            # Fallback: Uvicorn running pattern (older vLLM versions)
            port_match = re.search(r"Uvicorn running on http://[^:]+:(\d+)", content)
            if port_match:
                return int(port_match.group(1))

            return None
        except Exception as e:
            logger.debug(f"Error reading log {log_path}: {e}")
            return None

    def _wait_for_endpoints(
        self,
        jobs: list[dict],
        min_endpoints: int = 1,
        poll_interval: float = 5,
    ) -> list[dict]:
        """Wait for at least min_endpoints to become available.

        No timeout - relies on SLURM dependency to ensure jobs are running.
        """
        discovered = []

        while len(discovered) < min_endpoints:
            discovered = self._discover_endpoints_from_jobs(jobs)

            if len(discovered) >= min_endpoints:
                return discovered

            logger.info(
                f"Found {len(discovered)}/{min_endpoints} endpoints, waiting..."
            )
            time.sleep(poll_interval)

        return discovered

    def _start_background_discovery(self, jobs: list[dict]) -> None:
        """Start background thread to discover remaining endpoints.

        In pool mode (no explicit job_ids), continuously re-scans nodes_dir
        for new .node files so newly launched vLLM instances are picked up.
        In explicit mode, polls the fixed job list until all are discovered.
        """
        def poll_for_new_endpoints():
            while True:
                if self._pool_mode:
                    # Re-scan nodes_dir for any new .node files
                    node_files = sorted(self._nodes_dir.glob("*.node"))
                    current_jobs = [
                        {"job_id": f.stem, "instance": i}
                        for i, f in enumerate(node_files)
                    ]
                else:
                    current_jobs = jobs

                for job_info in current_jobs:
                    job_id = job_info["job_id"]
                    if job_id in self._discovered_job_ids:
                        continue

                    endpoint = self._discover_single_endpoint(
                        job_id, job_info.get("instance", 0)
                    )
                    if endpoint:
                        self._add_endpoint(endpoint)
                        logger.info(
                            f"Background discovery: added {endpoint['node']}:{endpoint['port']} "
                            f"(job {job_id}, {len(self._discovered_job_ids)} endpoints total)"
                        )

                # In explicit mode, stop once all expected endpoints are found
                if not self._pool_mode and len(self._discovered_job_ids) >= self._expected_jobs:
                    logger.info(
                        "All expected endpoints discovered, stopping background polling"
                    )
                    break

                time.sleep(10)  # Poll every 10 seconds

        self._discovery_thread = threading.Thread(
            target=poll_for_new_endpoints, daemon=True
        )
        self._discovery_thread.start()
        if self._pool_mode:
            logger.info("Started background discovery in pool mode (continuous scanning)")
        else:
            logger.info(
                f"Started background discovery for remaining {self._expected_jobs - len(self._discovered_job_ids)} endpoints"
            )

    def _cleanup_stale_node_files(self, node_files: list[Path]) -> list[Path]:
        """Remove node files whose SLURM jobs are no longer running.

        Uses 'squeue' to check job status in a single batch call.
        Falls back to keeping all files if squeue is unavailable.
        """
        if not node_files:
            return node_files

        job_ids = [f.stem for f in node_files]

        try:
            result = subprocess.run(
                ["squeue", "-j", ",".join(job_ids), "--noheader", "--format=%i"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return node_files

            active_ids = set(result.stdout.strip().split())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return node_files

        live = []
        for f in node_files:
            if f.stem in active_ids:
                live.append(f)
            else:
                logger.info(f"Removing stale node file: {f.name} (job no longer running)")
                try:
                    f.unlink()
                except OSError:
                    pass
        return live

    def _probe_endpoint(self, node: str, port: int, timeout: float = 30.0) -> bool:
        """Check if an endpoint is responding."""
        try:
            response = httpx.get(f"http://{node}:{port}/health", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def _verify_model_at_endpoint(self, node: str, port: int) -> bool:
        """Verify the endpoint is serving the expected model.

        Queries /v1/models and checks the served model name matches
        self.served_model_name (if already known) or self.model.
        Prevents cross-model contamination when multiple vLLM instances
        share a node via SO_REUSEPORT or stale node files exist.
        """
        try:
            tmp_client = OpenAI(
                base_url=f"http://{node}:{port}/v1",
                api_key="dummy",
                timeout=10.0,
            )
            models_list = tmp_client.models.list()
            if not models_list.data:
                return False

            served = models_list.data[0].id
            expected = getattr(self, "served_model_name", None) or self.model

            if _model_ids_match(expected, served):
                return True

            logger.warning(
                f"Model mismatch at {node}:{port}: "
                f"serving '{served}', expected '{expected}'"
            )
            return False
        except Exception as e:
            logger.debug(f"Could not verify model at {node}:{port}: {e}")
            return False

    def _remove_dead_endpoint(self, endpoint_str: str) -> None:
        """Remove a dead endpoint from the pool."""
        with self._endpoints_lock:
            for i, (ep_str, _) in enumerate(self._endpoints):
                if ep_str == endpoint_str:
                    self._endpoints.pop(i)
                    logger.warning(
                        f"Removed dead endpoint: {endpoint_str} ({len(self._endpoints)} remaining)"
                    )

                    # Reset index if now out of bounds and update primary client
                    if self._endpoints:
                        self._endpoint_index = self._endpoint_index % len(self._endpoints)
                        self._openai_client = self._endpoints[0][1]
                    else:
                        self._endpoint_index = 0
                        self._openai_client = None
                    break

    def _ensure_ready_remote(self) -> None:
        """Remote mode: use SSH tunneling (original behavior)."""
        assert self.orchestrator is not None, "orchestrator required for remote mode"
        assert (
            self._tunnel_registry is not None
        ), "tunnel registry required for remote mode"

        logger.info(f"Ensuring model '{self.model}' is ready")

        # 1. Prepare job (blocks until RUNNING + healthy)
        # Note: SSH connections will be created on-demand by get_client()
        # and will reuse existing ControlMaster sockets automatically
        cluster_map = {self.model: self.cluster} if self.cluster else None
        instance_map = (
            {self.model: self.instance} if self.instance is not None else None
        )
        jobs = self.orchestrator.ensure_models_ready(
            [self.model],
            cluster_map=cluster_map,
            time_override=self.walltime,
            instance_map=instance_map,
        )
        self._job = jobs[self.model]

        # Store max_model_len from job info (populated from vLLM runtime or model_specs)
        self.max_model_len = self._job.max_model_len

        logger.info(
            f"Job ready: {self._job.job_id} on {self._job.cluster_name} "
            f"({self._job.node}:{self._job.local_port}, context={self.max_model_len})"
        )

        # 2. Get SSH connection
        self._ssh_client = self.orchestrator.conn_manager.get_client(
            self._job.cluster_name
        )

        # 3. Create tunnel
        self._tunnel_handle = self._create_tunnel()
        logger.info(
            f"Tunnel created: localhost:{self.local_port} -> {self._job.node}:{self._job.local_port}"
        )

        # 4. Create OpenAI client with extended timeout for large models
        self._openai_client = OpenAI(
            base_url=f"http://localhost:{self.local_port}/v1",
            api_key="dummy",  # vLLM doesn't require real API key
            timeout=1800.0,  # 30 minutes timeout for large/reasoning models
        )

        # 5. Verify connectivity (ensure tunnel is actually forwarding)
        logger.info("Verifying tunnel connectivity...")

        start_time = time.time()
        connected = False
        while time.time() - start_time < 10:  # 10s timeout
            try:
                # Try to list models to verify connection
                models_list = self._openai_client.models.list()
                if models_list.data:
                    actual_model = models_list.data[0].id
                    expected_model = self.served_model_name or self.model
                    if not _model_ids_match(expected_model, actual_model):
                        self.close()
                        raise RuntimeError(
                            "Tunnel reached an unexpected model: "
                            f"expected '{expected_model}', got '{actual_model}' "
                            f"for job {self._job.job_id} at "
                            f"{self._job.node}:{self._job.local_port}"
                        )
                    self.served_model_name = actual_model
                    connected = True
                    break
                time.sleep(1)
            except (httpx.ConnectError, httpx.ReadTimeout, openai.APIConnectionError):
                # httpx error or openai wrapper error
                time.sleep(1)
            except Exception as e:
                # If it's a connection refused type error, wait
                if "Connection refused" in str(e) or "Connection error" in str(e):
                    time.sleep(1)
                else:
                    raise e

        if not connected:
            self.close()
            raise RuntimeError(
                f"Tunnel created but failed to forward traffic to {self.local_port} after 10s. "
                "Check remote vLLM logs."
            )

        logger.info("Tunnel verified and active.")

    def _create_tunnel(self) -> TunnelHandle:
        """Create SSH tunnel localhost:local_port -> remote_node:remote_port.

        Uses SSH master connection via ControlPath.
        If local_port is 0 or in use, finds a free port.

        Only called in remote mode where _job and _tunnel_registry are guaranteed set.

        Returns:
            Description of the forward installed in the SSH control master.
        """

        def is_port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        # If a specific port is requested, verify it's free
        if self.local_port > 0 and is_port_in_use(self.local_port):
            logger.warning(
                f"Port {self.local_port} is in use. Switching to random free port."
            )
            self.local_port = 0

        # Select free port if 0
        if self.local_port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self.local_port = s.getsockname()[1]
                logger.info(f"Selected random free local port: {self.local_port}")

        if not self._ssh_client:
            raise RuntimeError("SSH client not initialized before tunnel creation.")

        assert self._job is not None, "_job must be set before tunnel creation"
        assert (
            self._tunnel_registry is not None
        ), "_tunnel_registry required for remote mode"

        target_parts = shlex.split(self._ssh_client._get_ssh_target())
        ssh_target = target_parts[-1]
        ssh_options = tuple(target_parts[:-1])
        handle = TunnelHandle(
            local_port=self.local_port,
            forward_spec=f"{self.local_port}:{self._job.node}:{self._job.local_port}",
            ssh_target=ssh_target,
            control_socket=str(self._ssh_client.ssh_master_socket),
            ssh_options=ssh_options,
            owner_pid=os.getpid(),
        )

        cmd = handle.command("forward")
        logger.debug("Starting tunnel: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while creating SSH tunnel on port {self.local_port}."
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or "no SSH error output"
            raise RuntimeError(
                f"Tunnel failed to start (exit code {result.returncode}): {detail}"
            )

        # Register tunnel for cleanup tracking
        self._tunnel_registry.register_tunnel(
            handle=handle,
            model=self.model,
            cluster=self._job.cluster_name if self._job else "unknown",
        )

        return handle

    def _get_client_for_request(self) -> tuple[OpenAI, str | None]:
        """Get the OpenAI client to use for the next request.

        In load_balance mode with multiple endpoints, uses round-robin.
        Otherwise returns the primary client.

        Returns:
            Tuple of (OpenAI client, endpoint_str or None)

        Raises:
            RuntimeError: If no client is available
        """
        if self.load_balance:
            with self._endpoints_lock:
                if len(self._endpoints) > 1:
                    # Clamp index to valid range (in case endpoints were removed)
                    self._endpoint_index = self._endpoint_index % len(self._endpoints)
                    endpoint_str, client = self._endpoints[self._endpoint_index]
                    self._endpoint_index = (self._endpoint_index + 1) % len(self._endpoints)
                    return client, endpoint_str
                elif len(self._endpoints) == 1:
                    return self._endpoints[0][1], self._endpoints[0][0]
        if self._openai_client is None:
            raise RuntimeError("Client not ready. Call ensure_ready() first.")
        return self._openai_client, None

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        """Generate a chat completion.

        This is the primary API for SlurmLLM. It abstracts over the underlying
        vLLM/OpenAI endpoint and automatically uses the correct model name.

        In load_balance mode, requests are distributed across available instances.
        If an endpoint fails with a connection error, it's removed from the pool.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            OpenAI ChatCompletion response object

        Raises:
            RuntimeError: If client is not ready or all endpoints are dead

        Example:
            response = client.generate(
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.7,
                max_tokens=100
            )
            print(response.choices[0].message.content)
        """
        if not self._openai_client and not self._endpoints:
            raise RuntimeError("Client not ready. Call ensure_ready() first.")

        # Get client (with load balancing if enabled)
        client, endpoint_str = self._get_client_for_request()

        # Smart max_tokens handling
        # Estimate input tokens (optimistic: ~4 chars per token)
        input_estimate = sum(len(m.get("content", "")) for m in messages) // 4

        if max_tokens is None:
            # AUTO-FIT MODE: Calculate best-effort max output
            if self.max_model_len:
                effective_max_tokens = self.max_model_len - input_estimate - 100
                effective_max_tokens = max(1, effective_max_tokens)
            else:
                effective_max_tokens = None  # Let vLLM decide
        else:
            # EXPLICIT MODE: Validate upfront
            if self.max_model_len and max_tokens > self.max_model_len:
                raise ValueError(
                    f"max_tokens ({max_tokens}) exceeds model's max context "
                    f"({self.max_model_len}). Set max_tokens=None to auto-fit."
                )
            effective_max_tokens = max_tokens

        # Build request with correct model name
        request_kwargs = {
            "model": self.served_model_name,
            "messages": messages,
            **kwargs,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens

        # Make request with retry on max_tokens overflow and dead endpoint handling
        try:
            return client.chat.completions.create(**request_kwargs)
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            openai.APIConnectionError,
        ) as e:
            # Connection error - endpoint may be dead
            if endpoint_str and self.local:
                should_remove = True
                node = None
                port = None
                try:
                    node, port_str = endpoint_str.rsplit(":", 1)
                    port = int(port_str)
                except (ValueError, AttributeError):
                    node = None
                    port = None

                if node and port:
                    # Avoid killing healthy endpoints due to transient connect errors.
                    if self._probe_endpoint(node, port):
                        should_remove = False
                        logger.warning(
                            "Endpoint %s failed with %s but health check passed; keeping in pool",
                            endpoint_str,
                            type(e).__name__,
                        )

                if should_remove:
                    self._remove_dead_endpoint(endpoint_str)

                    # Retry with another endpoint if available
                    if self._endpoints:
                        logger.info(
                            f"Retrying request with another endpoint after {endpoint_str} failed"
                        )
                        return self.generate(
                            messages, temperature, max_tokens, **kwargs
                        )
                    raise RuntimeError(f"All endpoints are dead. Last error: {e}")
            raise
        except openai.NotFoundError as e:
            # Model name mismatch - endpoint is serving a different model.
            # This happens when multiple vLLM instances share a node (SO_REUSEPORT)
            # or when a stale node file points to a recycled endpoint.
            if endpoint_str and self.local:
                logger.warning(
                    f"Endpoint {endpoint_str} serving wrong model (got 404): {e}"
                )
                self._remove_dead_endpoint(endpoint_str)

                if self._endpoints:
                    return self.generate(
                        messages, temperature, max_tokens, **kwargs
                    )
                raise RuntimeError(
                    f"All endpoints removed (model mismatch). Last error: {e}"
                )
            raise
        except openai.BadRequestError as e:
            error_str = str(e)

            # Handle max_tokens overflow by parsing error and retrying
            if "max_tokens" in error_str and "too large" in error_str:
                corrected_max = self._extract_max_tokens_on_error(error_str)
                if corrected_max:
                    logger.info(f"max_tokens too large, retrying with {corrected_max}")
                    request_kwargs["max_tokens"] = corrected_max
                    return client.chat.completions.create(**request_kwargs)

            # Re-raise if not a max_tokens error or couldn't parse
            raise

    def generate_batch(
        self,
        requests: list[dict],
        max_concurrent_per_endpoint: int = 10,
        temperature: float | None = None,
        max_tokens: int | None = None,
        per_request_callback: Callable[[int, Any], None] | None = None,
        batch_timeout: float | None = None,
        **kwargs,
    ) -> list:
        """Generate completions for multiple requests in parallel.

        Distributes requests evenly across all available endpoints and
        processes them concurrently using a thread pool.

        Args:
            requests: List of request dicts, each containing at minimum:
                      - "messages": List of message dicts
                      Optionally:
                      - "temperature": Override temperature for this request
                      - "max_tokens": Override max_tokens for this request
            max_concurrent_per_endpoint: Max concurrent requests per endpoint
            temperature: Default temperature for all requests (can be overridden per-request)
            max_tokens: Default max_tokens for all requests (can be overridden per-request)
            per_request_callback: Optional callback(idx, response_or_exception) called
                                  immediately after each request completes. Useful for
                                  incremental progress tracking or saving results as they
                                  arrive. Called from worker threads, so must be thread-safe.
            batch_timeout: Optional timeout (in seconds) for the entire batch operation.
                          If None, allows one per-request timeout window for each
                          sequential worker wave, or 3600s when requests have no
                          timeout. Endpoint count never shortens a request's
                          timeout window.
            **kwargs: Additional arguments passed to all requests

        Returns:
            List of responses in the same order as input requests.
            Failed requests will have an Exception object instead of a response.

        Example:
            responses = client.generate_batch([
                {"messages": [{"role": "user", "content": "What is 2+2?"}]},
                {"messages": [{"role": "user", "content": "What is 3+3?"}]},
            ])
            for resp in responses:
                if isinstance(resp, Exception):
                    print(f"Error: {resp}")
                else:
                    print(resp.choices[0].message.content)
        """
        from concurrent.futures import (
            ThreadPoolExecutor,
            TimeoutError as FuturesTimeoutError,
            as_completed,
        )

        _check_batch_requests(requests)
        if not self._openai_client and not self._endpoints:
            raise RuntimeError("Client not ready. Call ensure_ready() first.")
        if max_concurrent_per_endpoint < 1:
            raise ValueError("max_concurrent_per_endpoint must be at least 1")

        num_endpoints = max(1, len(self._endpoints))
        max_workers = num_endpoints * max_concurrent_per_endpoint

        # Compute effective batch timeout
        # Every submitted worker has its own request timeout. The batch only
        # needs additional time when there are more requests than workers and
        # they must run in sequential waves. More endpoints must never reduce
        # the time available to an individual request.
        default_request_timeout = kwargs.get("timeout")
        request_timeouts = [
            float(timeout)
            for request in requests
            if (timeout := request.get("timeout", default_request_timeout))
        ]
        per_request_timeout = max(request_timeouts, default=None)
        if batch_timeout is None:
            if per_request_timeout:
                worker_waves = max(1, math.ceil(len(requests) / max_workers))
                effective_batch_timeout = per_request_timeout * worker_waves
            else:
                effective_batch_timeout = 3600.0  # 1 hour default
        else:
            effective_batch_timeout = float(batch_timeout)

        # Results indexed by original position
        results: list[Any] = [None] * len(requests)
        results_lock = threading.Lock()
        notified: set[int] = set()
        notification_lock = threading.Lock()

        def notify_once(idx: int, result: Any) -> None:
            """Invoke the incremental callback at most once per request."""
            if per_request_callback is None:
                return
            with notification_lock:
                if idx in notified:
                    return
                notified.add(idx)
            per_request_callback(idx, result)

        def record_result(idx: int, result: Any) -> None:
            """Keep the first terminal result, discarding late completions."""
            with results_lock:
                if results[idx] is not None:
                    return
                results[idx] = result
            notify_once(idx, result)

        def process_request(idx: int, req: dict) -> tuple[int, Any]:
            """Process a single request and return (index, result)."""
            try:
                # Get request-specific overrides
                req_messages = req["messages"]
                req_temp = req.get("temperature", temperature)
                req_max_tokens = req.get("max_tokens", max_tokens)

                # Merge any extra kwargs from request
                req_kwargs = {
                    k: v
                    for k, v in req.items()
                    if k not in ("messages", "temperature", "max_tokens")
                }
                req_kwargs.update(kwargs)

                response = self.generate(
                    messages=req_messages,
                    temperature=req_temp,
                    max_tokens=req_max_tokens,
                    **req_kwargs,
                )
                record_result(idx, response)
                return (idx, response)
            except Exception as e:
                logger.warning(f"Request {idx} failed: {e}")
                record_result(idx, e)
                return (idx, e)

        # Process all requests in parallel
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = [
            executor.submit(process_request, idx, req)
            for idx, req in enumerate(requests)
        ]
        batch_timed_out = False

        try:
            for future in as_completed(futures, timeout=effective_batch_timeout):
                future.result()
        except FuturesTimeoutError:
            batch_timed_out = True
            logger.warning(
                f"Batch timed out after {effective_batch_timeout}s. "
                f"Some requests may not have completed."
            )

            timed_out: list[tuple[int, TimeoutError]] = []
            with results_lock:
                for idx, result in enumerate(results):
                    if result is None:
                        timeout_error = TimeoutError(
                            f"Request {idx} did not complete within batch timeout"
                        )
                        results[idx] = timeout_error
                        timed_out.append((idx, timeout_error))

            for idx, timeout_error in timed_out:
                notify_once(idx, timeout_error)

            # Pending work can be cancelled immediately. Running synchronous
            # HTTP requests cannot be interrupted by Future.cancel(), but their
            # eventual completion is discarded by record_result and cannot
            # trigger a second callback.
            for future in futures:
                future.cancel()
        finally:
            executor.shutdown(
                wait=not batch_timed_out,
                cancel_futures=batch_timed_out,
            )

        return results

    def cancel_job(self) -> bool:
        """Cancel the SLURM job on the remote cluster.

        This stops the remote vLLM server and frees cluster resources.
        Also closes the SSH tunnel.

        Returns:
            True if job was cancelled successfully, False otherwise
        """
        if self.local:
            logger.warning("cancel_job not supported in local mode")
            return False

        if not self._job:
            logger.warning("No job to cancel")
            return False

        assert self.orchestrator is not None

        # Close tunnel first
        self.close()

        # Get SSH client for the cluster
        ssh_client = self.orchestrator.conn_manager.get_client(self._job.cluster_name)
        if not ssh_client:
            logger.error(f"Could not get SSH client for {self._job.cluster_name}")
            return False

        # Cancel the SLURM job
        job_id = self._job.job_id
        success = ssh_client.cancel_job(job_id)

        if success:
            logger.info(f"Cancelled job {job_id} on {self._job.cluster_name}")
            # Clean up job from local tracking
            self.orchestrator.job_manager.cleanup_job(job_id)
            self._job = None
        else:
            logger.error(f"Failed to cancel job {job_id}")

        return success

    def close(self) -> None:
        """Close tunnel and cleanup resources."""
        handle = self._tunnel_handle
        if handle:
            try:
                result = subprocess.run(
                    handle.command("cancel"),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    logger.info("Closed tunnel on port %s", handle.local_port)
                else:
                    logger.debug(
                        "Could not cancel tunnel on port %s: %s",
                        handle.local_port,
                        result.stderr.strip(),
                    )
            except Exception as e:
                logger.warning(f"Error closing tunnel: {e}")
            finally:
                self._tunnel_handle = None

        # Unregister from tunnel registry
        if handle and self._tunnel_registry is not None:
            self._tunnel_registry.unregister_tunnel(
                handle.local_port, handle.owner_pid
            )

    def _extract_max_tokens_on_error(self, error_str):
        """Extract max_tokens from error message."""
        match = re.search(r"(\d+)\s*input tokens.*?(\d+)\s*-\s*\d+", error_str)
        if match:
            actual_input = int(match.group(1))
            max_context = int(match.group(2))
            corrected_max = max_context - actual_input - 10  # small buffer
            return corrected_max
        return None

    def __del__(self):
        """Cleanup tunnel on deletion."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup tunnel."""
        self.close()


def __getattr__(name):
    if name == "RemoteLLM":  # old name
        from . import _renamed

        return _renamed(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
