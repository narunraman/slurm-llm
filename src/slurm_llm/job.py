"""Slurm job scripts and the local job cache."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import ClusterConfig, ConfigError, ModelConfig, ResourceConfig, state_dir
from .template_source import TemplateSource, resolve_template
from .ssh import SSHClient, shell_path

logger = logging.getLogger(__name__)

ACTIVE_STATES = {"PENDING", "SUBMITTED", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}

# Placeholders that job templates (slurm_template) and setup templates (setup_template) may use
SLURM_TEMPLATE_PLACEHOLDERS = (
    "model_id", "model_name", "account", "sbatch_lines", "time", "memory", "num_gpus", "gpu_slurm_arg",
    "nodes_arg", "cpus_per_task", "output_dir", "model_path", "venv_path", "setup_commands",
    "max_model_len_arg", "vllm_extra_args",
)
SETUP_TEMPLATE_PLACEHOLDERS = ("venv_path", "setup_commands")


@dataclass
class JobInfo:
    """A vLLM job known to this machine."""

    job_id: str
    model_name: str
    model_id: str
    cluster_name: str
    submit_time: str
    status: str = "SUBMITTED"
    node: str | None = None
    local_port: int | None = None  # the vLLM port on the compute node
    gpu_info: dict[str, Any] | None = None
    max_model_len: int | None = None  # as reported by vLLM
    time_left: str | None = None
    instance: int = 0  # to run several copies of one model
    template: str | None = None  # the job template the script was made from, and its fingerprint
    template_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobInfo":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# Also print the cluster's clock: squeue reports times in the cluster's local time zone.
DISCOVERY_COMMAND = (
    "squeue -u $USER -h -o '%i|%j|%T|%N|%L|%V'; rc=$?; "
    "echo SLURM_LLM_NOW $(date +%s) $(date +%Y-%m-%dT%H:%M:%S); exit $rc"
)


def cluster_time_to_local(timestamp: str, clock: tuple[int, str] | None) -> datetime | None:
    """A time printed by Slurm on a cluster (its local time) as a naive local time on this machine.

    `clock` is the cluster's (`date +%s`, `date +%Y-%m-%dT%H:%M:%S`) taken at the same moment, which
    gives its UTC offset relative to this machine (rounded to 15 minutes, so clock skew does not
    count). Without it, the cluster is assumed to be in this machine's time zone. None if
    `timestamp` is not a time (e.g. "N/A").
    """
    try:
        when = datetime.fromisoformat(timestamp.strip())
    except ValueError:
        return None
    if clock:
        epoch, cluster_now = clock
        try:
            offset = datetime.fromisoformat(cluster_now) - datetime.fromtimestamp(epoch)
        except ValueError:
            return when
        when -= timedelta(minutes=round(offset.total_seconds() / 900) * 15)
    return when


def model_snapshot_expr(model_path: str, model_id: str) -> str:
    """Shell expression for the model's snapshot directory in a Hugging Face cache.

    Uses the revision in refs/main (what `hf download` fetched last), falling back to the
    newest snapshot directory.
    """
    model_dir = shell_path(f"{model_path}/{model_id}")
    return (
        f"$(d={model_dir}; if [ -f \"$d/refs/main\" ]; then printf '%s/snapshots/%s' \"$d\" \"$(cat \"$d/refs/main\")\"; "
        "else ls -dt \"$d\"/snapshots/*/ 2>/dev/null | head -1 | sed 's#/$##'; fi)"
    )


class JobManager:
    """Creates job scripts and keeps a cache of submitted jobs in ~/.local/state/slurm-llm/jobs.json."""

    STALE_THRESHOLD_DAYS = 7

    def __init__(self, job_store_path: Path | None = None, template_dir: Path | None = None):
        self.job_store_path = Path(job_store_path) if job_store_path else state_dir() / "jobs.json"
        self.template_dir = Path(template_dir) if template_dir else None
        self.last_template: TemplateSource | None = None  # the job template of the last script made

    # --- job cache ------------------------------------------------------------------------------

    def _load_jobs(self) -> dict[str, JobInfo]:
        """Cached jobs; SUBMITTED jobs older than STALE_THRESHOLD_DAYS are marked EXPIRED."""
        if not self.job_store_path.exists():
            return {}
        try:
            data = json.loads(self.job_store_path.read_text())
            jobs = {job_id: JobInfo.from_dict(d) for job_id, d in data.items()}
        except (json.JSONDecodeError, TypeError, AttributeError, OSError) as e:
            logger.warning(f"Ignoring unreadable job cache {self.job_store_path}: {e}")
            return {}
        if self._prune_stale_jobs(jobs):
            self._save_jobs(jobs)
        return jobs

    def _save_jobs(self, jobs: dict[str, JobInfo]) -> None:
        """Write the cache atomically, so concurrent readers never see a partial file."""
        self.job_store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.job_store_path.parent, prefix=".jobs-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({job_id: job.to_dict() for job_id, job in jobs.items()}, f, indent=2)
            os.replace(tmp, self.job_store_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save_job(self, job: JobInfo) -> None:
        jobs = self._load_jobs()
        jobs[job.job_id] = job
        self._save_jobs(jobs)

    def _prune_stale_jobs(self, jobs: dict[str, JobInfo]) -> bool:
        cutoff = datetime.now() - timedelta(days=self.STALE_THRESHOLD_DAYS)
        pruned = 0
        for job in jobs.values():
            if job.status != "SUBMITTED":
                continue
            try:
                if datetime.fromisoformat(job.submit_time) < cutoff:
                    job.status = "EXPIRED"
                    pruned += 1
            except (ValueError, TypeError):
                continue
        return pruned > 0

    def _batch_refresh_statuses(self, jobs: dict[str, JobInfo], ssh_clients: dict[str, SSHClient]) -> None:
        """Refresh active jobs with one squeue call per cluster (updates `jobs` in place).

        Jobs missing from a successful squeue get their final state from sacct (NOT_FOUND if it
        does not know them); jobs on clusters that could not be queried keep their cached state.
        """
        from .ssh_manager import execute_parallel

        cluster_jobs: dict[str, list[str]] = defaultdict(list)
        for job_id, job in jobs.items():
            if job.status in ACTIVE_STATES and job.cluster_name in ssh_clients:
                cluster_jobs[job.cluster_name].append(job_id)
        if not cluster_jobs:
            return

        commands = {
            name: f"squeue -j {shlex.quote(','.join(ids))} -h -o '%i|%T|%N|%L'"
            for name, ids in cluster_jobs.items()
        }
        results = execute_parallel({n: ssh_clients[n] for n in cluster_jobs}, commands, timeout=30)

        gone: dict[str, list[str]] = {}
        for cluster_name, (exit_code, stdout, stderr) in results.items():
            # squeue exits 1 with "Invalid job id" when every listed job has left the queue
            if exit_code != 0 and "invalid job id" not in stderr.lower():
                logger.warning(
                    f"Could not refresh jobs on {cluster_name} (squeue failed: "
                    f"{stderr.strip() or f'exit code {exit_code}'}); showing their last known state"
                )
                continue
            found = set()
            for line in stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 4 or parts[0].strip() not in jobs:
                    continue
                job = jobs[parts[0].strip()]
                job.status = parts[1].strip()
                job.node = parts[2].strip() or None
                job.time_left = parts[3].strip() or None
                found.add(job.job_id)
            gone[cluster_name] = [job_id for job_id in cluster_jobs[cluster_name] if job_id not in found]
        self._mark_finished(jobs, gone, ssh_clients)

    def _mark_finished(self, jobs: dict[str, JobInfo], gone: dict[str, list[str]], ssh_clients: dict[str, SSHClient]) -> None:
        """Give jobs that have left the queue their final state from Slurm accounting.

        `gone` maps clusters to job ids. The state is sacct's (COMPLETED, CANCELLED, FAILED,
        TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, ...), or NOT_FOUND when sacct does not know the job. If
        sacct fails, the jobs keep their cached state. The time left no longer applies.
        """
        from .ssh_manager import execute_parallel

        gone = {name: ids for name, ids in gone.items() if ids}
        commands = {name: f"sacct -j {shlex.quote(','.join(ids))} -X -n -P -o JobID,State" for name, ids in gone.items()}
        results = execute_parallel({n: ssh_clients[n] for n in gone}, commands, timeout=30)
        for cluster_name, ids in gone.items():
            exit_code, stdout, stderr = results.get(cluster_name, (1, "", ""))
            if exit_code != 0:
                logger.warning(
                    f"Could not get the final state of jobs on {cluster_name} (sacct failed: "
                    f"{stderr.strip() or f'exit code {exit_code}'}); showing their last known state"
                )
                continue
            final = {}
            for line in stdout.splitlines():
                job_id, _, state = line.partition("|")
                if state.split():
                    final[job_id.strip()] = state.split()[0]  # "CANCELLED by 1234" -> CANCELLED
            for job_id in ids:
                jobs[job_id].status = final.get(job_id, "NOT_FOUND")
                jobs[job_id].time_left = None

    def list_jobs(self, ssh_clients: dict[str, SSHClient] | None = None, refresh: bool = True) -> dict[str, JobInfo]:
        """Cached jobs, with active ones refreshed from the given clusters."""
        jobs = self._load_jobs()
        if refresh and ssh_clients:
            self._batch_refresh_statuses(jobs, ssh_clients)
            self._save_jobs(jobs)
        return jobs

    def get_job(self, job_id: str) -> JobInfo | None:
        return self._load_jobs().get(job_id)

    def update_job_status(self, job_id: str, ssh_client: SSHClient) -> JobInfo | None:
        jobs = self._load_jobs()
        job = jobs.get(job_id)
        if not job:
            return None
        job.status, job.node, job.time_left = ssh_client.get_job_status(job_id)
        self._save_jobs(jobs)
        return job

    def cleanup_job(self, job_id: str) -> None:
        jobs = self._load_jobs()
        if jobs.pop(job_id, None):
            self._save_jobs(jobs)

    # --- job scripts ----------------------------------------------------------------------------

    def generate_slurm_script(
        self,
        model_config: ModelConfig,
        resource_config: ResourceConfig,
        cluster_config: ClusterConfig,
        template_name: str | None = None,
        gpu_spec: dict | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Fill in a Slurm template for one vLLM server.

        `output_dir` (for logs and endpoint files) must be an absolute path: #SBATCH lines do not
        expand `~` or variables.
        """
        source = resolve_template(template_name or cluster_config.slurm_template, self.template_dir)
        self.last_template = source
        template_path = source.path
        if not cluster_config.model_path:
            raise ConfigError(
                f"cluster '{cluster_config.name}' has no model_path; set it in clusters.yaml "
                "(`sllm config cluster`)"
            )
        model_name = model_config.served_name
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", model_name):
            raise ConfigError(f"Model id '{model_config.id}' contains characters that are unsafe in a job script")

        if gpu_spec and "slurm_gpu_arg" in gpu_spec:
            gpu_slurm_arg = gpu_spec["slurm_gpu_arg"]
            num_gpus = gpu_spec.get("num_gpus", 1)
            allocation_mode = gpu_spec.get("allocation_mode", cluster_config.gpu_allocation_mode)
        else:
            gpu_slurm_arg, num_gpus, allocation_mode = "--gpus-per-node=1", 1, "per-node"

        # Tensor parallelism needs all GPUs on one node; `--gpus=N` alone may spread them out.
        if cluster_config.require_single_node or allocation_mode != "flexible":
            nodes_arg = "#SBATCH --nodes=1"
        else:
            nodes_arg = ""

        sbatch_lines = [f"#SBATCH --account={cluster_config.account}"] if cluster_config.account else []
        sbatch_lines += [f"#SBATCH {arg}" for arg in cluster_config.sbatch_args]
        for value in [resource_config.time, str(output_dir or ""), gpu_slurm_arg, *sbatch_lines]:
            if "\n" in value or "\r" in value:
                raise ConfigError("Slurm settings must not contain newlines")

        template_vars = {
            "model_id": model_snapshot_expr(cluster_config.model_path, model_config.id),
            "model_name": model_name,
            "account": cluster_config.account or "",
            "sbatch_lines": "\n".join(sbatch_lines),
            "time": resource_config.time,
            "memory": resource_config.memory,
            "num_gpus": num_gpus,
            "gpu_slurm_arg": gpu_slurm_arg,
            "nodes_arg": nodes_arg,
            "cpus_per_task": cluster_config.cpus_per_task,
            "output_dir": output_dir or cluster_config.remote_root or "~",
            "model_path": cluster_config.model_path,
            "venv_path": shell_path(f"{cluster_config.model_path}/.venv"),
            "setup_commands": "\n".join(cluster_config.setup_commands),
            "max_model_len_arg": f"--max-model-len {model_config.max_model_len}" if model_config.max_model_len else "",
            "vllm_extra_args": " ".join(shlex.quote(a) for a in model_config.vllm_extra_args),
        }
        try:
            return template_path.read_text().format(**template_vars)
        except (KeyError, IndexError, ValueError) as e:
            raise ConfigError(
                f"Template {template_path} uses an unknown placeholder {e} "
                f"(available: {', '.join(template_vars)}; write literal braces as {{{{ }}}})"
            ) from None

    def submit_job(
        self,
        model_config: ModelConfig,
        resource_config: ResourceConfig,
        cluster_config: ClusterConfig,
        ssh_client: SSHClient,
        gpu_spec: dict | None = None,
        instance: int = 0,
    ) -> str:
        """Submit a vLLM server job and record it. Returns the Slurm job id.

        The job picks its port at run time and prints it to its log, where it is found later.
        """
        remote_dir = ssh_client.resolve_dir(cluster_config.remote_root or "~")
        script = self.generate_slurm_script(
            model_config, resource_config, cluster_config, gpu_spec=gpu_spec, output_dir=remote_dir
        )
        suffix = f"_{instance}" if instance > 0 else ""
        job_id = ssh_client.submit_slurm_job(script, f"{remote_dir}/vllm_{model_config.served_name}{suffix}.slurm")
        source = self.last_template
        self.save_job(
            JobInfo(
                job_id=job_id,
                model_name=model_config.served_name,
                model_id=model_config.id,
                cluster_name=cluster_config.name,
                submit_time=datetime.now().isoformat(),
                gpu_info=gpu_spec,
                instance=instance,
                template=str(source.path),
                template_fingerprint=source.fingerprint,
            )
        )
        return job_id

    # --- finding jobs ---------------------------------------------------------------------------

    def discover_and_cache_vllm_jobs(
        self,
        ssh_clients: dict[str, SSHClient],
        config_loader=None,
        model_filter: set[str] | None = None,
    ) -> dict[str, JobInfo]:
        """Add your queued vLLM jobs (e.g. submitted from another machine) to the cache.

        Jobs are recognised by their name, vllm-<model>. Cached active jobs that have left the
        queue of a cluster that answered get their final state from sacct, as in a refresh.
        """
        from .ssh_manager import execute_parallel

        results = execute_parallel(ssh_clients, DISCOVERY_COMMAND, timeout=30)
        jobs = self._load_jobs()
        seen: set[str] = set()
        answered: set[str] = set()

        for cluster_name, (exit_code, stdout, stderr) in results.items():
            if exit_code != 0:
                logger.warning(f"Could not list jobs on {cluster_name}: {stderr.strip() or f'exit code {exit_code}'}")
                continue
            answered.add(cluster_name)
            now = re.search(r"SLURM_LLM_NOW (\d+) (\S+)", stdout)
            clock = (int(now.group(1)), now.group(2)) if now else None
            for line in stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 5:
                    continue
                job_id, job_name, status, node, time_left = parts[:5]
                submitted = cluster_time_to_local(parts[5], clock) if len(parts) > 5 else None
                seen.add(job_id)
                if not job_name.startswith(("vllm-", "vllm_")):
                    continue
                model_name = job_name[5:]
                model = config_loader.resolve_model(model_name) if config_loader else None
                if model_filter:
                    names = {model_name} | ({model.id, model.served_name} if model else set())
                    if not model_filter & names:
                        continue
                node = node if node and node != "(None)" else None
                time_left = time_left if time_left and time_left != "N/A" else None
                if job_id in jobs:
                    job = jobs[job_id]
                    job.status, job.node, job.time_left = status, node, time_left
                    if submitted:  # also corrects jobs that earlier versions recorded as submitted "now"
                        job.submit_time = submitted.isoformat()
                else:
                    jobs[job_id] = JobInfo(
                        job_id=job_id,
                        model_name=model_name,
                        model_id=model.id if model else model_name,
                        cluster_name=cluster_name,
                        submit_time=(submitted or datetime.now()).isoformat(),
                        status=status,
                        node=node,
                        time_left=time_left,
                    )

        gone: dict[str, list[str]] = defaultdict(list)
        for job_id, job in jobs.items():
            if job.status in ACTIVE_STATES and job.cluster_name in answered and job_id not in seen:
                gone[job.cluster_name].append(job_id)
        self._mark_finished(jobs, gone, ssh_clients)
        self._save_jobs(jobs)
        return jobs

    def find_running_job_by_model(
        self,
        model_name: str,
        config_loader,
        ssh_clients: dict[str, SSHClient] | None = None,
        refresh: bool = True,
        instance: int | None = None,
    ) -> JobInfo | None:
        """An active job for the model (RUNNING preferred over CONFIGURING over PENDING).

        With `ssh_clients`, only jobs on those clusters count, and the clusters' queues are
        searched when the cache has no match.
        """
        jobs = self._load_jobs()
        if refresh and ssh_clients:
            self._batch_refresh_statuses(jobs, ssh_clients)
            self._save_jobs(jobs)

        model = config_loader.resolve_model(model_name)
        names = {model_name} | ({model.id, model.served_name} if model else set())

        def matches(job: JobInfo) -> bool:
            if job.status not in {"RUNNING", "PENDING", "CONFIGURING"}:
                return False
            if ssh_clients and job.cluster_name not in ssh_clients:
                return False
            if instance is not None and job.instance != instance:
                return False
            if job.model_name in names or job.model_id in names:
                return True
            job_model = config_loader.resolve_model(job.model_name)
            return bool(job_model and job_model.id in names)

        candidates = [job for job in jobs.values() if matches(job)]
        if not candidates and ssh_clients:
            logger.info("No cached job found; searching the clusters' queues")
            jobs = self.discover_and_cache_vllm_jobs(ssh_clients, config_loader=config_loader, model_filter=names)
            candidates = [job for job in jobs.values() if matches(job)]
        if not candidates:
            return None
        priority = {"RUNNING": 0, "CONFIGURING": 1, "PENDING": 2}
        return min(candidates, key=lambda j: priority.get(j.status, 99))
