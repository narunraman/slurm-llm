"""Submitting vLLM jobs from Python, e.g. to make other Slurm jobs depend on them."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ConfigLoader
from .job import JobManager
from .scheduler import ClusterScheduler, parse_gpu_override
from .ssh import shell_path
from .ssh_manager import SSHConnectionManager


@dataclass
class SubmitResult:
    """Jobs submitted by `submit_model`."""

    model_name: str  # served model name, e.g. "Qwen2.5-7B-Instruct"
    cluster_name: str
    job_ids: list[str] = field(default_factory=list)
    nodes_dir: str = ""  # where the jobs write <job_id>.node files (for RemoteLLM(local=True))
    remote_root: str = ""

    @property
    def after_any_dependency(self) -> str:
        """Slurm dependency: start after all these jobs end, in any state."""
        return f"afterany:{':'.join(self.job_ids)}" if self.job_ids else ""

    @property
    def after_all_dependency(self) -> str:
        """Slurm dependency: start after all these jobs complete successfully."""
        return f"afterok:{':'.join(self.job_ids)}" if self.job_ids else ""


def submit_model(
    model: str,
    instances: int = 1,
    cluster: str | None = None,
    time: str | None = None,
    gpus: str | None = None,
    config_dir: str | None = None,
) -> SubmitResult:
    """Submit `instances` vLLM jobs for a model without waiting for them.

    Args:
        model: model name from the registry (e.g. "qwen-2.5-7b-instruct")
        instances: number of copies to run
        cluster: cluster to use (default: chosen by queue estimate)
        time: Slurm time limit (default: from resources.yaml)
        gpus: GPU override, COUNT or TYPE:COUNT (default: sized to the model)
        config_dir: slurm-llm config directory (default: ~/.config/slurm-llm)

    Example:
        >>> result = submit_model("qwen-2.5-7b-instruct", instances=3)
        >>> result.after_any_dependency   # 'afterany:123:124:125'
    """
    config_loader = ConfigLoader(config_dir)
    conn_manager = SSHConnectionManager(config_loader)
    scheduler = ClusterScheduler(config_loader, conn_manager)
    job_manager = JobManager()

    model_config = config_loader.require_model(model)
    resource_config = config_loader.resource_for(model_config)
    if time:
        resource_config.time = time
    if not cluster:
        cluster, _ = scheduler.select_cluster_for_model(model_config, resource_config, model_name=model)
    cluster_config = config_loader.get_cluster(cluster)
    gpu_spec = (
        parse_gpu_override(cluster_config, gpus)
        if gpus
        else scheduler.match_gpu_for_model(cluster_config, model_config, resource_config)
    )

    ssh_client = conn_manager.get_client(cluster)
    job_ids = [
        job_manager.submit_job(model_config, resource_config, cluster_config, ssh_client, gpu_spec=gpu_spec, instance=i)
        for i in range(instances)
    ]
    remote_root = ssh_client.resolve_dir(cluster_config.remote_root or "~")
    nodes_dir = f"{remote_root}/manifests/{model_config.served_name}/nodes"
    ssh_client.execute_command(f"mkdir -p {shell_path(nodes_dir)}")
    return SubmitResult(
        model_name=model_config.served_name,
        cluster_name=cluster,
        job_ids=job_ids,
        nodes_dir=nodes_dir,
        remote_root=remote_root,
    )
