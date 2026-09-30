"""Choosing a cluster and GPUs for a model.

Queue-time estimates come from rclust, which probes each candidate cluster with
`sbatch --test-only`. This module adds what is specific to serving models: how much GPU memory a
model needs, which clusters can provide it, and size-based preferences.
"""

import logging
import math
from datetime import datetime

from .config import ClusterConfig, ConfigError, ConfigLoader, ModelConfig, ResourceConfig
from .ssh_manager import SSHConnectionManager

logger = logging.getLogger(__name__)


class ClusterScheduler:
    """Picks a cluster and GPU allocation for a model."""

    def __init__(self, config_loader: ConfigLoader, conn_manager: SSHConnectionManager | None = None):
        self.config_loader = config_loader
        self.conn_manager = conn_manager or SSHConnectionManager(config_loader)
        self._rclust = None

    @property
    def rclust(self):
        """rclust's Scheduler, using this process's SSH connections for its probes."""
        if self._rclust is None:
            from rclust import Scheduler
            from rclust.config import ConfigError as RclustConfigError

            try:
                self._rclust = Scheduler(
                    config_path=str(self.config_loader.rclust_config_path()),
                    ssh_provider=self.conn_manager.get_client,
                )
            except RclustConfigError as e:
                raise ConfigError(f"rclust config: {e}") from None
        return self._rclust

    def _get_total_gpu_memory(self, model_config: ModelConfig) -> float | None:
        """GPU memory (GB) the model needs: registry override, scanned specs, else a size-based guess."""
        total = self.config_loader.get_total_gpu_memory(model_config)
        if total:
            return total
        resource = self.config_loader.load_resources().get(model_config.size)
        if resource and resource.estimated_model_memory:
            return resource.estimated_model_memory * 1.15
        logger.warning(f"No memory estimate for {model_config.id}; run `sllm scan` after downloading it")
        return None

    def select_cluster_for_model(
        self,
        model_config: ModelConfig,
        resource_config: ResourceConfig,
        model_name: str | None = None,
        explicit_cluster: str | None = None,
        cluster_map: dict[str, str] | None = None,
        exclude_clusters: list[str] | None = None,
    ) -> tuple[str, int | None]:
        """The cluster to run the model on, and its estimated queue wait in seconds (None if unknown).

        In order: `explicit_cluster`; `cluster_map[model_name]`; otherwise the clusters with
        enough GPU memory (or the size preferences in cluster_preferences.yaml), of which rclust
        picks the one expected to start the job soonest (rclust's `default_policy`).
        """
        exclude = set(exclude_clusters or [])
        clusters = self.config_loader.load_clusters()

        forced = explicit_cluster
        if not forced and cluster_map and model_name in cluster_map:
            forced = cluster_map[model_name]
            if forced in exclude:
                logger.warning(f"cluster_map cluster '{forced}' is excluded; choosing another")
                forced = None
        if forced:
            if forced in exclude:
                raise ConfigError(f"Cluster '{forced}' is in the exclude list")
            if forced not in clusters:
                raise ConfigError(f"Unknown cluster '{forced}' (configured: {', '.join(clusters)})")
            candidates = [forced]
        else:
            candidates = self._get_candidate_clusters(model_config, exclude)
            if not candidates:
                raise ConfigError("No clusters left to choose from after exclusions")

        job_specs = {name: self._job_spec(model_config, resource_config, clusters[name]) for name in candidates}
        try:
            cluster, metrics = self.rclust.select_cluster(job_spec=job_specs)
        except Exception as e:  # probing is best effort; a forced or first candidate still works
            logger.warning(f"Queue probe failed: {e}")
            cluster, metrics = None, None
        if cluster and metrics:
            wait = max(0, int((metrics.estimated_start_time - datetime.now()).total_seconds()))
            logger.info(f"Selected {cluster.name} (est. wait {wait // 60} min, {len(candidates)} candidate(s))")
            return cluster.name, wait

        errors = getattr(getattr(self._rclust, "dispatcher", None), "last_errors", None) or {}
        for name, error in errors.items():
            logger.warning(f"Queue probe on {name} failed: {error}")
        logger.warning(f"No queue estimate; using {candidates[0]}")
        return candidates[0], None

    def _job_spec(self, model_config: ModelConfig, resource_config: ResourceConfig, cluster_config: ClusterConfig):
        """The rclust JobSpec to probe with: the same GPUs, memory and time the job will request."""
        from rclust import JobSpec

        gpu = self.match_gpu_for_model(cluster_config, model_config, resource_config)
        per_node = cluster_config.gpu_allocation_mode != "flexible"
        return JobSpec(
            gpus=gpu["num_gpus"] if gpu else 1,
            gpu_type=gpu["slurm_spec"] if gpu else None,
            gpus_per_node=per_node,
            memory=f"{resource_config.memory}G",
            time=resource_config.time or "1:00:00",
            nodes=1 if (per_node or cluster_config.require_single_node) else None,
        )

    def _get_candidate_clusters(self, model_config: ModelConfig, exclude_set: set[str]) -> list[str]:
        """Clusters that can hold the model; else its size preferences; else all (minus exclusions)."""
        clusters = [c for c in self.config_loader.load_clusters() if c not in exclude_set]
        total_memory = self._get_total_gpu_memory(model_config)
        if total_memory:
            capable = self.filter_clusters_by_capability(clusters, total_memory)
            if capable:
                return capable

        preferred = self.load_cluster_preferences().get(model_config.size) or []
        if "any" in preferred:
            return clusters
        preferred = [c for c in preferred if c in clusters]
        return preferred or clusters

    def load_cluster_preferences(self) -> dict[str, list[str]]:
        try:
            return self.config_loader.load_cluster_preferences()
        except Exception as e:
            logger.warning(f"Could not load cluster preferences: {e}")
            return {}

    def filter_clusters_by_capability(self, candidate_clusters: list[str], required_model_memory: float) -> list[str]:
        """Clusters with a GPU big enough, or full GPUs to split the model over (all if none qualify)."""
        clusters = self.config_loader.load_clusters()
        compatible = []
        for name in candidate_clusters:
            config = clusters.get(name)
            if not config:
                continue
            if not config.gpu_types or any(
                g.memory >= required_model_memory or g.is_full_gpu for g in config.gpu_types
            ):
                compatible.append(name)
        return compatible or candidate_clusters

    def _get_valid_tp_size(self, min_gpus: int, num_attention_heads: int, max_gpus: int = 8) -> int:
        """Smallest tensor-parallel size >= min_gpus that divides the number of attention heads."""
        for tp_size in range(max(1, min_gpus), max_gpus + 1):
            if num_attention_heads % tp_size == 0:
                return tp_size
        next_pow2 = 2 ** math.ceil(math.log2(min_gpus)) if min_gpus > 0 else 1
        logger.warning(
            f"No tensor-parallel size >= {min_gpus} divides {num_attention_heads} attention heads; using {next_pow2}"
        )
        return next_pow2

    def match_gpu_for_model(
        self,
        cluster_config: ClusterConfig,
        model_config: ModelConfig,
        resource_config: ResourceConfig,
    ) -> dict | None:
        """The GPU type and count for the model on a cluster, with the Slurm flag to request them.

        The smallest GPU that fits the whole model, else as many of the largest full GPU as needed
        (a valid tensor-parallel size). None if the cluster lists no GPU types or memory is unknown.
        """
        if not cluster_config.gpu_types:
            return None
        total_memory = self._get_total_gpu_memory(model_config)
        if not total_memory:
            return None

        fits = [g for g in cluster_config.gpu_types if g.memory >= total_memory]
        if fits:
            best_gpu, num_gpus = min(fits, key=lambda g: g.memory), 1
        else:
            full_gpus = [g for g in cluster_config.gpu_types if g.is_full_gpu]
            if not full_gpus:
                return None
            best_gpu = max(full_gpus, key=lambda g: g.memory)
            specs = self.config_loader.get_model_specs(model_config.id)
            heads = specs.num_attention_heads if specs else 64
            num_gpus = self._get_valid_tp_size(math.ceil(total_memory / best_gpu.memory), heads)

        mode = cluster_config.gpu_allocation_mode
        flag = "--gpus" if mode == "flexible" else "--gpus-per-node"
        value = f"{best_gpu.slurm_spec}:{num_gpus}" if best_gpu.slurm_spec else str(num_gpus)
        return {
            "type": best_gpu.type,
            "memory": best_gpu.memory,
            "slurm_spec": best_gpu.slurm_spec,
            "num_gpus": num_gpus,
            "slurm_gpu_arg": f"{flag}={value}",
            "allocation_mode": mode,
            "is_full_gpu": best_gpu.is_full_gpu,
            "total_model_memory": round(total_memory, 1),
        }


def parse_gpu_override(cluster_config: ClusterConfig, gpus: str) -> dict:
    """A GPU spec from a `--gpus` value: COUNT or TYPE:COUNT (TYPE as named in the cluster config)."""
    gpu_type, _, count = gpus.rpartition(":")
    if not count.isdigit() or int(count) < 1:
        raise ConfigError(f"Invalid --gpus '{gpus}': use COUNT or TYPE:COUNT, e.g. 2 or h100:2")
    num_gpus = int(count)
    gpu = next((g for g in cluster_config.gpu_types if g.type == gpu_type), None) if gpu_type else None
    if gpu_type and gpu is None:
        known = ", ".join(g.type for g in cluster_config.gpu_types) or "none configured"
        raise ConfigError(f"GPU type '{gpu_type}' is not configured for {cluster_config.name} (known: {known})")
    mode = cluster_config.gpu_allocation_mode
    flag = "--gpus" if mode == "flexible" else "--gpus-per-node"
    slurm_type = (gpu.slurm_spec if gpu else None)
    return {
        "type": gpu_type or None,
        "memory": gpu.memory if gpu else None,
        "slurm_spec": slurm_type,
        "num_gpus": num_gpus,
        "slurm_gpu_arg": f"{flag}={slurm_type}:{num_gpus}" if slurm_type else f"{flag}={num_gpus}",
        "allocation_mode": mode,
    }
