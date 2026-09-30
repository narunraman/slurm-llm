"""Choosing clusters and GPUs."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from slurm_llm.config import ConfigError, ModelConfig, ModelSpecs, ResourceConfig
from slurm_llm.scheduler import ClusterScheduler, parse_gpu_override

RESOURCE = ResourceConfig(memory=32, time="2:00:00", estimated_model_memory=20)


def _model(memory_gb=None, size="small"):
    return ModelConfig(id="models--org--Model", size=size, model_memory_gb=memory_gb)


@pytest.fixture
def scheduler(loader):
    s = ClusterScheduler(loader, conn_manager=Mock())
    s._rclust = Mock()
    s._rclust.dispatcher.last_errors = {}
    return s


def test_smallest_gpu_that_fits(scheduler, loader):
    gpu = scheduler.match_gpu_for_model(loader.get_cluster("cluster-a"), _model(30), RESOURCE)
    assert (gpu["type"], gpu["num_gpus"], gpu["slurm_gpu_arg"]) == ("a100_3g", 1, "--gpus-per-node=a100_3g.40gb:1")


def test_large_model_is_split_over_full_gpus_with_a_valid_tensor_parallel_size(scheduler, loader):
    model = _model(250)
    loader.save_model_specs(ModelSpecs(
        model_id=model.id, model_memory_gb=100, num_hidden_layers=80, num_attention_heads=64,
        num_key_value_heads=8, head_dim=128, dtype_bytes=2, max_position_embeddings=8192,
    ))
    gpu = scheduler.match_gpu_for_model(loader.get_cluster("cluster-a"), model, RESOURCE)
    # 250 GB needs 4 x 80 GB (3 would do, but 3 does not divide 64 heads); MIG slices are never combined
    assert (gpu["type"], gpu["num_gpus"], gpu["slurm_gpu_arg"]) == ("a100", 4, "--gpus-per-node=a100:4")


def test_flexible_allocation_uses_gpus_flag(scheduler, loader):
    gpu = scheduler.match_gpu_for_model(loader.get_cluster("cluster-b"), _model(40), RESOURCE)
    assert gpu["slurm_gpu_arg"] == "--gpus=l40s:1"


def test_size_estimate_is_used_for_unscanned_models(scheduler, loader):
    gpu = scheduler.match_gpu_for_model(loader.get_cluster("cluster-a"), _model(), RESOURCE)
    # packaged resources.yaml: small models are assumed to need 20 GB (+15%)
    assert gpu["type"] == "a100_3g" and gpu["total_model_memory"] == 23.0


def test_explicit_cluster_is_probed_with_the_real_request(scheduler):
    start = datetime.now() + timedelta(minutes=30)
    scheduler._rclust.select_cluster.return_value = (SimpleNamespace(name="cluster-a"), SimpleNamespace(estimated_start_time=start))

    name, wait = scheduler.select_cluster_for_model(_model(30), RESOURCE, explicit_cluster="cluster-a")

    assert name == "cluster-a" and 1700 < wait <= 1800
    spec = scheduler._rclust.select_cluster.call_args.kwargs["job_spec"]["cluster-a"]
    assert (spec.gpus, spec.gpu_type, spec.gpus_per_node, spec.nodes, spec.memory, spec.time) == (
        1, "a100_3g.40gb", True, 1, "32G", "2:00:00"
    )


def test_candidates_are_clusters_with_enough_memory(scheduler, loader):
    scheduler._rclust.select_cluster.return_value = (None, None)
    name, wait = scheduler.select_cluster_for_model(_model(30), RESOURCE, exclude_clusters=["cluster-b"])
    assert (name, wait) == ("cluster-a", None)  # probe failed: first candidate, unknown wait
    assert set(scheduler._rclust.select_cluster.call_args.kwargs["job_spec"]) == {"cluster-a"}


def test_cluster_map_and_errors(scheduler):
    scheduler._rclust.select_cluster.return_value = (None, None)
    assert scheduler.select_cluster_for_model(_model(30), RESOURCE, model_name="m", cluster_map={"m": "cluster-b"})[0] == "cluster-b"
    with pytest.raises(ConfigError, match="exclude list"):
        scheduler.select_cluster_for_model(_model(30), RESOURCE, explicit_cluster="cluster-a", exclude_clusters=["cluster-a"])
    with pytest.raises(ConfigError, match="Unknown cluster"):
        scheduler.select_cluster_for_model(_model(30), RESOURCE, explicit_cluster="nope")
    with pytest.raises(ConfigError, match="No clusters left"):
        scheduler.select_cluster_for_model(_model(30), RESOURCE, exclude_clusters=["cluster-a", "cluster-b"])


def test_probe_errors_fall_back_to_first_candidate(scheduler):
    scheduler._rclust.select_cluster.side_effect = RuntimeError("probe exploded")
    assert scheduler.select_cluster_for_model(_model(30), RESOURCE, explicit_cluster="cluster-b") == ("cluster-b", None)


def test_size_preferences_apply_when_memory_is_unknown(scheduler, loader):
    (loader.config_dir / "cluster_preferences.yaml").write_text("cluster_preferences:\n  odd: [cluster-b]\n")
    scheduler._rclust.select_cluster.return_value = (None, None)
    assert scheduler.select_cluster_for_model(_model(size="odd"), RESOURCE)[0] == "cluster-b"


def test_gpu_override(loader):
    cluster = loader.get_cluster("cluster-a")
    assert parse_gpu_override(cluster, "2")["slurm_gpu_arg"] == "--gpus-per-node=2"
    spec = parse_gpu_override(cluster, "a100_3g:1")
    assert (spec["slurm_gpu_arg"], spec["memory"]) == ("--gpus-per-node=a100_3g.40gb:1", 40)
    with pytest.raises(ConfigError, match="not configured"):
        parse_gpu_override(cluster, "h100:2")
    with pytest.raises(ConfigError, match="Invalid --gpus"):
        parse_gpu_override(cluster, "two")


def test_rclust_probes_through_slurm_llm_connections(loader, fake_ssh):
    """rclust's Scheduler reads the same config and probes over the SSHClients slurm-llm hands it."""
    import time

    now = datetime.now().replace(microsecond=0)
    soon = (now + timedelta(minutes=10)).isoformat()
    later = (now + timedelta(hours=5)).isoformat()
    clock = f"RCLUST_NOW {int(time.time())} {now.isoformat()}"
    clients = {
        "cluster-a": fake_ssh(loader.get_cluster("cluster-a"), [("sbatch --test-only", (0, clock, f"sbatch: Job 1 to start at {later}"))]),
        "cluster-b": fake_ssh(loader.get_cluster("cluster-b"), [("sbatch --test-only", (0, clock, f"sbatch: Job 2 to start at {soon}"))]),
    }
    conn_manager = Mock()
    conn_manager.get_client.side_effect = clients.__getitem__
    scheduler = ClusterScheduler(loader, conn_manager)

    name, wait = scheduler.select_cluster_for_model(_model(30), RESOURCE)

    assert name == "cluster-b" and 500 < wait <= 600
    probe = next(c for c in clients["cluster-b"].commands if "--test-only" in c)
    assert "--gpus=l40s:1" in probe and "--mem=32G" in probe and "--time=2:00:00" in probe
    probe_a = next(c for c in clients["cluster-a"].commands if "--test-only" in c)
    assert "--gpus-per-node=a100_3g.40gb:1" in probe_a and "--account=proj-a" in probe_a
