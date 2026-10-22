"""Job scripts and the local job cache."""

import json
import re
from datetime import datetime, timedelta

import pytest

from slurm_llm.config import ConfigError, ModelConfig, ResourceConfig
from slurm_llm.job import JobInfo, JobManager

RESOURCE = ResourceConfig(memory=32, time="2:00:00")
MODEL = ModelConfig(
    id="models--Qwen--Qwen3-8B", max_model_len=8192, vllm_extra_args=['--limit-mm-per-prompt={"image":0}']
)


@pytest.fixture
def manager(tmp_path):
    return JobManager(job_store_path=tmp_path / "jobs.json")


def test_script_has_the_cluster_settings(manager, loader):
    gpu = {"slurm_gpu_arg": "--gpus-per-node=a100:2", "num_gpus": 2, "allocation_mode": "per-node"}
    script = manager.generate_slurm_script(MODEL, RESOURCE, loader.get_cluster("cluster-a"), gpu_spec=gpu, output_dir="/abs/llm")

    for line in (
        "#SBATCH --job-name=vllm-Qwen3-8B", "#SBATCH --account=proj-a", "#SBATCH --partition=gpu",
        "#SBATCH --time=2:00:00", "#SBATCH --mem=32G", "#SBATCH --nodes=1", "#SBATCH --gpus-per-node=a100:2",
        "#SBATCH --output=/abs/llm/slurm-%j.out", "module load python/3.12", 'VENV="$SCRATCH/models/.venv"',
        "--tensor-parallel-size 2", "--served-model-name Qwen3-8B",
        "--max-model-len 8192 '--limit-mm-per-prompt={\"image\":0}'", "Port selection:",
    ):
        assert line in script
    assert '"$SCRATCH/models/models--Qwen--Qwen3-8B"' in script  # snapshot lookup
    assert not re.search(r"\{[a-z_]+\}", script)  # every placeholder filled


def test_script_without_account_or_single_node(manager, loader):
    gpu = {"slurm_gpu_arg": "--gpus=l40s:4", "num_gpus": 4, "allocation_mode": "flexible"}
    script = manager.generate_slurm_script(MODEL, RESOURCE, loader.get_cluster("cluster-b"), gpu_spec=gpu, output_dir="/o")
    assert "--account" not in script
    assert "--nodes" not in script


def test_script_requires_model_path_and_a_safe_name(manager, loader):
    cluster = loader.get_cluster("cluster-a")
    cluster.model_path = None
    with pytest.raises(ConfigError, match="model_path"):
        manager.generate_slurm_script(MODEL, RESOURCE, cluster)
    cluster.model_path = "/m"
    with pytest.raises(ConfigError, match="unsafe"):
        manager.generate_slurm_script(ModelConfig(id="models--x--a;rm -rf"), RESOURCE, cluster)


def test_custom_template_with_unknown_placeholder(manager, loader, isolated_home):
    templates = isolated_home / ".config" / "slurm-llm" / "templates"
    templates.mkdir(parents=True)
    (templates / "vllm.slurm").write_text("#!/bin/bash\necho {not_a_placeholder}\n")
    with pytest.raises(ConfigError, match="not_a_placeholder"):
        manager.generate_slurm_script(MODEL, RESOURCE, loader.get_cluster("cluster-a"))


def test_submit_writes_the_script_to_the_resolved_remote_root(manager, loader, fake_ssh):
    ssh = fake_ssh(loader.get_cluster("cluster-a"))
    job_id = manager.submit_job(MODEL, RESOURCE, loader.get_cluster("cluster-a"), ssh, instance=2)

    script, path = ssh.submitted
    assert job_id == "4242"
    assert path == "/home/alice/llm/vllm_Qwen3-8B_2.slurm"
    assert "#SBATCH --output=/home/alice/llm/slurm-%j.out" in script
    job = manager.get_job("4242")
    assert (job.model_id, job.cluster_name, job.instance, job.status) == (MODEL.id, "cluster-a", 2, "SUBMITTED")
    # which template made the script, so a job can be matched to its source later
    assert job.template.endswith("templates/vllm.slurm") and len(job.template_fingerprint) == 12


def test_cache_prunes_old_submissions_and_tolerates_corruption(manager):
    old = JobInfo("1", "m", "id", "cluster-a", (datetime.now() - timedelta(days=30)).isoformat())
    manager._save_jobs({"1": old})
    assert manager._load_jobs()["1"].status == "EXPIRED"
    manager.job_store_path.write_text("{not json")
    assert manager._load_jobs() == {}


def test_cache_ignores_unknown_fields(manager):
    data = {"1": {**JobInfo("1", "m", "id", "cluster-a", datetime.now().isoformat()).to_dict(), "future_field": 1}}
    manager.job_store_path.write_text(json.dumps(data))
    assert manager.get_job("1").model_name == "m"


def _jobs(manager, *specs):
    jobs = {job_id: JobInfo(job_id, "Qwen3-8B", MODEL.id, cluster, datetime.now().isoformat(), status=status)
            for job_id, cluster, status in specs}
    manager._save_jobs(jobs)
    return jobs


def test_refresh_updates_found_jobs_and_marks_missing_ones(manager, fake_ssh):
    _jobs(manager, ("1", "cluster-a", "PENDING"), ("2", "cluster-a", "RUNNING"), ("3", "cluster-b", "RUNNING"))
    clients = {
        "cluster-a": fake_ssh(responses=[("squeue", (0, "1|RUNNING|gpu7|3:59:00\n", ""))]),
        "cluster-b": fake_ssh(responses=[("squeue", (255, "", "Connection closed"))]),
    }
    jobs = manager.list_jobs(ssh_clients=clients)
    assert (jobs["1"].status, jobs["1"].node, jobs["1"].time_left) == ("RUNNING", "gpu7", "3:59:00")
    assert jobs["2"].status == "NOT_FOUND"
    assert jobs["3"].status == "RUNNING"  # the cluster could not be asked, so nothing changes


def test_jobs_that_left_the_queue_get_their_final_state(manager, fake_ssh):
    jobs = _jobs(manager, ("1", "cluster-a", "RUNNING"), ("2", "cluster-a", "RUNNING"), ("3", "cluster-a", "PENDING"))
    for job in jobs.values():
        job.time_left = "3:00:00"
    manager._save_jobs(jobs)
    ssh = fake_ssh(responses=[
        ("squeue", (1, "", "slurm_load_jobs error: Invalid job id specified")),
        ("sacct", (0, "1|CANCELLED by 1000\n2|OUT_OF_MEMORY\n", "")),
    ])
    jobs = manager.list_jobs(ssh_clients={"cluster-a": ssh})
    assert [(jobs[i].status, jobs[i].time_left) for i in "123"] == [
        ("CANCELLED", None), ("OUT_OF_MEMORY", None), ("NOT_FOUND", None)  # 3: unknown to sacct
    ]
    assert "sacct -j 1,2,3 -X -n -P -o JobID,State" in ssh.commands[-1]


def test_refresh_and_discovery_agree_on_finished_jobs(manager, loader, fake_ssh):
    responses = [("sacct", (0, "1|FAILED\n", ""))]
    _jobs(manager, ("1", "cluster-a", "RUNNING"))
    refreshed = manager.list_jobs(ssh_clients={"cluster-a": fake_ssh(responses=[("squeue", (0, "", ""))] + responses)})
    _jobs(manager, ("1", "cluster-a", "RUNNING"))
    discovered = manager.discover_and_cache_vllm_jobs(
        {"cluster-a": fake_ssh(responses=[("squeue", (0, "", ""))] + responses)}, config_loader=loader
    )
    assert refreshed["1"].status == discovered["1"].status == "FAILED"


def test_failed_slurm_commands_keep_the_last_known_state(manager, fake_ssh, caplog):
    # e.g. squeue/sacct not on the PATH: a failed check must not look like a finished job
    _jobs(manager, ("1", "cluster-a", "RUNNING"), ("2", "cluster-b", "RUNNING"))
    clients = {
        "cluster-a": fake_ssh(responses=[("squeue", (127, "", "bash: squeue: command not found"))]),
        "cluster-b": fake_ssh(responses=[("squeue", (0, "", "")), ("sacct", (1, "", "sacct: error: no database"))]),
    }
    jobs = manager.list_jobs(ssh_clients=clients)
    assert jobs["1"].status == jobs["2"].status == "RUNNING"
    assert "squeue: command not found" in caplog.text and "sacct: error: no database" in caplog.text


def test_refresh_treats_invalid_job_id_as_gone(manager, fake_ssh):
    _jobs(manager, ("1", "cluster-a", "RUNNING"))
    clients = {"cluster-a": fake_ssh(responses=[("squeue", (1, "", "slurm_load_jobs error: Invalid job id specified"))])}
    assert manager.list_jobs(ssh_clients=clients)["1"].status == "NOT_FOUND"


def test_discovery_adds_jobs_from_other_machines(manager, loader, fake_ssh):
    _jobs(manager, ("9", "cluster-a", "RUNNING"))
    queue = "10|vllm-Qwen3-8B|PENDING|(None)|N/A\n11|train|RUNNING|n1|1:00\n"
    ssh = fake_ssh(responses=[("squeue", (0, queue, "")), ("sacct", (0, "9|TIMEOUT\n", ""))])
    jobs = manager.discover_and_cache_vllm_jobs({"cluster-a": ssh}, config_loader=loader)
    assert (jobs["10"].model_id, jobs["10"].status, jobs["10"].node) == (MODEL.id, "PENDING", None)
    assert "11" not in jobs
    assert jobs["9"].status == "TIMEOUT"  # left the queue: final state from accounting
    assert "sacct -j 9 " in ssh.commands[-1]


def test_find_running_job_prefers_running_and_matches_instance(manager, loader):
    _jobs(manager, ("1", "cluster-a", "PENDING"), ("2", "cluster-b", "RUNNING"), ("3", "cluster-a", "COMPLETED"))
    assert manager.find_running_job_by_model("qwen-3-8b", loader, refresh=False).job_id == "2"
    assert manager.find_running_job_by_model("qwen-3-8b", loader, refresh=False, instance=1) is None
    assert manager.find_running_job_by_model("llama-3.1-8b-instruct", loader, refresh=False) is None


def test_cluster_times_are_converted_to_local_time():
    from slurm_llm.job import cluster_time_to_local

    epoch = 1_800_000_000
    local_now = datetime.fromtimestamp(epoch)
    # a cluster three hours ahead of this machine, whose clock is also 40 s fast
    cluster_now = (local_now + timedelta(hours=3, seconds=40)).isoformat(timespec="seconds")
    submitted = (local_now + timedelta(hours=3) - timedelta(minutes=30)).isoformat(timespec="seconds")
    assert cluster_time_to_local(submitted, (epoch, cluster_now)) == local_now - timedelta(minutes=30)
    assert cluster_time_to_local("2026-01-02T03:04:05", None) == datetime(2026, 1, 2, 3, 4, 5)
    assert cluster_time_to_local("N/A", (epoch, cluster_now)) is None


def test_discovery_uses_the_real_submit_time(manager, loader, fake_ssh):
    now = datetime.now().replace(microsecond=0)
    cluster_now = now + timedelta(hours=-5)  # the cluster's time zone is five hours behind
    old = (cluster_now - timedelta(hours=2)).isoformat()
    queue = (
        f"10|vllm-Qwen3-8B|RUNNING|g1|1:00:00|{old}\n"
        f"SLURM_LLM_NOW {int(now.timestamp())} {cluster_now.isoformat()}\n"
    )
    _jobs(manager, ("12", "cluster-a", "RUNNING"))  # recorded by an earlier version as submitted "now"
    queue += f"12|vllm-Qwen3-8B|RUNNING|g2|1:00:00|{old}\n"
    jobs = manager.discover_and_cache_vllm_jobs(
        {"cluster-a": fake_ssh(responses=[("squeue", (0, queue, ""))])}, config_loader=loader
    )
    assert jobs["10"].submit_time == jobs["12"].submit_time == (now - timedelta(hours=2)).isoformat()
    assert "SLURM_LLM_NOW" not in jobs and len(jobs) == 2


def test_jobs_hides_old_finished_jobs_found_on_the_cluster(manager, loader, fake_ssh, monkeypatch):
    """A cancelled job submitted hours ago must not look recent (the 20-minute window of `sllm jobs`)."""
    from click.testing import CliRunner

    from slurm_llm.cli import main

    now = datetime.now().replace(microsecond=0)
    queue = f"10|vllm-Qwen3-8B|RUNNING|g1|1:00:00|{(now - timedelta(hours=3)).isoformat()}\n"
    manager.discover_and_cache_vllm_jobs({"cluster-a": fake_ssh(responses=[("squeue", (0, queue, ""))])}, loader)
    jobs = manager._load_jobs()
    jobs["10"].status = "CANCELLED"
    manager._save_jobs(jobs)
    monkeypatch.setenv("SLURM_LLM_STATE_DIR", str(manager.job_store_path.parent))
    result = CliRunner().invoke(main, ["jobs", "--no-refresh"])
    assert result.exit_code == 0 and "No jobs (use --all" in result.output
    assert "10" in CliRunner().invoke(main, ["jobs", "--no-refresh", "--all"]).output
