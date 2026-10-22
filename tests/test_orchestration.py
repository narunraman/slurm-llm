"""The prepare workflow, with the cluster side mocked."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from slurm_llm import orchestration
from slurm_llm.config import ConfigError, SlurmLLMError
from slurm_llm.job import JobInfo, JobManager
from slurm_llm.orchestration import JobOrchestrator, OrchestratorConfig
from slurm_llm.ssh import NotConnectedError, SlurmUnavailableError, StatusUnknownError


@pytest.fixture
def orchestrator(tmp_path, fake_ssh):
    o = JobOrchestrator(config=OrchestratorConfig(show_progress=False))
    o.job_manager = JobManager(job_store_path=tmp_path / "jobs.json")
    o.conn_manager = Mock()
    o.conn_manager.get_client.side_effect = lambda name: fake_ssh(o.config_loader.get_cluster(name))
    o.scheduler = Mock()
    return o


def _job(job_id="5", status="SUBMITTED", **kw):
    return JobInfo(job_id, "Qwen3-8B", "models--Qwen--Qwen3-8B", "cluster-a", datetime.now().isoformat(), status=status, **kw)


def test_unknown_model_fails_before_contacting_clusters(orchestrator):
    with pytest.raises(ConfigError, match="not in the model registry"):
        orchestrator.ensure_model_ready("no-such-model")
    orchestrator.conn_manager.get_client.assert_not_called()


def test_reuses_a_running_job(orchestrator):
    orchestrator.job_manager.find_running_job_by_model = Mock(return_value=_job(status="RUNNING", node="g1", local_port=30005))
    with patch.object(orchestration, "wait_for_vllm_health") as healthy, \
         patch.object(orchestration, "query_model_info", return_value={"max_model_len": 40960}):
        job = orchestrator.ensure_model_ready("qwen-3-8b")
    assert (job.job_id, job.max_model_len) == ("5", 40960)
    healthy.assert_called_once()
    orchestrator.scheduler.select_cluster_for_model.assert_not_called()


def test_submits_and_waits_for_a_new_job(orchestrator):
    orchestrator.scheduler.select_cluster_for_model.return_value = ("cluster-a", 120)
    orchestrator.scheduler.match_gpu_for_model.return_value = None

    def running(job, *args, **kwargs):
        job.status, job.node = "RUNNING", "g2"
        return True

    with patch.object(orchestration, "wait_for_job_running", side_effect=running) as wait, \
         patch.object(orchestration, "detect_vllm_port", return_value=30042), \
         patch.object(orchestration, "wait_for_vllm_health"), \
         patch.object(orchestration, "query_model_info", return_value=None):
        job = orchestrator.ensure_model_ready("qwen-3-8b", instance=1)

    assert (job.job_id, job.node, job.local_port, job.instance) == ("4242", "g2", 30042, 1)
    assert wait.call_args.kwargs["estimated_wait"] == 120
    assert orchestrator.job_manager.get_job("4242").local_port == 30042


def test_job_that_ends_while_queued_is_reported_with_its_logs(orchestrator, fake_ssh):
    orchestrator.scheduler.select_cluster_for_model.return_value = ("cluster-a", None)
    orchestrator.scheduler.match_gpu_for_model.return_value = None
    ssh = fake_ssh(responses=[("tail", (0, "sbatch: error: Invalid account\n", ""))])
    orchestrator.conn_manager.get_client.side_effect = lambda name: ssh
    with patch.object(orchestration, "wait_for_job_running", return_value=False):
        with pytest.raises(SlurmLLMError, match="(?s)ended before it started running.*Invalid account"):
            orchestrator.ensure_model_ready("qwen-3-8b")


@pytest.mark.parametrize("failure", [
    NotConnectedError("Lost the connection to cluster-a", "cluster-a"),
    SlurmUnavailableError("squeue: command not found"),
    StatusUnknownError("Could not check job 5 for 10 minutes"),
])
def test_a_job_that_cannot_be_checked_is_never_replaced(orchestrator, monkeypatch, failure):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    orchestrator.job_manager.find_running_job_by_model = Mock(return_value=_job(status="PENDING"))
    with patch.object(orchestration, "wait_for_job_running", side_effect=failure):
        with pytest.raises(type(failure)):
            orchestrator.ensure_model_ready("qwen-3-8b")
    orchestrator.scheduler.select_cluster_for_model.assert_not_called()  # no second job


def test_prepare_logs_in_again_and_resumes_the_same_job(orchestrator, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    orchestrator.job_manager.find_running_job_by_model = Mock(return_value=_job(status="PENDING"))
    outcomes = [NotConnectedError("Lost the connection to cluster-a", "cluster-a"), None]

    def wait(job, *args, **kwargs):
        if outcome := outcomes.pop(0):
            raise outcome
        job.status, job.node = "RUNNING", "g1"
        return True

    with patch.object(orchestration, "wait_for_job_running", side_effect=wait), \
         patch.object(orchestration, "detect_vllm_port", return_value=30005), \
         patch.object(orchestration, "wait_for_vllm_health"), \
         patch.object(orchestration, "query_model_info", return_value=None):
        job = orchestrator.ensure_model_ready("qwen-3-8b")
    orchestrator.conn_manager.reconnect.assert_called_once_with("cluster-a")
    assert job.job_id == "5"
    orchestrator.scheduler.select_cluster_for_model.assert_not_called()


def test_without_a_terminal_a_lost_connection_is_reported(orchestrator, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    orchestrator.job_manager.find_running_job_by_model = Mock(return_value=_job(status="PENDING"))
    with patch.object(orchestration, "wait_for_job_running",
                      side_effect=NotConnectedError("Lost the connection to cluster-a", "cluster-a")):
        with pytest.raises(NotConnectedError, match="Lost the connection"):
            orchestrator.ensure_model_ready("qwen-3-8b")
    orchestrator.conn_manager.reconnect.assert_not_called()
