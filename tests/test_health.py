"""Port discovery and waiting for jobs, against canned cluster responses."""

from unittest.mock import Mock

import pytest

from slurm_llm import health
from slurm_llm.health import _check_job_errors, detect_vllm_port, wait_for_job_running, wait_for_vllm_health
from slurm_llm.job import JobInfo
from slurm_llm.ssh import NotConnectedError


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(health.time, "sleep", lambda s: None)


def _job(**kw):
    return JobInfo(job_id="77", model_name="m", model_id="id", cluster_name="cluster-a", submit_time="", **kw)


@pytest.mark.parametrize("grep, line, port", [
    ("Port selection", "Port selection: preferred=30077, actual=30078", 30078),
    ("Starting vLLM", "(APIServer pid=1) INFO Starting vLLM server on http://0.0.0.0:35000", 35000),  # vLLM >= 0.19
    ("Starting vLLM", "INFO Starting vLLM API server 0 on http://0.0.0.0:35001", 35001),  # vLLM <= 0.11
])
def test_port_comes_from_the_jobs_own_log(fake_ssh, grep, line, port):
    ssh = fake_ssh(responses=[(grep, (0, line + "\n", ""))])
    assert detect_vllm_port(_job(), ssh, timeout=0) == port
    assert all("slurm-77.out" in c for c in ssh.commands)
    assert not any("curl" in c for c in ssh.commands)  # never probes other servers on the node


def test_port_falls_back_to_recorded_port_or_fails(fake_ssh):
    assert detect_vllm_port(_job(local_port=31000), fake_ssh(), timeout=0) == 31000
    with pytest.raises(RuntimeError, match="Could not find the vLLM port"):
        detect_vllm_port(_job(), fake_ssh(), timeout=0)


def _status_ssh(fake_ssh, statuses):
    ssh = fake_ssh()
    ssh.get_job_status = Mock(side_effect=[(s, n, None) for s, n in statuses])
    return ssh


def test_wait_for_running(fake_ssh):
    manager = Mock()
    job = _job()
    ssh = _status_ssh(fake_ssh, [("PENDING", None), ("UNKNOWN", None), ("CONFIGURING", "g1"), ("RUNNING", "g1")])
    assert wait_for_job_running(job, ssh, manager, show_progress=False)
    assert (job.status, job.node) == ("RUNNING", "g1")
    assert manager.save_job.called


def test_wait_for_running_never_caches_unknown(fake_ssh):
    # UNKNOWN is not an active state: cached, the job would never be refreshed (or reused) again
    manager = Mock()
    job = _job(status="PENDING")
    ssh = _status_ssh(fake_ssh, [("UNKNOWN", None), ("UNKNOWN", None), ("RUNNING", "g1")])
    assert wait_for_job_running(job, ssh, manager, show_progress=False)
    assert [c.args[0].status for c in manager.save_job.call_args_list] == ["RUNNING"]


@pytest.mark.parametrize("final", ["FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "NOT_FOUND", "CANCELLED"])
def test_wait_for_running_stops_when_the_job_ends(fake_ssh, final):
    ssh = _status_ssh(fake_ssh, [("PENDING", None), (final, None)])
    assert not wait_for_job_running(_job(), ssh, Mock(), show_progress=False)


def test_health_wait_reports_why_the_job_died(fake_ssh, monkeypatch):
    clock = iter(range(0, 10_000, 40))
    monkeypatch.setattr(health.time, "time", lambda: next(clock))
    ssh = fake_ssh(responses=[
        ("curl", (7, "", "")),
        ("tail -100", (0, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n", "")),
    ])
    ssh.get_job_status = Mock(return_value=("FAILED", None, None))
    with pytest.raises(RuntimeError, match="ended \\(status: FAILED\\).*Out of Memory"):
        wait_for_vllm_health(_job(node="g1", local_port=30077), ssh, show_progress=False)


def test_health_wait_succeeds(fake_ssh):
    ssh = fake_ssh(responses=[("curl", (0, "", ""))])
    assert wait_for_vllm_health(_job(node="g1", local_port=30077), ssh, show_progress=False)
    assert "http://g1:30077/health" in ssh.commands[-1]


def test_log_diagnosis(fake_ssh):
    kv = "ValueError: ... KV cache ... estimated maximum model length is 20480\n"
    msg = _check_job_errors(fake_ssh(responses=[("tail", (0, kv, ""))]), "77")
    assert "KV cache memory error" in msg and "Suggested max_model_len: 20480" in msg
    msg = _check_job_errors(fake_ssh(responses=[("tail", (0, "Traceback\nImportError: no vllm\n", ""))]), "77")
    assert msg.startswith("Error in the job's logs")
    assert "Could not read" in _check_job_errors(fake_ssh(), "77")


def test_wait_gives_up_after_the_status_stays_unknown(fake_ssh, monkeypatch):
    clock = iter(range(0, 100_000, 60))
    monkeypatch.setattr(health.time, "time", lambda: next(clock))
    ssh = fake_ssh()
    ssh.last_status_error = "squeue failed: Socket timed out"
    ssh.get_job_status = Mock(return_value=("UNKNOWN", None, None))
    with pytest.raises(health.StatusUnknownError, match="(?s)job 77 on cluster-a for 2 minutes.*Socket timed out.*not been cancelled"):
        wait_for_job_running(_job(), ssh, Mock(), show_progress=False)


def test_busy_timeout_is_settable(fake_ssh, monkeypatch):
    clock = iter(range(0, 100_000, 60))
    monkeypatch.setattr(health.time, "time", lambda: next(clock))
    ssh = fake_ssh()
    ssh.get_job_status = Mock(return_value=("UNKNOWN", None, None))
    with pytest.raises(health.StatusUnknownError, match="for 5 minutes"):
        wait_for_job_running(_job(), ssh, Mock(), show_progress=False, busy_timeout=300)
    monkeypatch.setenv("SLURM_LLM_BUSY_TIMEOUT", "90")
    with pytest.raises(health.StatusUnknownError, match="for 90 seconds.*SLURM_LLM_BUSY_TIMEOUT"):
        wait_for_job_running(_job(), ssh, Mock(), show_progress=False)
    monkeypatch.setenv("SLURM_LLM_BUSY_TIMEOUT", "2m")
    with pytest.raises(health.ConfigError, match="must be a number of seconds"):
        wait_for_job_running(_job(), ssh, Mock(), show_progress=False)


def test_unknown_streak_resets_when_a_check_works(fake_ssh, monkeypatch):
    clock = iter(range(0, 100_000, 90))  # each check 90 seconds apart
    monkeypatch.setattr(health.time, "time", lambda: next(clock))
    statuses = [("UNKNOWN", None), ("PENDING", None)] * 5 + [("RUNNING", "g1")]
    assert wait_for_job_running(_job(), _status_ssh(fake_ssh, statuses), Mock(), show_progress=False)


def test_health_wait_stops_when_the_connection_is_lost(fake_ssh, monkeypatch):
    ssh = fake_ssh(responses=[("curl", (255, "", "Control socket connect: No such file"))])
    ssh.has_active_connection = lambda: False
    with pytest.raises(NotConnectedError, match="Lost the connection to cluster-a while waiting for vLLM"):
        wait_for_vllm_health(_job(node="g1", local_port=30077), ssh, show_progress=False)
