"""SSH command construction and Slurm status parsing (no ssh is run)."""

import subprocess
from pathlib import Path

import pytest

from slurm_llm.config import SlurmLLMError
from slurm_llm.ssh import NotConnectedError, SlurmUnavailableError, SSHClient, shell_path


@pytest.fixture
def client(loader):
    return SSHClient(loader.get_cluster("cluster-a"))


def _record(monkeypatch, completed, results):
    """Replace subprocess.run with a recorder returning `results` in turn."""
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        result = completed(*results.pop(0)) if results else completed(0, "", "")
        if hasattr(kwargs.get("stderr"), "write"):  # stderr redirected to a file
            kwargs["stderr"].write(result.stderr)
        return result

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_shell_path_keeps_home_and_variables_working():
    assert shell_path("~") == '"$HOME"'
    assert shell_path("~/my models") == '"$HOME/my models"'
    assert shell_path("$SCRATCH/m") == '"$SCRATCH/m"'
    assert shell_path('a"b`c') == '"a\\"b\\`c"'


def test_uses_rclusts_socket_and_destination(client, tmp_path):
    # the same socket as `rclust connect` for this host, so connections are shared
    assert client.ssh_master_socket == str(tmp_path / "sock-hpc1.example.org")
    assert client.destination == "alice@hpc1.example.org"


def test_commands_never_prompt_and_reuse_the_master(client, monkeypatch, completed):
    calls = _record(monkeypatch, completed, [(0, "out", "")])
    assert client.execute_command("hostname", input="secret", use_login_shell=False) == (0, "out", "")
    cmd, kwargs = calls[0]
    assert cmd[:1] == ["ssh"] and cmd[-2:] == ["alice@hpc1.example.org", "hostname"]
    assert "BatchMode=yes" in cmd and "ControlMaster=no" in cmd
    assert cmd[cmd.index("-S") + 1] == client.ssh_master_socket
    assert kwargs["input"] == "secret"  # secrets go over stdin, not the command line


def test_commands_run_in_a_login_shell_by_default(client, monkeypatch, completed):
    # on some clusters only the login profile puts Slurm on the PATH and sets SLURM_CONF and $SCRATCH
    calls = _record(monkeypatch, completed, [])
    client.execute_command("sbatch 'x y'")
    assert calls[0][0][-1] == "bash -l -c 'echo SLURM_LLM_OUTPUT_START; sbatch '\"'\"'x y'\"'\"''"


def test_login_shell_drops_what_the_profile_prints(client, monkeypatch, completed):
    _record(monkeypatch, completed, [(0, "Welcome to the cluster!\nSLURM_LLM_OUTPUT_START\nRUNNING|gpu3|1:00\n", "")])
    assert client.execute_command("squeue")[1] == "RUNNING|gpu3|1:00\n"
    # no marker (e.g. the profile exited early): the output is passed on unchanged
    _record(monkeypatch, completed, [(1, "oops\n", "")])
    assert client.execute_command("squeue")[1] == "oops\n"


def test_not_connected_error_names_the_fix(client, monkeypatch, completed):
    _record(monkeypatch, completed, [(255, "", "Permission denied (keyboard-interactive).")])
    code, _, err = client.execute_command("true")
    assert code == 255 and "sllm connect cluster-a" in err


def test_non_interactive_connect_failure(client, monkeypatch, completed):
    calls = _record(monkeypatch, completed, [(255, "", "Permission denied (keyboard-interactive).")])
    with pytest.raises(NotConnectedError, match="sllm connect cluster-a"):
        client.connect(interactive=False)
    cmd = calls[0][0]
    assert "-M" in cmd and "BatchMode=yes" in cmd and "ControlPersist=1h" in cmd
    assert Path(client.ssh_master_socket).parent.exists()


def test_job_status(client, monkeypatch, completed):
    _record(monkeypatch, completed, [(0, "RUNNING|gpu3|1:00:00\n", "")])
    assert client.get_job_status("7") == ("RUNNING", "gpu3", "1:00:00")

    # left the queue: ask accounting
    _record(monkeypatch, completed, [(0, "", ""), (0, "OUT_OF_MEMORY\n", "")])
    assert client.get_job_status("7") == ("OUT_OF_MEMORY", None, None)
    _record(monkeypatch, completed, [(1, "", "Invalid job id"), (0, "CANCELLED by 1000\n", "")])
    assert client.get_job_status("7")[0] == "CANCELLED"
    _record(monkeypatch, completed, [(0, "", ""), (0, "", "")])
    assert client.get_job_status("7")[0] == "NOT_FOUND"



def test_failed_status_checks_are_never_read_as_finished_jobs(client, monkeypatch, completed):
    # a busy controller or accounting database: UNKNOWN, so callers try again
    _record(monkeypatch, completed, [(1, "", "squeue: error: Socket timed out on send/recv operation")])
    assert client.get_job_status("7")[0] == "UNKNOWN"
    assert "Socket timed out" in client.last_status_error
    _record(monkeypatch, completed, [(0, "", ""), (1, "", "sacct: error: Problem talking to the database")])
    assert client.get_job_status("7")[0] == "UNKNOWN"
    # a network blip while the master connection stays up: also worth retrying
    monkeypatch.setattr(SSHClient, "has_active_connection", lambda self: True)
    _record(monkeypatch, completed, [(255, "", "Connection reset")])
    assert client.get_job_status("7")[0] == "UNKNOWN"


@pytest.mark.parametrize("error", [
    "bash: squeue: command not found",
    "squeue: fatal: Could not establish a configuration source",
])
def test_missing_slurm_is_an_error_not_a_retry(client, monkeypatch, completed, error):
    monkeypatch.setattr(SSHClient, "has_active_connection", lambda self: True)
    _record(monkeypatch, completed, [(127 if "not found" in error else 1, "", error)])
    with pytest.raises(SlurmUnavailableError, match="(?s)Could not check job 7 on cluster-a.*not been cancelled. Check that `bash -lc squeue` works on cluster-a"):
        client.get_job_status("7")


def test_lost_connection_is_an_error_not_a_retry(client, monkeypatch, completed):
    _record(monkeypatch, completed, [(255, "", "Control socket connect: No such file")])  # no master left
    with pytest.raises(NotConnectedError, match="Lost the connection to cluster-a.*sllm connect cluster-a") as e:
        client.get_job_status("7")
    assert e.value.cluster == "cluster-a"


def test_failed_status_checks_warn_once_per_reason(client, monkeypatch, completed, caplog):
    busy = (1, "", "squeue: error: Socket timed out on send/recv operation")
    _record(monkeypatch, completed, [busy, busy, (0, "RUNNING|gpu3|1:00:00\n", ""), busy])
    for _ in range(4):
        client.get_job_status("7")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    # repeated polls stay quiet; after a successful check, a new failure is reported again
    assert warnings == [
        "Could not check job status on cluster-a (squeue failed: squeue: error: Socket timed out on send/recv operation); will keep trying"
    ] * 2


def test_submit_uploads_over_stdin_and_parses_the_job_id(client, monkeypatch, completed):
    calls = _record(monkeypatch, completed, [(0, "", ""), (0, "Submitted batch job 123\n", "")])
    assert client.submit_slurm_job("#!/bin/bash\n", "/home/alice/llm/job.slurm") == "123"
    assert calls[0][1]["input"] == "#!/bin/bash\n"
    assert calls[1][0][-1].startswith("bash -l -c 'echo SLURM_LLM_OUTPUT_START; sbatch ")


def test_streaming_hides_the_shared_connection_closed_message(client, monkeypatch, completed):
    calls = _record(monkeypatch, completed, [(0, "", "")])
    assert client.execute_command_streaming("bash -l") == 0
    cmd = calls[0][0]
    assert "-tt" in cmd and "LogLevel=ERROR" in cmd and "BatchMode=yes" in cmd


SCRIPT = "#!/bin/bash\n#SBATCH --job-name=vllm-qwen\nvllm serve\n"


@pytest.fixture
def connected(monkeypatch):
    monkeypatch.setattr(SSHClient, "has_active_connection", lambda self: True)
    monkeypatch.setattr("slurm_llm.ssh.time.sleep", lambda s: None)


def test_sbatch_refusal_is_reported_without_looking_further(client, monkeypatch, completed, connected):
    err = "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"
    calls = _record(monkeypatch, completed, [(0, "", ""), (0, "", ""), (1, "", err)])
    with pytest.raises(SlurmLLMError, match="Invalid account"):
        client.submit_slurm_job(SCRIPT, "~/x.slurm")
    assert len(calls) == 3  # upload, queue before, sbatch


def test_sbatch_without_an_answer_finds_the_job_it_submitted(client, monkeypatch, completed, connected, caplog):
    timeout = "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"
    _record(monkeypatch, completed, [(0, "", ""), (0, "100\n", ""), (1, "", timeout), (0, "100\n101\n", "")])
    assert client.submit_slurm_job(SCRIPT, "~/x.slurm") == "101"  # 100 was already queued
    assert "job 101 is in the queue; using it" in caplog.text


def test_sbatch_without_an_answer_and_no_new_job(client, monkeypatch, completed, connected):
    _record(monkeypatch, completed, [(0, "", ""), (0, "", ""), (1, "", "Socket timed out")] + [(0, "", "")] * 3)
    with pytest.raises(SlurmLLMError, match="not in the queue. Check `squeue -u \\$USER`"):
        client.submit_slurm_job(SCRIPT, "~/x.slurm")


def test_sbatch_connection_lost_may_have_submitted(client, monkeypatch, completed):
    monkeypatch.setattr(SSHClient, "has_active_connection", lambda self: False)  # only asked after a 255
    _record(monkeypatch, completed, [(0, "", ""), (0, "", ""), (255, "", "Broken pipe")])
    with pytest.raises(NotConnectedError, match="may have been submitted.*will find the job"):
        client.submit_slurm_job(SCRIPT, "~/x.slurm")


def test_sbatch_missing_is_a_slurm_error(client, monkeypatch, completed, connected):
    _record(monkeypatch, completed, [(0, "", ""), (127, "", "bash: squeue: command not found"),
                                     (127, "", "bash: sbatch: command not found")])
    with pytest.raises(SlurmUnavailableError, match="sbatch: command not found. Check that `bash -lc sbatch` works on cluster-a"):
        client.submit_slurm_job(SCRIPT, "~/x.slurm")


def test_slurm_problem_names_missing_commands(client, monkeypatch, completed, connected):
    _record(monkeypatch, completed, [(0, "", ""), (0, "sacct\nscancel\n", "")])
    assert client.slurm_problem() is None
    assert client.slurm_problem() == (
        "sacct, scancel not found, even in a login shell. Check that `bash -lc sacct` works on cluster-a."
    )


def test_sbatch_refusal_with_exit_255_is_not_a_lost_answer(client, monkeypatch, completed, connected):
    # sbatch exits 255 for an invalid #SBATCH line, as ssh does when the connection drops
    err = "sbatch: error: Invalid directive found in batch script: PR"
    calls = _record(monkeypatch, completed, [(0, "", ""), (0, "", ""), (255, "", err)])
    with pytest.raises(SlurmLLMError, match="sbatch failed on cluster-a: sbatch: error: Invalid directive"):
        client.submit_slurm_job(SCRIPT, "~/x.slurm")
    assert len(calls) == 3  # no search of the queue
