"""SSH command construction and Slurm status parsing (no ssh is run)."""

import subprocess
from pathlib import Path

import pytest

from slurm_llm.ssh import NotConnectedError, SSHClient, shell_path


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

    # the cluster could not be asked at all
    _record(monkeypatch, completed, [(255, "", "Connection reset")])
    assert client.get_job_status("7")[0] == "UNKNOWN"

    # a failed check is not a finished job (e.g. Slurm not on the PATH, or no SLURM_CONF)
    _record(monkeypatch, completed, [(127, "", "bash: squeue: command not found")])
    assert client.get_job_status("7")[0] == "UNKNOWN"
    _record(monkeypatch, completed, [(1, "", "squeue: fatal: Could not establish a configuration source")])
    assert client.get_job_status("7")[0] == "UNKNOWN"
    _record(monkeypatch, completed, [(0, "", ""), (1, "", "sacct: error: Problem talking to the database")])
    assert client.get_job_status("7")[0] == "UNKNOWN"


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
