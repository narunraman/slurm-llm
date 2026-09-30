"""Shared fixtures. Every test runs against a temporary home directory and never runs ssh."""

import subprocess
from pathlib import Path

import pytest

RCLUST_CONFIG = """\
clusters:
  cluster-a:
    host: hpc1.example.org
    user: alice
    account: proj-a
    gpu_types:
      - type: a100
        slurm_spec: a100
      - type: a100_3g
        slurm_spec: a100_3g.40gb
  cluster-b:
    host: hpc2.example.org
"""

CLUSTERS_YAML = """\
clusters:
  cluster-a:
    remote_root: ~/llm
    model_path: $SCRATCH/models
    setup_commands:
      - module load python/3.12
    sbatch_args: [--partition=gpu]
    gpu_types:
      - type: a100
        memory: 80
      - type: a100_3g
        memory: 40
        is_full_gpu: false
  cluster-b:
    model_path: /data/models
    gpu_allocation_mode: flexible
    gpu_types:
      - type: l40s
        memory: 48
        slurm_spec: l40s
"""


class _NoSubprocess:
    """Stands in for subprocess.run: tests must mock the commands they expect (so ssh never runs)."""

    def __call__(self, cmd, *args, **kwargs):
        raise AssertionError(f"unexpected subprocess call in a test: {cmd}")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """A fresh home with rclust and slurm-llm configs; no real config or SSH is ever used."""
    home = tmp_path / "home"
    (home / ".config" / "rclust").mkdir(parents=True)
    (home / ".config" / "slurm-llm").mkdir(parents=True)
    (home / ".config" / "rclust" / "config.yaml").write_text(RCLUST_CONFIG)
    (home / ".config" / "slurm-llm" / "clusters.yaml").write_text(CLUSTERS_YAML)
    monkeypatch.setenv("HOME", str(home))
    import os

    for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "RCLUST_CONFIG", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    for var in list(os.environ):
        if var.startswith("SLURM_LLM_"):
            monkeypatch.delenv(var)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    # rclust derives the control socket path from `ssh -G`; use a fixed path instead
    import rclust.ssh

    monkeypatch.setattr(
        rclust.ssh.SSHClient, "_get_control_master_path", lambda self: str(tmp_path / f"sock-{self.host}")
    )
    monkeypatch.setattr(subprocess, "run", _NoSubprocess())
    return home


@pytest.fixture
def loader():
    from slurm_llm.config import ConfigLoader

    return ConfigLoader()


class FakeSSH:
    """An SSHClient double: answers commands from a list of (substring, (code, stdout, stderr))."""

    def __init__(self, cluster_config=None, responses=None):
        from slurm_llm.config import ClusterConfig

        self.config = cluster_config or ClusterConfig(name="cluster-a", host="hpc1.example.org", remote_root="~/llm")
        self.responses = list(responses or [])
        self.commands = []
        self.inputs = []
        self.destination = self.config.host
        self.ssh_master_socket = "/tmp/sock"

    def execute_command(self, command, timeout=30, use_login_shell=False, input=None):
        self.commands.append(command)
        self.inputs.append(input)
        for needle, result in self.responses:
            if needle in command:
                return result
        return 0, "", ""

    def get_remote_slurm_log_path(self, job_id):
        return f"{self.config.remote_root}/slurm-{job_id}.out"

    def get_remote_slurm_error_path(self, job_id):
        return f"{self.config.remote_root}/slurm-{job_id}.err"

    def resolve_dir(self, path):
        return "/home/alice/llm"

    def submit_slurm_job(self, script, remote_script_path):
        self.submitted = (script, remote_script_path)
        return "4242"


@pytest.fixture
def fake_ssh():
    return FakeSSH


@pytest.fixture
def completed():
    """Build a subprocess.CompletedProcess."""

    def make(returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)

    return make
