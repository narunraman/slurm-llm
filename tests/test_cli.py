"""The command-line interface: fast start, clear errors, exit codes."""

import stat
import subprocess
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from slurm_llm.cli import main, parse_model_instances
from slurm_llm.job import JobInfo


@pytest.fixture
def run():
    runner = CliRunner()

    def invoke(*args, input=None):
        return runner.invoke(main, list(args), input=input, catch_exceptions=False)

    return invoke


def test_help_does_not_import_heavy_modules():
    code = (
        "import sys; from slurm_llm.cli import main\n"
        "try:\n    main(['--help'])\nexcept SystemExit:\n    pass\n"
        "print(sorted(m for m in ('openai', 'aiohttp', 'httpx', 'psutil', 'rclust.api') if m in sys.modules))"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=60)
    assert out.strip().splitlines()[-1] == "[]"


def test_parse_model_instances():
    assert parse_model_instances("qwen-3-8b") == ("qwen-3-8b", [0])
    assert parse_model_instances("qwen-3-8b:2") == ("qwen-3-8b", [2])
    assert parse_model_instances("qwen-3-8b:0-2") == ("qwen-3-8b", [0, 1, 2])
    assert parse_model_instances("qwen-3-8b:1,3") == ("qwen-3-8b", [1, 3])
    assert parse_model_instances("org:model") == ("org:model", [0])


def test_missing_cluster_config_is_one_line_and_exit_1(run, isolated_home):
    (isolated_home / ".config/rclust/config.yaml").unlink()
    result = run("jobs")
    assert result.exit_code == 1
    assert "Error: No cluster config" in result.output and "rclust config" in result.output
    assert "Traceback" not in result.output


def test_unknown_model_and_cluster(run):
    result = run("prepare", "no-such-model")
    assert result.exit_code == 1 and "not in the model registry" in result.output
    result = run("setup", "--cluster", "nope")
    assert result.exit_code == 1 and "Unknown cluster 'nope'" in result.output


def test_usage_errors_exit_2(run):
    assert run("health").exit_code == 2
    assert run("submit").exit_code == 2


def test_prepare_prepares_every_instance(run):
    job = JobInfo("1", "Qwen3-8B", "id", "cluster-a", "", node="g1", local_port=30001)
    with patch("slurm_llm.orchestration.JobOrchestrator") as orchestrator_cls:
        orchestrator = orchestrator_cls.return_value.__enter__.return_value
        orchestrator.ensure_model_ready.return_value = job
        result = run("prepare", "qwen-3-8b:0-2", "-x", "cluster-b")
    assert result.exit_code == 0, result.output
    calls = orchestrator.ensure_model_ready.call_args_list
    assert [c.kwargs["instance"] for c in calls] == [0, 1, 2]
    assert all(c.kwargs["exclude_clusters"] == ["cluster-b"] for c in calls)


def test_not_connected_cluster_is_explained(run, completed):
    def ssh(cmd, **kwargs):
        return completed(255, "", "Permission denied (keyboard-interactive).")

    with patch("subprocess.run", ssh):
        result = run("jobs", "-c", "cluster-a")
    assert result.exit_code == 0
    assert "sllm connect cluster-a" in result.output


def test_init_writes_a_cluster_entry_for_each_rclust_cluster(run, isolated_home):
    (isolated_home / ".config/slurm-llm/clusters.yaml").unlink()
    result = run("init")
    assert result.exit_code == 0, result.output
    text = (isolated_home / ".config/slurm-llm/clusters.yaml").read_text()
    assert "cluster-a:" in text and "cluster-b:" in text and "- type: a100" in text
    assert (isolated_home / ".config/slurm-llm/model_registry.yaml").exists()


def test_config_validate(run, isolated_home):
    assert run("config", "validate").exit_code == 0
    (isolated_home / ".config/slurm-llm/clusters.yaml").write_text("clusters: {}\n")
    result = run("config", "validate")
    assert result.exit_code == 1 and "model_path is not set" in result.output


def test_hf_token_is_saved_privately(run, isolated_home):
    with patch("getpass.getpass", return_value="hf_example"):
        assert run("config", "hf-token").exit_code == 0
    path = isolated_home / ".config/slurm-llm/.hf_token"
    assert path.read_text().strip() == "hf_example"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_models_filters(run):
    result = run("models", "--filter", "tags:math")
    assert result.exit_code == 0
    assert "qwen-2.5-math-7b-instruct" in result.output and "qwen-3-8b" not in result.output
    assert run("models", "--filter", "colour:red").exit_code == 1


def test_setup_explains_an_outdated_custom_template(run, isolated_home, monkeypatch):
    templates = isolated_home / ".config" / "slurm-llm" / "templates"
    templates.mkdir(parents=True)
    (templates / "setup_venv.sh").write_text("hf auth login --token '{hf_token}'\n")
    monkeypatch.setattr("slurm_llm.cli._connect", lambda cfg: object())
    result = run("setup", "-c", "cluster-a")
    assert result.exit_code == 1
    assert "unknown placeholder 'hf_token'" in result.output


class _SetupSSH:
    """A cluster for `sllm setup`: the setup script "succeeds"; `import_ok` decides the import check."""

    def __init__(self, import_ok=True):
        self.import_ok = import_ok
        self.commands, self.streamed = [], []

    def resolve_dir(self, path):
        return "/scratch/alice/models"

    def execute_command_streaming(self, command, timeout=300):
        self.streamed.append(command)
        return 0

    def execute_command(self, command, timeout=30, use_login_shell=False, input=None):
        self.commands.append((command, use_login_shell))
        if "import vllm" in command:
            return (0, "", "") if self.import_ok else (1, "", "Traceback ...\nModuleNotFoundError: No module named 'vllm'\n")
        if command.startswith("df"):
            return 0, "lustre-mds1.example.org@o2ib:/scratch/very/long/filesystem 1.0P 400T 600T 40% /scratch\n", ""
        return 0, "", ""


def test_setup_reports_failure_when_vllm_is_not_importable(run, monkeypatch):
    ssh = _SetupSSH(import_ok=False)
    monkeypatch.setattr("slurm_llm.cli._connect", lambda cfg: ssh)
    result = run("setup", "-c", "cluster-a")
    assert result.exit_code == 1
    assert "No module named 'vllm'" in result.output and "ready" not in result.output
    # checked the way jobs use it: after the cluster's setup commands, in a login shell
    check, login = next(c for c in ssh.commands if "import vllm" in c[0])
    assert login and check.startswith("module load python/3.12\n")
    assert '"$SCRATCH/models/.venv"/bin/python -c' in check


def test_setup_output(run, monkeypatch):
    ssh = _SetupSSH()
    monkeypatch.setattr("slurm_llm.cli._connect", lambda cfg: ssh)
    result = run("setup", "-c", "cluster-a")
    assert result.exit_code == 0, result.output
    assert "ready" in result.output
    # the installer says itself when it installs vLLM; nothing is announced when it is already there
    assert "Installing vLLM" not in result.output
    assert "Installing vLLM (the first time can take 10 minutes)" in ssh.streamed[0]
    # the disk column is a short summary, not the (long, truncated) df line
    assert "600T free of 1.0P (40% used)" in result.output


def test_disk_summary():
    from slurm_llm.cli import disk_summary

    assert disk_summary("fs 1.0T 400G 600G 40% /scratch\n") == "600G free of 1.0T (40% used)"
    assert disk_summary("df: no such file\n") == "df: no such file"


def test_kill_cancels_forwards_left_by_the_stopped_proxy(run, monkeypatch, completed):
    import psutil

    from slurm_llm.tunnels import TunnelHandle, TunnelRegistry

    dead = subprocess.Popen(["true"])
    dead.wait()
    TunnelRegistry().register_tunnel(
        TunnelHandle(30000, "30000:g1:35000", "hpc1.example.org", "/tmp/sock", (), dead.pid), "m", "cluster-a"
    )

    class Proxy:
        pid = 999999
        info = {"pid": 999999, "cmdline": ["python", "/usr/bin/sllm", "proxy"]}
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            pass

    proxy = Proxy()
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [proxy])
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or completed(0))
    result = run("kill")
    assert result.exit_code == 0, result.output
    assert proxy.terminated and "cancelled 1 leftover port forward" in result.output
    assert calls[0][-4:] == ["cancel", "-L", "30000:g1:35000", "hpc1.example.org"]


def test_init_reads_gpu_types_under_resources(run, isolated_home):
    (isolated_home / ".config/rclust/config.yaml").write_text(
        "clusters:\n  c1:\n    resources:\n      gpu_types: [{type: h100, slurm_spec: h100}]\n"
    )
    (isolated_home / ".config/slurm-llm/clusters.yaml").unlink()
    assert run("init").exit_code == 0
    assert "- type: h100" in (isolated_home / ".config/slurm-llm/clusters.yaml").read_text()


def test_config_validate_warns_about_clusters_missing_from_either_config(run, isolated_home):
    clusters_yaml = isolated_home / ".config/slurm-llm/clusters.yaml"
    clusters_yaml.write_text(clusters_yaml.read_text().replace("  cluster-b:", "  cluster-c:"))
    output = " ".join(run("config", "validate").output.split())
    assert "cluster 'cluster-c' is not in rclust's config" in output
    assert "cluster 'cluster-b' (from rclust's config) has no entry in clusters.yaml" in output
