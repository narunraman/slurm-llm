"""The packaged templates stay valid shell scripts whatever the placeholders are filled with."""

import re
from subprocess import run as real_run  # imported before conftest replaces subprocess.run

import pytest

from slurm_llm.config import PACKAGE_DIR, ModelConfig, ResourceConfig
from slurm_llm.job import JobManager

# Several commands, with quotes, so that a placeholder filled in inside a comment would spill onto
# uncommented lines (as {setup_commands} in a header comment once did). The last command stops the
# script, so running it executes nothing but the setup commands.
SETUP_COMMANDS = ["echo 'sentinel-1'", 'echo "sentinel-2"', "exit 7"]
TEMPLATES = sorted(p.name for p in (PACKAGE_DIR / "templates").iterdir() if p.suffix in (".sh", ".slurm"))


def _render(name, loader, tmp_path):
    cluster = loader.get_cluster("cluster-a")
    cluster.setup_commands = SETUP_COMMANDS
    if name.endswith(".slurm"):
        gpu = {"slurm_gpu_arg": "--gpus-per-node=a100:2", "num_gpus": 2, "allocation_mode": "per-node"}
        return JobManager(job_store_path=tmp_path / "jobs.json").generate_slurm_script(
            ModelConfig(id="models--Qwen--Qwen3-8B"), ResourceConfig(memory=32, time="1:00:00"),
            cluster, template_name=name, gpu_spec=gpu, output_dir="/abs/llm",
        )
    # as `sllm setup` fills it in
    return (PACKAGE_DIR / "templates" / name).read_text().format(
        venv_path='"$SCRATCH/models/.venv"', setup_commands="\n".join(SETUP_COMMANDS)
    )


def test_every_template_is_tested():
    assert {"vllm.slurm", "setup_venv.sh"} <= set(TEMPLATES)


@pytest.mark.parametrize("name", TEMPLATES)
def test_rendered_template_is_valid_bash(name, loader, tmp_path):
    script = _render(name, loader, tmp_path)
    result = real_run(["bash", "-n"], input=script, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", TEMPLATES)
def test_placeholders_are_not_filled_in_inside_comments(name, loader, tmp_path):
    script = _render(name, loader, tmp_path)
    comments = [line for line in script.splitlines() if line.lstrip().startswith("#")
                and not line.startswith(("#!", "#SBATCH"))]
    for line in comments:
        assert "sentinel" not in line and "exit 7" not in line and "$SCRATCH" not in line, line
    # in the template itself, placeholder names in comments are escaped ({{name}})
    for line in (PACKAGE_DIR / "templates" / name).read_text().splitlines():
        if line.lstrip().startswith("#") and not line.startswith(("#!", "#SBATCH")):
            assert not re.search(r"(?<!\{)\{[a-z_]+\}(?!\})", line), line
    # the setup commands are each on their own uncommented line
    for command in SETUP_COMMANDS:
        assert command in script.splitlines()


@pytest.mark.parametrize("name", TEMPLATES)
def test_setup_commands_run_first_and_exactly_once(name, loader, tmp_path):
    script = _render(name, loader, tmp_path)
    # no login shell (-l): the site profile of this machine does not matter here
    result = real_run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert (result.returncode, result.stdout) == (7, "sentinel-1\nsentinel-2\n"), result.stderr


def test_setup_script_keeps_set_e_and_the_venv(loader, tmp_path):
    lines = _render("setup_venv.sh", loader, tmp_path).splitlines()
    assert "set -e" in lines
    assert 'VENV="$SCRATCH/models/.venv"' in lines
    assert lines.index("set -e") < lines.index(SETUP_COMMANDS[0])
