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


# --- what setup_venv.sh installs, and how, run against stand-ins for python, pip, virtualenv and uv ---

FAKE_PYTHON = r'''#!/bin/bash
echo "$(basename "$0") $*" >> "$LOG"
case "$1 $2" in
  "-m pip")
    if [ "$3" = config ]; then [ -n "$FAKE_PIP_CONFIG" ] && echo "$FAKE_PIP_CONFIG"; exit 0; fi
    package="${@: -1}"
    if [[ " $* " == *" --no-index "* && " $NOT_IN_WHEELHOUSE " == *" $package "* ]]; then
      echo "ERROR: No matching distribution found for $package" >&2; exit 1
    fi
    if [[ " $NEEDS_OPENCV_MODULE " == *" $package "* ]] && ! grep -q opencv "$MODULES" 2>/dev/null; then
      echo "Processing .../opencv_noinstall-9999+dummy.computecanada.tar.gz (from opencv-python-headless->vllm)"
      echo "        2. Load the opencv module: module load gcc opencv/x.y.z" >&2
      echo "ERROR: Failed to build 'opencv-noinstall' when getting requirements to build wheel" >&2; exit 1
    fi
    [ "$package" != pip ] && echo "$package" >> "$INSTALLED"; exit 0 ;;
  "-m venv") mkdir -p "$3/bin"; echo 'deactivate() { :; }' > "$3/bin/activate"; exit 0 ;;
  "-c import"*) grep -qx "${2#import }" "$INSTALLED" 2>/dev/null; exit $? ;;
esac
'''
FAKE_TOOL = (
    '#!/bin/bash\necho "$(basename "$0") $*" >> "$LOG"\n'
    '[ "$1" = --no-download ] && mkdir -p "$2/bin" && echo "deactivate() { :; }" > "$2/bin/activate"\n'
    '[ "$(basename "$0")" = module ] && echo "$*" >> "$MODULES"\nexit 0\n'
)


@pytest.fixture
def setup_run(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in [("python3", FAKE_PYTHON), ("python", FAKE_PYTHON), ("virtualenv", FAKE_TOOL), ("uv", FAKE_TOOL),
                       ("module", FAKE_TOOL)]:
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    script = (PACKAGE_DIR / "templates" / "setup_venv.sh").read_text().format(
        venv_path=f'"{tmp_path}/venv"', setup_commands=""
    )

    def run(wheelhouse_via=None, from_pypi="", not_in_wheelhouse="", uv=True, needs_opencv_module=""):
        if not uv:
            (bin_dir / "uv").unlink(missing_ok=True)
        log, installed = tmp_path / "log", tmp_path / "installed"
        log.write_text(""), installed.write_text("")
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "LOG": str(log), "INSTALLED": str(installed),
               "NOT_IN_WHEELHOUSE": not_in_wheelhouse, "SLURM_LLM_FROM_PYPI": from_pypi, "SLURM_LLM_CLUSTER": "nibi",
               "NEEDS_OPENCV_MODULE": needs_opencv_module, "MODULES": str(tmp_path / "modules")}
        if wheelhouse_via == "pip":
            env["FAKE_PIP_CONFIG"] = f"install.find-links='{wheelhouse} /elsewhere'"
        elif wheelhouse_via == "file":  # pip not loaded yet: read the file pip would read
            (tmp_path / "pip.conf").write_text(f"[install]\nfind-links = {wheelhouse} /elsewhere\n")
            env["PIP_CONFIG_FILE"] = str(tmp_path / "pip.conf")
        result = real_run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30, env=env, cwd=tmp_path)
        return result, log.read_text().splitlines()

    return run


@pytest.mark.parametrize("via", ["pip", "file"])
def test_setup_installs_from_the_clusters_wheelhouse_only(setup_run, via):
    result, log = setup_run(wheelhouse_via=via)
    assert result.returncode == 0, result.stderr
    assert "Found this cluster's package wheelhouse" in result.stdout and "pip --no-index" in result.stdout
    assert any(line.startswith("virtualenv --no-download") for line in log)
    assert "python -m pip install --no-index huggingface_hub" in log
    assert "python -m pip install --no-index vllm" in log
    assert not any(line.startswith("uv ") for line in log)  # uv would ignore the wheelhouse


def test_setup_without_a_wheelhouse_is_unchanged(setup_run):
    result, log = setup_run()
    assert result.returncode == 0, result.stderr
    assert "wheelhouse" not in result.stdout
    assert any(line.startswith("python3 -m venv") for line in log)
    assert any(line.startswith("uv pip install") and line.endswith(" vllm") for line in log)
    assert not any("--no-index" in line for line in log)


def test_a_package_missing_from_the_wheelhouse_stops_with_the_override(setup_run):
    result, log = setup_run(wheelhouse_via="pip", not_in_wheelhouse="vllm")
    assert result.returncode == 1
    assert "Could not install vllm from this cluster's wheelhouse alone" in result.stderr
    assert "sllm setup --cluster nibi --from-pypi vllm" in result.stderr
    assert "python -m pip install vllm" not in log  # never falls back to PyPI on its own


def test_from_pypi_allows_pypi_for_one_package(setup_run):
    result, log = setup_run(wheelhouse_via="pip", from_pypi="vllm", not_in_wheelhouse="vllm")
    assert result.returncode == 0, result.stderr
    assert "python -m pip install --no-index huggingface_hub" in log
    assert "python -m pip install vllm" in log


def test_from_pypi_all_ignores_the_wheelhouse(setup_run):
    result, log = setup_run(wheelhouse_via="pip", from_pypi="all")
    assert result.returncode == 0, result.stderr
    assert "wheelhouse" not in result.stdout
    assert not any("--no-index" in line for line in log)


def test_a_package_provided_as_a_module_is_loaded_and_recorded(setup_run, tmp_path):
    # Alliance's wheelhouse holds a placeholder for OpenCV whose error says which module to load
    result, log = setup_run(wheelhouse_via="pip", needs_opencv_module="vllm")
    assert result.returncode == 0, result.stderr
    assert "loading it (module load gcc opencv) and trying again" in result.stdout
    assert "module load gcc opencv" in log
    assert log.count("python -m pip install --no-index vllm") == 2
    assert (tmp_path / "venv" / ".slurm-llm-modules").read_text() == "gcc opencv\n"  # for setup_commands


def test_a_rerun_loads_the_modules_an_earlier_run_needed(setup_run, tmp_path):
    setup_run(wheelhouse_via="pip", needs_opencv_module="vllm")
    (tmp_path / "modules").write_text("")  # a new login: nothing loaded
    (tmp_path / "installed").write_text("")  # and vllm does not import without the module
    result, log = setup_run(wheelhouse_via="pip", needs_opencv_module="vllm")
    assert result.returncode == 0, result.stderr
    assert log[log.index("module load gcc opencv") - 1].startswith("python3 -m pip config")  # loaded before activating
    assert (tmp_path / "venv" / ".slurm-llm-modules").read_text() == "gcc opencv\n"
