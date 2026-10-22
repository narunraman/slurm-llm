#!/bin/bash -l
# Creates the Python environment that jobs run vLLM from; run by `sllm setup` on the
# login node. It is safe to run again. slurm-llm fills in:
#   {{venv_path}}       where to create the venv ({{model_path}}/.venv)
#   {{setup_commands}}  the setup_commands of the cluster from clusters.yaml (e.g. module loads)
# and sets SLURM_LLM_CLUSTER (the cluster's name) and SLURM_LLM_FROM_PYPI (packages to install
# from PyPI even on a cluster with its own wheelhouse: `sllm setup --from-pypi`). Modules loaded
# here for a wheelhouse are listed in $VENV/.slurm-llm-modules, for `sllm setup` to add to
# setup_commands (jobs need them too).
# Copy this file to ~/.config/slurm-llm/templates/ to customise it (set setup_template in clusters.yaml
# to use a different name).

set -e

{setup_commands}

VENV={venv_path}
FROM_PYPI=" ${{SLURM_LLM_FROM_PYPI:-}} "

# Some clusters (e.g. Alliance Canada) provide their own builds of Python packages in a local
# "wheelhouse" and list it in pip's settings (find-links). Installing from PyPI there mixes the
# two and can fail, so install only from the wheelhouse (pip --no-index), as these sites ask.
WHEELHOUSE=$(python3 -m pip config list 2>/dev/null | sed -n "s/^[^=]*find-links='\([^ ']*\).*/\1/p" | head -1)
if [ -z "$WHEELHOUSE" ] && [ -r "${{PIP_CONFIG_FILE:-}}" ]; then  # pip itself may not be loaded yet
    WHEELHOUSE=$(sed -n "s/^find-links *= *\([^ ]*\).*/\1/p" "$PIP_CONFIG_FILE" | head -1)
fi
if [ -n "$WHEELHOUSE" ] && [ -d "$WHEELHOUSE" ] && [ "$FROM_PYPI" != " all " ]; then
    echo "Found this cluster's package wheelhouse ($WHEELHOUSE, from pip's settings); installing from it only (pip --no-index)"
else
    WHEELHOUSE=""
fi

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV..."
    if [ -n "$WHEELHOUSE" ] && command -v virtualenv >/dev/null 2>&1; then
        virtualenv --no-download "$VENV"
    else
        python3 -m venv "$VENV"
    fi
else
    echo "Virtual environment already exists at $VENV"
fi
if [ -s "$VENV/.slurm-llm-modules" ]; then  # modules an earlier run needed: part of this environment
    module load $(cat "$VENV/.slurm-llm-modules")
fi
source "$VENV/bin/activate"

if [ -n "$WHEELHOUSE" ]; then
    # uv ignores pip's settings, so it would not use the wheelhouse
    python -m pip install --quiet --no-index --upgrade pip || true
    INSTALL="python -m pip install --no-index"
elif command -v uv >/dev/null 2>&1; then
    # uv is much faster than pip for the dependencies of vLLM
    export UV_CACHE_DIR="$(dirname "$VENV")/.uv_cache"  # same filesystem, so files can be hard-linked
    INSTALL="uv pip install --python $VENV/bin/python"
else
    python -m pip install --quiet --upgrade pip
    INSTALL="python -m pip install"
fi

# Some packages are provided only as modules (e.g. OpenCV on Alliance Canada): the wheelhouse has a
# placeholder whose error names the module ("module load gcc opencv/x.y.z"). Load it and try again.
TRIED_MODULES=""
install_from_wheelhouse() {{
    local log="$VENV/.slurm-llm-pip.log" modules
    for attempt in 1 2 3; do
        if ( set -o pipefail; $INSTALL "$1" 2>&1 | tee "$log" ); then
            return 0
        fi
        modules=$(sed -n 's#.*module load \([^/]*\)/x\.y\.z.*#\1#p' "$log" | head -1)
        if [ -z "$modules" ] || [[ "$TRIED_MODULES" == *"|$modules|"* ]]; then
            return 1
        fi
        TRIED_MODULES="$TRIED_MODULES|$modules|"
        echo "→ This cluster provides part of $1 as a module, not in its wheelhouse: loading it (module load $modules) and trying again"
        deactivate  # the module must be loaded before the venv is activated
        module load $modules
        source "$VENV/bin/activate"
        grep -qxF "$modules" "$VENV/.slurm-llm-modules" 2>/dev/null || echo "$modules" >> "$VENV/.slurm-llm-modules"
    done
    return 1
}}

for package in huggingface_hub vllm; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package already installed"
        continue
    fi
    if [ "$package" = vllm ]; then
        echo "Installing vLLM (the first time can take 10 minutes)..."
    else
        echo "Installing $package..."
    fi
    if [ -n "$WHEELHOUSE" ] && [[ "$FROM_PYPI" == *" $package "* ]]; then
        echo "  from PyPI (--from-pypi), with the wheelhouse as well"
        python -m pip install "$package"
    elif [ -n "$WHEELHOUSE" ]; then
        if ! install_from_wheelhouse "$package"; then
            cluster="${{SLURM_LLM_CLUSTER:-CLUSTER}}"
            echo "✗ Could not install $package from this cluster's wheelhouse alone (see pip's message above)." >&2
            echo "  To allow PyPI for it, run: sllm setup --cluster $cluster --from-pypi $package" >&2
            echo "  (or --from-pypi all to ignore the wheelhouse entirely)" >&2
            exit 1
        fi
    else
        $INSTALL "$package"
    fi
done

echo "✓ Environment ready at $VENV"
