#!/bin/bash -l
# Creates the Python environment that jobs run vLLM from; run by `sllm setup` on the
# login node. It is safe to run again. slurm-llm fills in:
#   {{venv_path}}       where to create the venv ({{model_path}}/.venv)
#   {{setup_commands}}  the setup_commands of the cluster from clusters.yaml (e.g. module loads)
# Copy this file to ~/.config/slurm-llm/templates/ to customise it (set setup_template in clusters.yaml
# to use a different name).

set -e

{setup_commands}

VENV={venv_path}

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV..."
    python3 -m venv "$VENV"
else
    echo "Virtual environment already exists at $VENV"
fi
source "$VENV/bin/activate"

# uv is much faster than pip for the dependencies of vLLM; use it when it is available.
if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="$(dirname "$VENV")/.uv_cache"  # same filesystem, so files can be hard-linked
    INSTALL="uv pip install --python $VENV/bin/python"
else
    python -m pip install --quiet --upgrade pip
    INSTALL="python -m pip install"
fi

for package in huggingface_hub vllm; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package already installed"
    else
        if [ "$package" = vllm ]; then
            echo "Installing vLLM (the first time can take 10 minutes)..."
        else
            echo "Installing $package..."
        fi
        $INSTALL "$package"
    fi
done

echo "✓ Environment ready at $VENV"
