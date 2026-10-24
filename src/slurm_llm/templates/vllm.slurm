#!/bin/bash -l
#SBATCH --job-name=vllm-{model_name}
{sbatch_lines}
#SBATCH --time={time}
#SBATCH --mem={memory}G
{nodes_arg}
#SBATCH {gpu_slurm_arg}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err

# Serve one model with vLLM. slurm-llm fills in the placeholders; see the README in this
# directory. Copy this file to ~/.config/slurm-llm/templates/ to customise it.

# Cluster-specific setup from clusters.yaml (setup_commands), e.g. `module load python cuda`
{setup_commands}

# A port derived from the job id, moved up if taken (another server may share the node).
# slurm-llm reads it from the "Port selection" line below.
PORT=$((30000 + SLURM_JOB_ID % 10000))
for _ in $(seq 100); do
    python3 -c "import socket, sys; socket.socket().bind(('', int(sys.argv[1])))" "$PORT" 2>/dev/null && break
    PORT=$((PORT < 39999 ? PORT + 1 : 30000))
done
echo "Port selection: preferred=$((30000 + SLURM_JOB_ID % 10000)), actual=$PORT"

# vLLM cannot parse MIG device UUIDs in CUDA_VISIBLE_DEVICES; a MIG slice is device 0.
if [[ "$CUDA_VISIBLE_DEVICES" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

MODEL_DIR={model_id}
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: {model_name} is not downloaded to model_path (run 'sllm download')"
    exit 1
fi

VENV={venv_path}
if [ ! -d "$VENV" ]; then
    echo "ERROR: no Python environment at $VENV (run 'sllm setup')"
    exit 1
fi
# If the interpreter of the venv does not run on this node (e.g. module paths differ between login
# and compute nodes), point it at a python of the same version from the PATH of this node.
if ! "$VENV/bin/python" --version >/dev/null 2>&1; then
    PYVER=$(ls "$VENV/lib" | sed -n 's/^python\([0-9]*\.[0-9]*\)$/\1/p' | head -1)
    for candidate in "python$PYVER" python3; do
        if command -v "$candidate" >/dev/null && "$candidate" -c "import sys; sys.exit(sys.version[:len('$PYVER')] != '$PYVER')"; then
            ln -sf "$(command -v "$candidate")" "$VENV/bin/python"
            break
        fi
    done
fi
source "$VENV/bin/activate" || {{ echo "ERROR: could not activate $VENV"; exit 1; }}

echo "Model: {model_name} ($MODEL_DIR)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME, {num_gpus} GPU(s), port $PORT"

# Endpoint file for clients running on the cluster (SlurmLLM(local=True))
NODE_DIR="{output_dir}/manifests/{model_name}/nodes"
NODE_FILE="$NODE_DIR/$SLURM_JOB_ID.node"
mkdir -p "$NODE_DIR"
printf 'node=%s\nmodel=%s\njob_id=%s\n' "$SLURMD_NODENAME" "{model_name}" "$SLURM_JOB_ID" > "$NODE_FILE"
trap 'rm -f "$NODE_FILE"' EXIT

# Caches go next to the models, not in the home directory (often small on clusters); yours win if set
CACHE="{model_path}/.cache"
export VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT:-$CACHE/vllm}}" TRITON_CACHE_DIR="${{TRITON_CACHE_DIR:-$CACHE/triton}}"
export FLASHINFER_WORKSPACE_BASE="${{FLASHINFER_WORKSPACE_BASE:-{model_path}}}"  # it adds .cache/flashinfer

# FlashInfer's sampler (vLLM's fastest) compiles itself on first use with this node's CUDA compiler.
# If that compiler is missing or older than the CUDA PyTorch was built with, compiling fails, so use
# vLLM's own sampler instead; lines starting "slurm-llm:" are shown by `sllm prepare`.
if [ -z "${{VLLM_USE_FLASHINFER_SAMPLER:-}}" ]; then
    WHY=$(python - <<'SLURM_LLM_CHECK' 2>/dev/null
import importlib.util, os, re, subprocess
import torch
from torch.utils.cpp_extension import CUDA_HOME
if torch.version.cuda and importlib.util.find_spec("flashinfer") and not importlib.util.find_spec("flashinfer_jit_cache"):
    try:
        out = subprocess.run([os.path.join(CUDA_HOME or "", "bin", "nvcc"), "--version"],
                             capture_output=True, text=True, timeout=60).stdout
        found = re.search(r"release (\d+)\.(\d+)", out)
    except OSError:
        found = None
    built = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    if not found:
        print("there is no CUDA compiler on this node")
    elif tuple(int(x) for x in found.groups()) < built:
        print("this node's CUDA compiler is " + ".".join(found.groups()) + ", older than the CUDA "
              + torch.version.cuda + " that vLLM was built with")
SLURM_LLM_CHECK
)
    if [ -n "$WHY" ]; then
        export VLLM_USE_FLASHINFER_SAMPLER=0
        echo "slurm-llm: using vLLM's own sampler, not FlashInfer's: $WHY"
    fi
fi

vllm serve "$MODEL_DIR" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size {num_gpus} \
    --served-model-name {model_name} \
    {max_model_len_arg} {vllm_extra_args}
