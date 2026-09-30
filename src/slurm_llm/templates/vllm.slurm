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

# Endpoint file for clients running on the cluster (RemoteLLM(local=True))
NODE_DIR="{output_dir}/manifests/{model_name}/nodes"
NODE_FILE="$NODE_DIR/$SLURM_JOB_ID.node"
mkdir -p "$NODE_DIR"
printf 'node=%s\nmodel=%s\njob_id=%s\n' "$SLURMD_NODENAME" "{model_name}" "$SLURM_JOB_ID" > "$NODE_FILE"
trap 'rm -f "$NODE_FILE"' EXIT

vllm serve "$MODEL_DIR" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size {num_gpus} \
    --served-model-name {model_name} \
    {max_model_len_arg} {vllm_extra_args}
