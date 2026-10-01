# slurm-llm

Run open LLMs on your Slurm clusters with [vLLM](https://docs.vllm.ai) and use them from your laptop as if they were a local OpenAI-compatible API.

slurm-llm submits a Slurm job that serves the model with vLLM, sized to the model's GPU memory needs and sent to whichever of your clusters is expected to start it soonest. It waits for the server to come up and forwards a local port to it through your existing SSH connection, so any OpenAI client can use it:

```bash
sllm prepare qwen-3-8b        # submit (or reuse) a job and wait until the model is loaded
sllm proxy                    # http://localhost:8000/v1, routed by the "model" field
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="qwen-3-8b", messages=[{"role": "user", "content": "Hello!"}])
```

It works with clusters that require two-factor authentication: you log in once with `sllm connect`, and every later command reuses that connection (OpenSSH connection sharing). Nothing is installed on the cluster except a Python environment in a directory you choose.

## Requirements

- On your machine: Python 3.10 or newer and the OpenSSH client (`ssh`). macOS and Linux are supported.
- One or more Slurm clusters with GPUs that you can reach with `ssh`, where login nodes can reach the internet (to install vLLM and download models) and you can reach compute nodes from the login node over TCP (for the port forward).
- Enough disk space on the cluster for model weights (an 8B model is about 16 GB; a 70B model about 140 GB).

## Install

```bash
uv tool install git+https://github.com/narunraman/slurm-llm
# or: pipx install git+https://github.com/narunraman/slurm-llm
```

This installs the `sllm` command and its dependency [rclust](https://github.com/narunraman/rclust), which slurm-llm uses to describe clusters and estimate queue times.

## Getting started

### 1. Describe your clusters (rclust)

```bash
rclust config
```

or write `~/.config/rclust/config.yaml` yourself:

```yaml
clusters:
  cluster-a:                   # any name; used in commands
    host: login.hpc.example.edu  # or an alias from ~/.ssh/config (default: the name)
    user: your-username        # optional; default from ~/.ssh/config
    account: your-account      # optional Slurm account (--account)
    gpu_types:                 # optional, but needed to size jobs
      - type: a100
        slurm_spec: a100       # the name in Slurm's GPU requests (--gpus=a100:N)
```

`rclust discover cluster-a` lists the GPU types a cluster has (after `rclust connect`).

### 2. Create slurm-llm's config

```bash
sllm init
```

This writes `~/.config/slurm-llm/` with an entry per cluster in `clusters.yaml`. Fill in where models go and how much memory each GPU type has (or run `sllm config cluster cluster-a`):

```yaml
clusters:
  cluster-a:
    remote_root: ~/slurm-llm         # job scripts and logs
    model_path: $SCRATCH/models      # model weights and the Python environment: needs space
    setup_commands:                  # optional, run before creating the environment and in each job
      - module load python/3.12 cuda/12.4
    gpu_types:
      - type: a100
        memory: 80
```

`~` and variables such as `$SCRATCH` are expanded on the cluster. See [`clusters.example.yaml`](src/slurm_llm/configs/clusters.example.yaml) for all settings (extra `#SBATCH` options, CPUs, MIG slices). Check the result with `sllm config validate`.

### 3. Log in, set up, download

```bash
sllm connect                   # answers password / two-factor prompts once; -p 8h to keep it longer
sllm setup                     # creates <model_path>/.venv with vLLM (10 minutes the first time)
sllm download qwen-3-8b        # downloads to <model_path> on the login node, then scans it
```

`download` accepts registry names (`sllm models`) or Hugging Face repo ids. For gated models (e.g. Llama), save a [Hugging Face token](https://huggingface.co/settings/tokens) first with `sllm config hf-token` (or set `HF_TOKEN`) and re-run `setup`; the token is sent to the cluster over SSH's standard input, never on a command line.

### 4. Serve and use models

```bash
sllm prepare qwen-3-8b         # waits for the queue, then for vLLM to load
sllm proxy                     # keep this running
sllm query qwen-3-8b "What is tensor parallelism?"   # or omit the prompt to chat
```

Any OpenAI-compatible client can now use `http://localhost:8000/v1` with the model's short name (`qwen-3-8b`). Several models can run at once; the proxy routes each request by its `model` field and activates models that start later on their first request. Stop the servers with `sllm cancel JOB_ID` (see `sllm jobs`); they also stop at their Slurm time limit (`--time`, default from `resources.yaml`).

## How it works

- **Clusters and queue estimates.** Clusters come from rclust's config. To start a model, slurm-llm considers every cluster whose GPUs can hold it and asks rclust which will start the job soonest (`sbatch --test-only`, following rclust's `default_policy`). `--force-cluster` and `-x/--exclude` override this; `cluster_preferences.yaml` restricts clusters by model size.
- **GPU sizing.** `sllm scan` (run automatically after `download`) reads each model's `config.json` and estimates its GPU memory: weights (MoE- and quantization-aware) plus the KV cache for the full context window, plus 25% headroom. slurm-llm requests the smallest GPU that fits, or enough of the largest full GPU for tensor parallelism. Set `max_model_len` in `model_registry.yaml` to use a smaller context and fewer GPUs, or `model_memory_gb` to override the estimate.
- **The job.** [`vllm.slurm`](src/slurm_llm/templates/vllm.slurm) runs `vllm serve` on a port chosen at run time and prints it to the job log, where slurm-llm finds it. To change the job, copy the template to `~/.config/slurm-llm/templates/` and edit it (see the [templates README](src/slurm_llm/templates/README.md)).
- **SSH.** Every command runs over one shared connection per cluster (ssh ControlMaster, the same socket `rclust connect` uses) and never prompts. Clusters that accept SSH keys need no `connect`; for the others, a command that finds no open connection tells you to run `sllm connect CLUSTER`. Connections close after being idle for an hour (`connect -p` changes this).
- **Local state.** Submitted jobs are cached in `~/.local/state/slurm-llm/jobs.json`; `sllm jobs` also finds vLLM jobs you started elsewhere (Slurm job names `vllm-<model>`).

## Commands

| Command | What it does |
|---|---|
| `init` | Create the config files (`--templates` also copies the job templates for editing) |
| `config view \| validate \| cluster \| model \| preferences \| hf-token` | Inspect, check and edit the configuration |
| `connect [CLUSTER...]` | Log in once and keep the connection open (`--status` to check) |
| `setup [-c CLUSTER]` | Create the Python environment with vLLM on the clusters |
| `download MODEL...` | Download models to the clusters and scan them |
| `scan [MODEL...]` | Re-estimate GPU memory for downloaded models |
| `prepare MODEL...` | Start models (or reuse running jobs) and wait until they are ready |
| `submit MODEL...` | Submit jobs without waiting (`--gpus h100:2`, `--time`, `--instances N`) |
| `proxy` | Serve all running models on `http://localhost:8000/v1` |
| `query MODEL [PROMPT]` | Ask a question, or chat, through the proxy |
| `jobs` | List jobs with their status, node and time left |
| `health --job ID` | Check whether a job's server is ready |
| `cancel JOB_ID...` | Cancel jobs |
| `clean` | Forget finished jobs in the local list |
| `models` | List registry models (`--cluster` shows which are downloaded) |
| `kill` | Stop running proxies |

`MODEL:N` addresses instance N of a model (`prepare qwen-3-8b:0-3` starts four copies; use `qwen-3-8b:2` as the model name in requests). Every command has `--help`. Commands exit with status 1 on errors and 130 when interrupted.

## Configuration files

slurm-llm follows the [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/latest/) layout (`$XDG_CONFIG_HOME` and `$XDG_STATE_HOME` are honoured when set):

| Location | Contents |
|---|---|
| `~/.config/slurm-llm/` | Settings (override with `SLURM_LLM_CONFIG_DIR`), below |
| `~/.config/slurm-llm/templates/` | Optional: your own job templates, used instead of the packaged ones with the same name |
| `~/.local/state/slurm-llm/` | The job cache (`jobs.json`) and the tunnel registry (override with `SLURM_LLM_STATE_DIR`) |

Lock files go to the system temporary directory, which is always local to the machine. Settings files:

| File | Contents |
|---|---|
| `clusters.yaml` | Per-cluster settings for slurm-llm (paths, setup commands, GPU memory) |
| `model_registry.yaml` | Models you can serve, by short name ([packaged default](src/slurm_llm/configs/model_registry.yaml)) |
| `resources.yaml` | Memory and time limit for each model size |
| `cluster_preferences.yaml` | Optional: clusters to consider for each model size |
| `model_specs.yaml` | Written by `scan` |
| `.hf_token` | Optional Hugging Face token (mode 600) |

The clusters themselves are in rclust's config, `~/.config/rclust/config.yaml`.

To add a model, add an entry to `model_registry.yaml` (the `id` is the Hugging Face repo id with `/` replaced by `--`, prefixed with `models--`) and `sllm download` it.

## Using it from Python

```python
from slurm_llm import RemoteLLM

with RemoteLLM("qwen-3-8b") as llm:     # submits or reuses a job, waits, opens a tunnel
    llm.ensure_ready()
    reply = llm.generate(messages=[{"role": "user", "content": "Hello!"}])
    print(reply.choices[0].message.content)
```

`generate_batch` sends many requests in parallel (up to `max_concurrent_per_endpoint`, default 10, per server) and returns the results in the order of the requests. Each request is a dict with a `messages` list, plus optional per-request settings such as `temperature` or `max_tokens`; a request that fails gives its exception in place of a response:

```python
questions = ["What is 2+2?", "Name a prime number."]
with RemoteLLM("qwen-3-8b") as llm:
    llm.ensure_ready()
    replies = llm.generate_batch(
        [{"messages": [{"role": "user", "content": q}]} for q in questions],
        temperature=0.2,           # default for every request
    )
for question, reply in zip(questions, replies):
    print(question, "->", reply if isinstance(reply, Exception) else reply.choices[0].message.content)
```

For jobs that run on the cluster itself, `submit_model` starts servers without waiting and returns their job ids (for Slurm dependencies), and `RemoteLLM(model, local=True, nodes_dir=...)` connects to them directly from a compute node, balancing requests over several instances:

```python
from slurm_llm import submit_model

result = submit_model("qwen-3-8b", instances=4)
print(result.job_ids, result.after_any_dependency, result.nodes_dir)
```

## Troubleshooting

- **"not connected to CLUSTER"**: run `sllm connect CLUSTER`. Check with `sllm connect --status`.
- **Job fails immediately**: `sllm prepare` shows the end of the job's log. Common causes are a missing `account`, a wrong GPU type name (`slurm_spec`), or `setup_commands` that load a different Python than `setup` used.
- **CUDA out of memory / KV cache errors**: set `max_model_len` for the model, or `model_memory_gb` to request more GPU memory (`sllm config model NAME`); `prepare` offers this when it happens.
- **The proxy cannot reach a model**: the port forward goes from the login node to the compute node; some sites block this, in which case run your client on the cluster and use `RemoteLLM(local=True)`.

## Development

```bash
git clone https://github.com/narunraman/slurm-llm && cd slurm-llm
uv sync
uv run pytest            # offline; never contacts a cluster
```

## License

MIT; see [LICENSE](LICENSE).
