# slurm-llm

slurm-llm serves open language models with [vLLM](https://docs.vllm.ai) on your Slurm clusters and makes them available on your laptop as an OpenAI-compatible API at `http://localhost:8000/v1`. It sizes each job to the model's GPU memory, sends it to whichever cluster will start it soonest (using <a href="https://github.com/narunraman/rclust#gh-light-mode-only"><img src=".github/assets/github_logo_light.svg" height="14" alt=""></a><a href="https://github.com/narunraman/rclust#gh-dark-mode-only"><img src=".github/assets/github_logo_dark.svg" height="14" alt=""></a> [rclust](https://github.com/narunraman/rclust)), waits for the server to load, and forwards a port to it over SSH. Clusters with two-factor login work: you log in once and every later command reuses the connection.

## Run it

You need Python 3.10+, OpenSSH and [uv](https://docs.astral.sh/uv/) on your machine, and a Slurm cluster with GPUs you can reach with `ssh`, whose login node can reach the internet.

```bash
uv tool install git+https://github.com/narunraman/slurm-llm@v0.8.0
```

This installs `sllm` and `rclust`. Describe your clusters to rclust, then add where slurm-llm should keep models:

```bash
rclust config                  # login host, user, Slurm account, GPU types
rclust discover --save         # find each cluster's GPU types and their memory
sllm init                      # writes ~/.config/slurm-llm/clusters.yaml; set model_path there
sllm connect                   # log in once, answering any two-factor prompt
sllm setup                     # create the Python environment with vLLM (about 10 minutes the first time)
sllm download qwen-3-8b
```

Then start a model and keep the proxy running:

```bash
sllm prepare qwen-3-8b         # wait for the queue, then for vLLM to load
sllm proxy
```

Any OpenAI client can now use it, with the model's short name:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="qwen-3-8b", messages=[{"role": "user", "content": "Hello!"}])
```

Several models can run at once; the proxy routes each request by its `model` field. Servers stop at their Slurm time limit, or with `sllm cancel JOB_ID`. For gated models (e.g. Llama), save a [Hugging Face token](https://huggingface.co/settings/tokens) with `sllm config hf-token` before `setup`.

From Python, `SlurmLLM` does the same in one step, and `generate_batch` sends many requests in parallel:

```python
from slurm_llm import SlurmLLM

with SlurmLLM("qwen-3-8b") as llm:     # submits or reuses a job, waits, opens a tunnel
    llm.ensure_ready()
    replies = llm.generate_batch([{"messages": [{"role": "user", "content": q}]} for q in ["2+2?", "A prime?"]])
```

For jobs that run on the cluster itself, `submit_model("qwen-3-8b", instances=4)` starts servers without waiting and `SlurmLLM(model, local=True, nodes_dir=...)` connects to them from a compute node.

## Commands

| Command | What it does |
|---|---|
| `sllm init` | Create the config files (`--templates` also copies the job templates for editing) |
| `sllm config view` / `validate` / `cluster` / `model` / `preferences` / `hf-token` | Inspect, check and edit the configuration |
| `sllm connect [CLUSTER...]` | Log in once and keep the connection open (`--status` to check) |
| `sllm setup` | Create the Python environment with vLLM on the clusters |
| `sllm download MODEL...` | Download models to the clusters and estimate their GPU memory |
| `sllm prepare MODEL...` | Start models (or reuse running jobs) and wait until they are ready |
| `sllm submit MODEL...` | Submit jobs without waiting (`--gpus h100:2`, `--time`, `--instances N`) |
| `sllm proxy` | Serve all running models on `http://localhost:8000/v1` |
| `sllm query MODEL [PROMPT]` | Ask a question, or chat, through the proxy |
| `sllm jobs` / `cancel` / `clean` | List, cancel and forget jobs |
| `sllm models` | List the models you can serve (`--cluster` shows which are downloaded) |

`MODEL:N` names instance N of a model: `sllm prepare qwen-3-8b:0-3` starts four copies, and `qwen-3-8b:2` is one of
them in requests.

## Configuration

Clusters (hosts, accounts, GPU types) are in rclust's config, `~/.config/rclust/config.yaml`. slurm-llm's own
settings are in `~/.config/slurm-llm/` (or `$SLURM_LLM_CONFIG_DIR`):

| File | Contents |
|---|---|
| `clusters.yaml` | Per cluster: where models and logs go, and setup commands such as `module load` ([example](src/slurm_llm/configs/clusters.example.yaml)) |
| `model_registry.yaml` | The models you can serve, by short name ([default](src/slurm_llm/configs/model_registry.yaml)) |
| `resources.yaml` | Memory and time limit for each model size |
| `cluster_preferences.yaml` | Optional: which clusters to use for each model size |

A model's GPU memory is estimated from its `config.json` (weights plus the KV cache for the full context); set
`max_model_len` in `model_registry.yaml` to use a shorter context and fewer GPUs. The job script is
[`vllm.slurm`](src/slurm_llm/templates/vllm.slurm); to change it, run `sllm init --templates` and edit the copy in
`~/.config/slurm-llm/templates/`. Job state is kept in `~/.local/state/slurm-llm/` (or `$SLURM_LLM_STATE_DIR`).

On clusters with their own wheelhouse (as on Alliance Canada), `sllm setup` installs from it only, as these sites
ask; `sllm setup --from-pypi PACKAGE` allows PyPI for a package the wheelhouse lacks.

## Troubleshooting

- **"not connected to CLUSTER"**: run `sllm connect CLUSTER`.
- **The job fails at once**: `sllm prepare` shows the end of the job's log. Usually the `account`, the GPU type's
  Slurm name, or `setup_commands` loading a different Python than `setup` used.
- **CUDA out of memory**: set `max_model_len` or `model_memory_gb` for the model (`sllm config model NAME`).
- **The proxy cannot reach a model**: some sites block forwarding from the login node to compute nodes; run your
  client on the cluster with `SlurmLLM(local=True)`.

## Tests

```bash
uv run pytest          # offline; never contacts a cluster
```

## License

[MIT](LICENSE).
