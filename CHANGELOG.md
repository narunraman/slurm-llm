# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-10-01

### Changed
- The default `remote_root` (where job scripts, logs and endpoint files go on a cluster) is now
  `~/slurm-llm` instead of `~/remote-llm`. Clusters with `remote_root` set in `clusters.yaml` are
  not affected. If you relied on the old default, either add `remote_root: ~/remote-llm` to that
  cluster's entry or move the folder on the cluster.

## [0.6.1] - 2026-09-30

Fixes from an end-to-end test on a cluster.

### Fixed

- **`sllm setup` could run a corrupted script and report success.** Placeholders in the header
  comment of `setup_venv.sh` were filled in too: with several `setup_commands`, the second ran
  outside the comment and an apostrophe swallowed `set -e` and the venv activation, yet the table
  said "ready". Template comments no longer contain placeholders or apostrophes, and `setup` now
  checks that `vllm` and `huggingface_hub` import from the venv before reporting "ready" (exit
  code 1 otherwise). If you copied `setup_venv.sh` to `~/.config/slurm-llm/templates/`, copy it
  again (`sllm init --templates --force`) or write `{{venv_path}}` and `{{setup_commands}}` in its
  comments.
- **`sllm kill` left the proxy's port forwards open** on the shared SSH connection: the proxy now
  cancels them on SIGTERM and SIGHUP as on Ctrl+C, records them in `tunnels.json`, and cancels
  forwards left by killed processes when it starts; `kill` also cancels any left over.
- Jobs that left the queue showed NOT_FOUND (or COMPLETED) with a stale time left; they now get
  their final state from `sacct` (CANCELLED, COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY, ...), in
  both `sllm jobs` refreshes and queue discovery.
- Jobs found in the queue were recorded as submitted "now", so `sllm jobs` never hid them after 20
  minutes; the submit time is now read from Slurm and converted from the cluster's time zone.
- rclust's `gpu_types` under `resources:` (which rclust accepts) were ignored.
- `setup` announced the vLLM install even when vLLM was installed already, printed ssh's "Shared
  connection to ... closed." and truncated the disk column (now "AVAIL free of SIZE (USE% used)").

### Changed

- The proxy's `/v1/models` lists models under their short registry name (e.g. `qwen-3-8b`), with
  `hf_id` and `aliases` (Hugging Face repo id, registry id, served name); requests may use any of
  them, including the Hugging Face repo id.
- `config validate` warns about clusters in `clusters.yaml` that are not in rclust's config, and
  about rclust clusters without an entry in `clusters.yaml`.
- `generate_batch` raises a TypeError explaining the expected input (a list of
  `{"messages": [...]}` dicts) when given something else; the README documents it.

## [0.6.0] - 2026-09-29

First public release, under a new name.

### Renamed: remote-llm is now slurm-llm

- The project (distribution) is `slurm-llm`, the Python package `slurm_llm` and the command `sllm`;
  the repository is github.com/narunraman/slurm-llm. The old `remote-llm` command and `remote_llm`
  package are gone.
- Environment variables are now `SLURM_LLM_CONFIG_DIR`, `SLURM_LLM_STATE_DIR` and `SLURM_LLM_DEBUG`.

### Changed: file locations (XDG Base Directory layout)

- Settings (`clusters.yaml`, `model_registry.yaml`, `model_specs.yaml`, `resources.yaml`,
  `cluster_preferences.yaml`, `.hf_token`) are in `~/.config/slurm-llm/`, and custom templates in
  `~/.config/slurm-llm/templates/`.
- The job cache and tunnel registry are in `~/.local/state/slurm-llm/`. `$XDG_CONFIG_HOME` and
  `$XDG_STATE_HOME` are honoured. Lock files stay in the system temporary directory (host-local,
  so `flock` works even with network home directories).
- The old locations (`~/.remote-llm`, `~/.config/remote-llm`) and the old single-file
  `cluster.yaml` are no longer read.

### Changed

- **Depends on [rclust](https://github.com/narunraman/rclust)** (the public ClusterScheduler)
  and imports it as `rclust`. Its `Scheduler` is created with slurm-llm's config path and SSH
  connections; cluster choice follows rclust's `default_policy`.
- **One generic job template, `vllm.slurm`**, replaces the site-specific ones. Site differences
  are settings in `clusters.yaml`: `setup_commands` (e.g. module loads), `sbatch_args`,
  `cpus_per_task`. Your own templates (now in `~/.config/slurm-llm/templates/`) still take
  precedence over the packaged ones.
- **SSH**: commands never prompt (BatchMode); the control socket is the one rclust uses, so
  `rclust connect` and `sllm connect` share connections. `connect --timeout` is now
  `connect --persist` (as in rclust). The site-specific filtering of login output is gone, and
  `ssh_port` is no longer read (set `Port` in `~/.ssh/config`).
- `init` writes a `clusters.yaml` entry for each cluster in rclust's config and no longer copies
  templates unless asked (`--templates`). `config validate` checks the configuration actually used.
- `cancel` takes job ids as arguments (`--job` still works) and `--cluster` for jobs not in the cache.
- The CLI imports heavy modules only when needed (`--help` takes ~0.07 s instead of ~0.33 s),
  reports expected errors in one line with exit code 1, and exits with 130 on Ctrl+C.
- The Hugging Face token may come from `HF_TOKEN`; it is sent to the cluster on stdin.
- Removed unused dependencies (paramiko, requests, tqdm, huggingface-hub) and the obsolete
  migration and test scripts.

### Fixed

- `prepare MODEL --instances N` (or `MODEL:0-3`) prepared only the last instance.
- `model_memory_gb` in the registry (also set by the out-of-memory prompt) was ignored when sizing jobs.
- The model's context length was never read back from vLLM (`max_model_len` field).
- A failed status query marked every job on that cluster NOT_FOUND (or COMPLETED), so running
  jobs were no longer reused; jobs that ended as OUT_OF_MEMORY, NODE_FAIL or similar were waited
  for forever.
- Job logs were written to the home directory by templates without `--output`, where remote-llm
  did not look; `#SBATCH` paths with `~` or variables are now resolved to absolute paths.
- The proxy's deactivate and shutdown killed whatever process listened on the tunnel port, which
  was the shared SSH master connection; forwards are now added and cancelled on the master.
- `config preferences` and `config validate` read an obsolete file.
- The newest snapshot of a model is taken from Hugging Face's `refs/main` instead of sorting hashes.
- `prepare` exited silently on configuration errors, and with status 0 after updating the registry.

## [0.5.0] - 2026-05-12

### Added

- **MoE- and quantization-aware model scanning.** `model_scanner` now
  reads `num_local_experts` / `num_experts` / `n_routed_experts` /
  `num_routed_experts` (covering Mixtral, gpt-oss, Gemma-MoE, Qwen-MoE,
  DeepSeek) plus `n_shared_experts` and `moe_intermediate_size`, and
  multiplies FFN params by the total expert count. Also reads
  `quantization_config.quant_method` and `modules_to_not_convert`:
  applies 0.53 B/param for 4-bit families (mxfp4 / nf4 / gptq / awq /
  bnb_4bit) and 1.06 B/param for 8-bit (fp8 / int8 / bnb_8bit), and
  preserves native dtype on attention when excluded. Embeddings, LM
  head, and router stay at native dtype. Sanity-checked against
  published targets (gpt-oss-120b mxfp4 ≈ 60 GB, Mixtral-8x7B ≈ 87 GB,
  Llama-3.1-70B dense ≈ 131 GB).
- **Per-model `vllm_extra_args`.** Optional list of CLI tokens appended
  verbatim to `vllm serve`. Useful for model-specific quirks (e.g.
  `--limit-mm-per-prompt=image=0` on multimodal models used text-only).
  Wired through `ModelConfig`, `ConfigLoader`, and the job template.
- **`require_single_node` cluster flag** + `JobSpec(nodes=…)`. When set
  on a cluster, the scheduler passes `nodes=1` so all GPUs land on a
  single node. Requires `clusterscheduler >= 0.2.0`.
- **`remote-llm clean` CLI**: prunes terminal/old entries from the
  local job cache. Supports `--all` and `--older-than DAYS`.

### Changed

- **Parallel job-status refresh.** `JobManager.list_jobs` now accepts
  `ssh_clients={cluster: client}` and refreshes statuses across all
  clusters in one parallel pass. `api_proxy.list_jobs` and the
  `remote-llm jobs` CLI use the batched API instead of per-cluster
  loops.
- **Auto-prune stale jobs on load.** `SUBMITTED` jobs older than 7 days
  are marked `EXPIRED` automatically so dead cache entries from prior
  sessions don't accumulate.
- **`ssh_manager.execute_parallel`** now accepts either a single
  command string or a `{cluster: command}` dict.

### Fixed

- **vLLM job template.** Stop copying the shared venv to `$SLURM_TMPDIR`;
  the upfront copy cost more than it saved. Activate the shared venv
  directly and relink a broken `python` symlink when a compute node's
  module paths differ from the login node's.


## [0.2.0] - 2026-01-04

### ⚠️ Breaking Changes

This release introduces a new dependency on [ClusterScheduler](https://github.com/narunraman/rclust) (now rclust),
which requires a separate configuration directory. See [Migration from v0.1.0](#migration-from-v010) below.

### Added

- **Model scanning system** (`remote-llm scan`)
  - Scans downloaded models to extract architecture parameters from `config.json`
  - Calculates accurate KV cache memory requirements based on model architecture
  - Auto-generates `~/.config/remote-llm/model_specs.yaml` with model specifications
  - Automatically runs after `remote-llm download`

- **Accurate GPU memory estimation**
  - Calculates total GPU memory as: `(model_weights + kv_cache) * 1.1`
  - KV cache formula: `2 × layers × kv_heads × head_dim × dtype_bytes × context_len`
  - Supports full context lengths (e.g., 131K for Llama-3.1) without OOM

- **Job discovery from running vLLM processes**
  - Discovers existing vLLM jobs on clusters when local cache is empty
  - Recovers from interrupted sessions without losing track of jobs

- **Job cancellation** (`remote-llm cancel`)
  - Cancel running or pending jobs by job ID

- **Auto port conflict resolution**
  - Automatically finds available local ports when default is in use

- **Extended OpenAI client timeout**
  - Accommodates large model response times

### Changed

- **Memory display in `submit` command**
  - Now shows calculated memory including KV cache: `Memory: 49.0GB (incl. KV cache)`

- **GPU matching algorithm**
  - Uses accurate memory calculation instead of 1.15× multiplier

- **SSH connection management**
  - Replaced `disconnect()` with `release()` for better connection pooling

- **Cluster scheduling**
  - Now uses ClusterScheduler library for queue wait time estimation

### Deprecated

- `model_memory_gb` field in `model_registry.yaml` is no longer required
  - Memory is now calculated from `model_specs.yaml` (auto-generated)
  - Field is still read as a fallback if model specs are unavailable

### Migration from v0.1.0

Follow these steps:

1. **Create ClusterScheduler config directory:**
   ```bash
   mkdir -p ~/.config/rclust
   ```

2. **Copy cluster configuration:**
   ```bash
   cp ~/.config/remote-llm/cluster.yaml ~/.config/rclust/config.yaml
   ```

3. **Reinstall remote-llm** (to get new ClusterScheduler dependency):
   ```bash
   pip install --upgrade remote-llm
   # or if installed from source:
   pip install -e /path/to/remote-llm
   ```

4. **Scan your models** (populates model_specs.yaml):
   ```bash
   remote-llm scan --cluster <your-cluster> --force
   ```

5. **(Optional) Clean up model_registry.yaml:**
   - Remove any `model_memory_gb` fields (no longer needed)
   - Add `max_model_len` overrides if you want smaller context windows

### Rollback to v0.1.0

If you need to rollback:
```bash
pip install remote-llm==0.1.0
# or
pip install git+https://github.com/narunraman/remote-llm.git@v0.1.0
```

## [0.1.0] - 2025-12-15

### Added

- Initial release
- SSH-based vLLM job submission to SLURM clusters
- OpenAI-compatible API proxy
- Model download and management
- Multi-cluster support with connection pooling
