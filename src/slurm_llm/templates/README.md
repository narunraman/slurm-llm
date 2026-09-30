# Templates

- `vllm.slurm`: the batch job that serves one model with `vllm serve`.
- `setup_venv.sh`: creates the Python environment (vLLM and `huggingface_hub`) that jobs use; run by `sllm setup`.

Most sites need no template changes; per-cluster differences go in `~/.config/slurm-llm/clusters.yaml`:

```yaml
clusters:
  cluster-a:
    setup_commands:            # run before anything else, in both templates
      - module load python/3.12 cuda/12.4
    sbatch_args:               # extra #SBATCH options
      - --partition=gpu
    cpus_per_task: 8
```

To change a template, copy it to `~/.config/slurm-llm/templates/` (`sllm init` does this) and edit the copy; a file there takes precedence over the packaged one with the same name. To use a template with a different name for one cluster, set `slurm_template` or `setup_template` for that cluster in `clusters.yaml`.

Templates are filled in with Python's `str.format`, so literal braces must be doubled (`{{` and `}}`), also in comments: a placeholder in a comment is filled in like any other, and a multi-line value (such as several `setup_commands`) would continue on uncommented lines. Write placeholder names in comments as `{{name}}`.

## Placeholders in `vllm.slurm`

| Placeholder | Value |
|---|---|
| `{model_name}` | name vLLM serves the model under, e.g. `Qwen3-8B` |
| `{model_id}` | shell expression for the model's snapshot directory in the Hugging Face cache |
| `{model_path}` | the cluster's `model_path` (the Hugging Face cache), as written in the config |
| `{venv_path}` | `<model_path>/.venv`, quoted for the shell |
| `{sbatch_lines}` | `#SBATCH --account=...` (if an account is configured) and `sbatch_args` |
| `{account}` | Slurm account, or empty |
| `{time}`, `{memory}` | time limit and memory (GB) from `resources.yaml` |
| `{gpu_slurm_arg}`, `{num_gpus}` | GPU request, e.g. `--gpus-per-node=h100:2`, and its count |
| `{nodes_arg}` | `#SBATCH --nodes=1` unless GPUs may be spread over nodes |
| `{cpus_per_task}` | from `clusters.yaml` (default 4) |
| `{output_dir}` | absolute path of `remote_root`, for logs and endpoint files |
| `{setup_commands}` | the cluster's `setup_commands` |
| `{max_model_len_arg}` | `--max-model-len N` if the registry sets `max_model_len` |
| `{vllm_extra_args}` | the model's `vllm_extra_args` |

The job must print `Port selection: ... actual=PORT` (or let vLLM log its startup line) so that slurm-llm can find the server's port, and write its logs to `{output_dir}/slurm-%j.out` and `.err`.

## Placeholders in `setup_venv.sh`

`{venv_path}` and `{setup_commands}`.
