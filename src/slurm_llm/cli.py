"""Command-line interface for slurm-llm.

Heavy modules (the OpenAI client, aiohttp, rclust's scheduler) are imported inside the commands
that need them, so `--help` and simple commands start quickly.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import rich_click as click
from rich.console import Console

click.rich_click.TEXT_MARKUP = "markdown"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.MAX_WIDTH = 100
click.rich_click.ERRORS_SUGGESTION = "Try 'sllm --help' for help."
click.rich_click.COMMAND_GROUPS = {
    "sllm": [
        {"name": "Getting started", "commands": ["init", "config", "connect", "setup", "download", "scan"]},
        {"name": "Running models", "commands": ["prepare", "proxy", "query", "submit"]},
        {"name": "Jobs", "commands": ["jobs", "health", "cancel", "clean", "models", "kill"]},
    ],
}

console = Console()
err_console = Console(stderr=True, soft_wrap=True)


class CLIGroup(click.RichGroup):
    """Shows expected errors as one line (exit 1) and Ctrl+C as "Interrupted" (exit 130)."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except KeyboardInterrupt:
            err_console.print("\n[yellow]Interrupted[/yellow]")
            sys.exit(130)
        except click.exceptions.ClickException:
            raise
        except Exception as e:
            from .config import SlurmLLMError, env

            if isinstance(e, SlurmLLMError) or env("DEBUG") is None and _is_expected(e):
                err_console.print(f"[red]Error:[/red] {e}", highlight=False)
                sys.exit(1)
            raise


def _is_expected(e: Exception) -> bool:
    """Errors that describe a problem with the environment rather than a bug in slurm-llm."""
    return isinstance(e, (OSError, TimeoutError))


def fail(message: str):
    """Stop with an error message (exit code 1)."""
    from .config import SlurmLLMError

    raise SlurmLLMError(message)


@click.group(cls=CLIGroup)
@click.version_option(package_name="slurm-llm")
@click.option("--verbose", "-v", is_flag=True, help="Show progress details (log messages).")
def main(verbose: bool):
    """Run open LLMs on Slurm clusters with vLLM and use them locally through an OpenAI-compatible API.

    First time: `sllm init`, then `sllm connect` and `sllm setup`.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


# --- helpers ---------------------------------------------------------------------------------------


def _loader():
    from .config import ConfigLoader

    return ConfigLoader()


def _select_clusters(loader, names) -> dict:
    """The named clusters (all if none are named)."""
    clusters = loader.load_clusters()
    for name in names:
        if name not in clusters:
            fail(f"Unknown cluster '{name}' (configured: {', '.join(clusters)})")
    return {n: clusters[n] for n in names} if names else clusters


def _connect(cluster_config, interactive: bool = True):
    """An SSHClient with an open connection (prompting for login if needed)."""
    from .ssh import SSHClient

    client = SSHClient(cluster_config)
    client.connect(interactive=interactive)
    return client


def _require_model_path(cluster_config) -> str:
    if not cluster_config.model_path:
        fail(
            f"No model_path for cluster '{cluster_config.name}'. Set it with "
            f"`sllm config cluster {cluster_config.name}` (a directory with room for model weights)."
        )
    return cluster_config.model_path


def parse_model_instances(model_spec: str) -> tuple[str, list[int]]:
    """"model" -> [0]; "model:3" -> [3]; "model:0-4" -> [0..4]; "model:0,2,5" -> [0, 2, 5]."""
    name, sep, spec = model_spec.rpartition(":")
    if not sep:
        return model_spec, [0]
    try:
        if "-" in spec:
            start, end = spec.split("-", 1)
            return name, list(range(int(start), int(end) + 1))
        if "," in spec:
            return name, [int(x) for x in spec.split(",")]
        if spec.isdigit():
            return name, [int(spec)]
    except ValueError:
        pass
    return model_spec, [0]


def _expand_instances(models, instance: int | None, instances: int | None) -> list[tuple[str, int]]:
    pairs = []
    for spec in models:
        name, ids = parse_model_instances(spec)
        if instances is not None:
            ids = list(range(instances))
        elif instance is not None:
            ids = [instance]
        pairs += [(name, i) for i in ids]
    return pairs


# --- getting started ------------------------------------------------------------------------------


@main.command()
@click.option("--force", is_flag=True, help="Overwrite existing files.")
@click.option("--templates", "copy_templates", is_flag=True, help="Also copy the job templates for editing.")
def init(force: bool, copy_templates: bool):
    """Create the config files, with an entry for each cluster in rclust's config.

    Writes `model_registry.yaml`, `resources.yaml` and `clusters.yaml` to `~/.config/slurm-llm/`
    (existing files are kept unless `--force`). Clusters come from rclust's config
    (`~/.config/rclust/config.yaml`); create it first with `rclust config`.
    """
    import shutil

    import yaml

    from .config import PACKAGE_DIR, rclust_config_path, rclust_gpu_types, read_hf_token, tilde, user_template_dir

    loader = _loader()
    config_dir = loader.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, content: str | Path):
        path = config_dir / name
        if path.exists() and not force:
            console.print(f"  [dim]kept[/dim]     {tilde(path)}")
            return
        if isinstance(content, Path):
            shutil.copy(content, path)
        else:
            path.write_text(content)
        console.print(f"  [green]wrote[/green]    {tilde(path)}")

    console.print("[bold]Configuration[/bold]")
    write("model_registry.yaml", PACKAGE_DIR / "configs" / "model_registry.yaml")
    write("resources.yaml", PACKAGE_DIR / "configs" / "resources.yaml")

    rclust_path = rclust_config_path()
    missing_rclust = not rclust_path.exists()
    if missing_rclust:
        console.print(
            f"  [yellow]No cluster config at {tilde(rclust_path)}.[/yellow] Define your clusters with "
            "`rclust config`, then run `sllm init` again to create clusters.yaml."
        )
    else:
        rclust_clusters = (yaml.safe_load(rclust_path.read_text()) or {}).get("clusters") or {}
        lines = [
            "# slurm-llm settings for each cluster in rclust's config; see",
            f"# {tilde(PACKAGE_DIR / 'configs' / 'clusters.example.yaml')} for all options.",
            "clusters:",
        ]
        for name, base in rclust_clusters.items():
            lines += [
                f"  {name}:",
                "    remote_root: ~/slurm-llm",
                "    model_path:          # required: directory with room for model weights, e.g. $SCRATCH/models",
                "    setup_commands: []   # e.g. [\"module load python/3.12 cuda/12.4\"]",
            ]
            gpu_types = rclust_gpu_types(base)
            if gpu_types:
                lines.append("    gpu_types:           # memory (GB) of each GPU type in rclust's config")
                for g in gpu_types:
                    lines += [f"      - type: {g.get('type')}", f"        memory: {g.get('memory', '')}"]
        write("clusters.yaml", "\n".join(lines) + "\n")

    if copy_templates:
        from .template_source import record_copy

        template_dir = user_template_dir()
        template_dir.mkdir(parents=True, exist_ok=True)
        console.print("[bold]Templates[/bold]")
        for src in sorted((PACKAGE_DIR / "templates").glob("*")):
            if src.suffix in (".slurm", ".sh"):
                dst = template_dir / src.name
                if dst.exists() and not force:
                    console.print(f"  [dim]kept[/dim]     {dst}")
                else:
                    shutil.copy(src, dst)
                    record_copy(template_dir, src.name)  # so an unedited copy can follow future versions
                    console.print(f"  [green]wrote[/green]    {dst}")

    token = read_hf_token()
    console.print(
        "[bold]Hugging Face token[/bold]: "
        + ("found" if token else "none (needed only for gated models; `sllm config hf-token`)")
    )
    console.print(
        "\n[bold]Next:[/bold]\n"
        + ("  0. rclust config                 define your clusters\n" if missing_rclust else "")
        + f"  1. edit {tilde(config_dir / 'clusters.yaml')} (or `sllm config cluster`): set model_path\n"
        "  2. sllm connect             log in to your clusters\n"
        "  3. sllm setup               create the Python environment on each cluster\n"
        "  4. sllm download MODEL      download weights, e.g. qwen-3-8b\n"
        "  5. sllm prepare MODEL       start a server; then `sllm proxy`"
    )


@main.command()
@click.argument("clusters", nargs=-1)
@click.option("--persist", "-p", default=None, metavar="DURATION",
              help="How long an idle connection stays open (ssh ControlPersist, e.g. 30m, 4h). Default: 1h.")
@click.option("--status", is_flag=True, help="Only show which clusters are connected.")
def connect(clusters: tuple, persist: str | None, status: bool):
    """Log in to clusters once and keep the connections open.

    Answers to password and two-factor prompts are needed only here; other commands reuse the
    open connection (ssh ControlMaster), as does `rclust`. Without arguments, connects to all clusters.

    ```bash
    sllm connect                  # all clusters
    sllm connect cluster-a -p 4h  # one cluster, idle for up to 4 hours
    sllm connect --status
    ```
    """
    from rich.table import Table

    from .ssh import SSHClient

    targets = _select_clusters(_loader(), clusters)
    if status:
        table = Table()
        table.add_column("Cluster", style="cyan")
        table.add_column("Host", style="dim")
        table.add_column("Status")
        for name, cfg in targets.items():
            up = SSHClient(cfg).has_active_connection()
            table.add_row(name, cfg.host, "[green]connected[/green]" if up else "[dim]not connected[/dim]")
        console.print(table)
        return

    failed = []
    for name, cfg in targets.items():
        client = SSHClient(cfg)
        try:
            if client.has_active_connection():
                console.print(f"[green]✓[/green] {name}: already connected")
            else:
                client.connect(interactive=True, persist=persist)
                console.print(f"[green]✓[/green] {name}: connected")
            # checked now, so a broken setup shows up before a job has waited in the queue
            problem = client.slurm_problem()
        except Exception as e:
            err_console.print(f"[red]✗[/red] {name}: {e}", highlight=False)
            failed.append(name)
            continue
        if problem:
            err_console.print(f"[red]✗[/red] {name}: Slurm: {problem}", highlight=False)
            failed.append(name)
    if failed:
        sys.exit(1)


@main.command()
@click.option("--cluster", "-c", "cluster_names", multiple=True, help="Cluster to set up (repeatable). Default: all.")
def setup(cluster_names: tuple):
    """Create the Python environment for vLLM on each cluster.

    On each cluster: creates `remote_root` and `model_path`, installs vLLM and `huggingface_hub` into
    `<model_path>/.venv` (using the cluster's `setup_commands`), and, if you have a Hugging Face token,
    logs in to Hugging Face there. Safe to run again.
    """
    from rich.table import Table

    from .config import read_hf_token, tilde
    from .ssh import shell_path
    from .template_source import resolve_template

    loader = _loader()
    targets = _select_clusters(loader, cluster_names)
    token = read_hf_token()
    results = {}
    for name, cfg in targets.items():
        console.print(f"\n[bold cyan]── {name} ──[/bold cyan]")
        try:
            model_path = _require_model_path(cfg)
            venv = f"{model_path}/.venv"
            source = resolve_template(cfg.setup_template)
            template = source.path
            if source.yours:
                console.print(f"  [dim]Setup script from {source.describe()}[/dim]")
            try:
                script = template.read_text().format(
                    venv_path=shell_path(venv), setup_commands="\n".join(cfg.setup_commands)
                )
            except (KeyError, IndexError, ValueError) as e:
                fail(
                    f"{tilde(template)} uses an unknown placeholder {e} (available: venv_path, setup_commands; "
                    "the Hugging Face login is now done separately). Update it, or delete it to use the packaged one."
                )
            client = _connect(cfg)
            for path in (cfg.remote_root, model_path):
                console.print(f"  [green]✓[/green] {client.resolve_dir(path)}")
            code = client.execute_command_streaming(f"bash -l <<'SLURM_LLM_SETUP'\n{script}\nSLURM_LLM_SETUP", timeout=1800)
            if code != 0:
                raise RuntimeError(f"setup script failed (exit code {code}); see its output above")
            # A script can exit 0 without having installed anything (e.g. a custom template without
            # `set -e`), so check the environment the way jobs use it before reporting "ready".
            check = "\n".join([*cfg.setup_commands, f"{shell_path(venv)}/bin/python -c 'import vllm, huggingface_hub'"])
            code, _, err = client.execute_command(check, timeout=300)
            if code != 0:
                detail = err.strip().splitlines()[-1] if err.strip() else f"exit code {code}"
                raise RuntimeError(f"vLLM is not importable from {venv} after setup: {detail}")
            hf = "skipped (no token)"
            if token:
                # the token goes over stdin, so it never appears in a command line on the cluster
                code, _, err = client.execute_command(
                    f"{shell_path(venv)}/bin/python -c 'import sys; from huggingface_hub import login; "
                    "login(token=sys.stdin.readline().strip())'",
                    input=token + "\n", timeout=60,
                )
                hf = "logged in" if code == 0 else f"failed: {err.strip().splitlines()[-1] if err.strip() else code}"
            _, df, _ = client.execute_command(f"df -hP {shell_path(model_path)} | tail -1", timeout=15)
            results[name] = ("ready", hf, disk_summary(df))
        except Exception as e:
            err_console.print(f"  [red]✗[/red] {e}", highlight=False)
            results[name] = ("[red]failed[/red]", "", "")

    table = Table(title="Setup")
    for column in ("Cluster", "Environment", "Hugging Face"):
        table.add_column(column)
    table.add_column("Disk (model_path)", overflow="fold", min_width=24)
    for name, row in results.items():
        table.add_row(name, *row)
    console.print(table)
    if any(row[0] != "ready" for row in results.values()):
        sys.exit(1)


def disk_summary(df_line: str) -> str:
    """"fs 1.0T 400G 600G 40% /scratch" (a line of `df -hP`) -> "600G free of 1.0T (40% used)"."""
    parts = df_line.split()
    if len(parts) >= 5 and parts[4].endswith("%"):
        return f"{parts[3]} free of {parts[1]} ({parts[4]} used)"
    return df_line.strip()


def _scan_and_save(loader, client, model, cfg, force: bool = False) -> str:
    """Scan a downloaded model's architecture into model_specs.yaml. Returns a status word."""
    from .model_scanner import scan_model_specs

    if not force and loader.get_model_specs(model.id):
        return "already scanned"
    specs = scan_model_specs(client, model.id, cfg)
    if not specs:
        return "not found"
    loader.save_model_specs(specs)
    return f"weights {specs.model_memory_gb:.1f} GB, context {specs.max_position_embeddings:,}"


@main.command()
@click.argument("models", nargs=-1)
@click.option("--cluster", "-c", "cluster_names", multiple=True, help="Cluster to download to (repeatable). Default: all.")
@click.option("--size", "-s", help="Download every registry model of this size.")
def download(models: tuple, cluster_names: tuple, size: str | None):
    """Download models to the clusters' model_path with the Hugging Face CLI.

    MODELS are registry names or Hugging Face repo ids. The download runs on the login node; models
    in the registry are then scanned for GPU memory estimates.

    ```bash
    sllm download qwen-3-8b --cluster cluster-a
    sllm download Qwen/Qwen3-8B
    sllm download --size small
    ```
    """
    import shlex

    from .ssh import shell_path

    loader = _loader()
    if size:
        models = tuple(name for name, m in loader.load_models().items() if m.size == size and name != m.id)
        if not models:
            fail(f"No registry models of size '{size}'")
    elif not models:
        fail("Name the models to download, or use --size")
    targets = _select_clusters(loader, cluster_names)

    failures = 0
    for name, cfg in targets.items():
        console.print(f"\n[bold cyan]── {name} ──[/bold cyan]")
        model_path = _require_model_path(cfg)
        client = _connect(cfg)
        hf = f"{shell_path(model_path + '/.venv')}/bin/hf"
        if client.execute_command(f"test -x {hf}")[0] != 0:
            err_console.print(f"  [red]✗[/red] no environment at {model_path}/.venv; run `sllm setup -c {name}`")
            failures += len(models)
            continue
        for spec in models:
            model = loader.resolve_model(spec)
            repo = model.hf_repo if model else spec
            if not model and "/" not in spec:
                err_console.print(f"  [red]✗[/red] {spec}: not in the registry, and not a Hugging Face repo id (org/name)")
                failures += 1
                continue
            console.print(f"[bold]{repo}[/bold]")
            code = client.execute_command_streaming(
                f"{hf} download {shlex.quote(repo)} --cache-dir {shell_path(model_path)}", timeout=6 * 3600
            )
            if code != 0:
                err_console.print(f"  [red]✗[/red] download failed (exit code {code})")
                failures += 1
                continue
            console.print("  [green]✓[/green] downloaded")
            if model:
                console.print(f"  specs: {_scan_and_save(loader, client, model, cfg)}")
            else:
                console.print("  [dim]Add it to model_registry.yaml to serve it.[/dim]")
    if failures:
        sys.exit(1)


@main.command()
@click.argument("models", nargs=-1)
@click.option("--cluster", "-c", help="Cluster to read the models from. Default: the first configured.")
@click.option("--force", "-f", is_flag=True, help="Re-scan models that were scanned before.")
def scan(models: tuple, cluster: str | None, force: bool):
    """Read downloaded models' config.json to estimate their GPU memory needs.

    Results go to `~/.config/slurm-llm/model_specs.yaml` and decide how many GPUs a model gets.
    Without MODELS, scans every registry model.
    """
    loader = _loader()
    clusters = loader.load_clusters()
    cfg = loader.get_cluster(cluster) if cluster else next(iter(clusters.values()))
    _require_model_path(cfg)
    registry = loader.load_models()
    targets = [loader.require_model(m) for m in models] or list({m.id: m for m in registry.values()}.values())
    client = _connect(cfg)
    counts = {"scanned": 0, "skipped": 0, "missing": 0}
    for model in targets:
        status = _scan_and_save(loader, client, model, cfg, force=force)
        key = "skipped" if status == "already scanned" else "missing" if status == "not found" else "scanned"
        counts[key] += 1
        style = {"scanned": "green", "skipped": "dim", "missing": "yellow"}[key]
        console.print(f"[{style}]{model.served_name}: {'not downloaded on ' + cfg.name if key == 'missing' else status}[/{style}]")
    console.print(f"\n{counts['scanned']} scanned, {counts['skipped']} already scanned, {counts['missing']} not downloaded")


# --- running models --------------------------------------------------------------------------------


@main.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--force-cluster", "-c", help="Run on this cluster instead of choosing one.")
@click.option("--time", "-t", help="Time limit, e.g. 4:00:00 (default: from resources.yaml).")
@click.option("--gpus", "-g", help="GPUs: COUNT or TYPE:COUNT, e.g. 2 or h100:2 (default: sized to the model).")
@click.option("--instance", "-i", type=int, help="Instance number (to run several copies of a model).")
@click.option("--instances", "-n", type=int, help="Submit instances 0..N-1.")
def submit(models: tuple, force_cluster: str | None, time: str | None, gpus: str | None,
           instance: int | None, instances: int | None):
    """Submit vLLM jobs without waiting for them (see `prepare` to wait until ready).

    MODEL:N selects instance N; MODEL:0-4 or MODEL:0,2 several.

    ```bash
    sllm submit qwen-3-8b
    sllm submit qwen-3-8b --instances 4
    sllm submit llama-3.3-70b-instruct --gpus h100:4 --force-cluster cluster-a
    ```
    """
    from .job import JobManager
    from .scheduler import ClusterScheduler, parse_gpu_override
    from .ssh import shell_path
    from .ssh_manager import SSHConnectionManager

    loader = _loader()
    loader.load_clusters()
    conn_manager = SSHConnectionManager(loader)
    scheduler = ClusterScheduler(loader, conn_manager)
    job_manager = JobManager()
    pairs = _expand_instances(models, instance, instances)

    submitted, failed = [], []
    for model_name, inst in pairs:
        label = f"{model_name}:{inst}" if inst else model_name
        try:
            model = loader.require_model(model_name)
            resource = loader.resource_for(model)
            if time:
                resource.time = time
            cluster, wait = scheduler.select_cluster_for_model(
                model, resource, model_name=model_name, explicit_cluster=force_cluster
            )
            cfg = loader.get_cluster(cluster)
            gpu_spec = (
                parse_gpu_override(cfg, gpus) if gpus else scheduler.match_gpu_for_model(cfg, model, resource)
            )
            client = conn_manager.get_client(cluster)
            job_id = job_manager.submit_job(model, resource, cfg, client, gpu_spec=gpu_spec, instance=inst)
            remote_root = client.resolve_dir(cfg.remote_root)
            client.execute_command(f"mkdir -p {shell_path(f'{remote_root}/manifests/{model.served_name}/nodes')}")
            gpu = f"{gpu_spec['num_gpus']}x {gpu_spec['type'] or 'GPU'}" if gpu_spec else "1 GPU"
            est = f"{wait // 60} min" if wait is not None else "unknown"
            console.print(f"[green]✓[/green] {label}: job {job_id} on {cluster} ({gpu}, est. wait {est})")
            if job_manager.last_template and job_manager.last_template.yours:
                console.print(f"  [dim]Job script from {job_manager.last_template.describe()}[/dim]")
            submitted.append(job_id)
        except Exception as e:
            from .config import SlurmLLMError

            if not isinstance(e, (SlurmLLMError, OSError)):
                raise
            err_console.print(f"[red]✗[/red] {label}: {e}", highlight=False)
            failed.append(label)

    if submitted:
        console.print("\n[dim]Check progress with `sllm jobs`; `sllm proxy` serves the models once RUNNING.[/dim]")
    if failed:
        sys.exit(1)


@main.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--force-cluster", "-c", help="Run on this cluster instead of choosing one.")
@click.option("--exclude", "-x", multiple=True, help="Never choose this cluster (repeatable).")
@click.option("--time", "-t", help="Time limit for new jobs, e.g. 4:00:00.")
@click.option("--instance", "-i", type=int, help="Instance number (to run several copies of a model).")
@click.option("--instances", "-n", type=int, help="Prepare instances 0..N-1.")
def prepare(models: tuple, force_cluster: str | None, exclude: tuple, time: str | None,
            instance: int | None, instances: int | None):
    """Start models (or reuse running jobs) and wait until they answer requests.

    For each model: reuses a running or queued job if there is one, otherwise picks a cluster
    and GPUs and submits a job; then waits for Slurm to start it and for vLLM to load.

    ```bash
    sllm prepare qwen-3-8b
    sllm prepare qwen-3-8b:0-3              # four instances
    sllm prepare llama-3.3-70b-instruct -x cluster-b
    ```
    """
    from rich.table import Table

    from .orchestration import JobOrchestrator, OrchestratorConfig

    pairs = _expand_instances(models, instance, instances)
    loader = _loader()
    loader.load_clusters()
    for model_name, _ in pairs:
        loader.require_model(model_name)
    config = OrchestratorConfig(show_progress=True, exclude_clusters=list(exclude) or None)
    ready = []
    with JobOrchestrator(config=config) as orchestrator:
        for i, (model_name, inst) in enumerate(pairs, 1):
            label = f"{model_name}:{inst}" if inst else model_name
            console.print(f"\n[bold cyan]Model {i}/{len(pairs)}: {label}[/bold cyan]")
            job = orchestrator.ensure_model_ready(
                model_name, cluster_name=force_cluster, time_override=time,
                exclude_clusters=list(exclude) or None, instance=inst,
            )
            ready.append((label, job))

    table = Table(title="Ready")
    for column in ("Model", "Job", "Cluster", "Endpoint (on the cluster)"):
        table.add_column(column)
    for label, job in ready:
        table.add_row(label, job.job_id, job.cluster_name, f"{job.node}:{job.local_port}")
    console.print(table)
    console.print("\nNext: `sllm proxy`, then send OpenAI API requests to http://localhost:8000/v1")


@main.command()
@click.option("--port", "-p", default=8000, show_default=True, help="Local port to listen on.")
def proxy(port: int):
    """Serve all RUNNING models on one local OpenAI-compatible endpoint.

    Requests are routed by their `model` field, through SSH tunnels to the compute nodes.
    Models that start running later are picked up on their first request.
    """
    import asyncio

    from .api_proxy import run_proxy_server

    asyncio.run(run_proxy_server(port))


@main.command()
@click.argument("model_name")
@click.argument("prompt", required=False)
@click.option("--system", "-s", help="System prompt.")
@click.option("--max-tokens", "-n", default=1024, show_default=True, help="Maximum tokens to generate.")
@click.option("--temperature", "-t", default=0.7, show_default=True)
@click.option("--top-p", default=1.0, show_default=True)
@click.option("--stream/--no-stream", default=True, help="Stream the answer as it is generated.")
@click.option("--proxy-url", default="http://localhost:8000", show_default=True)
def query(model_name, prompt, system, max_tokens, temperature, top_p, stream, proxy_url):
    r"""Ask a model a question through the proxy, or chat with it if PROMPT is omitted.

    In chat mode, `/help` lists commands (`/system`, `/clear`, `/model`, `/set`, `/show`, `/exit`).

    ```bash
    sllm query qwen-3-8b "Explain tensor parallelism in two sentences"
    sllm query qwen-3-8b
    ```
    """
    import httpx

    try:
        httpx.get(f"{proxy_url}/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        fail(f"No proxy at {proxy_url}; start one with `sllm proxy`")

    from openai import OpenAI

    from .chat import chat, query_once

    client = OpenAI(base_url=f"{proxy_url}/v1", api_key="unused")
    settings = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if prompt:
        query_once(client, model_name, prompt, system, settings, stream)
    else:
        chat(client, model_name, system, settings, stream, proxy_url)


# --- jobs ------------------------------------------------------------------------------------------


def _connected_clients(loader, names) -> dict:
    """Clients for the clusters that can be reached without prompting (warns about the others)."""
    from concurrent.futures import ThreadPoolExecutor

    from .ssh_manager import SSHConnectionManager

    manager = SSHConnectionManager(loader)
    clients = {}

    def get(name):
        try:
            return name, manager.get_client(name)
        except Exception as e:
            err_console.print(f"[yellow]⚠ {e}[/yellow]", highlight=False)
            return name, None

    with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        for name, client in pool.map(get, names):
            if client:
                clients[name] = client
    return clients


@main.command()
@click.option("--cluster", "-c", "cluster_names", multiple=True, help="Only refresh this cluster (repeatable).")
@click.option("--all", "show_all", is_flag=True, help="Include finished jobs older than 20 minutes.")
@click.option("--no-refresh", is_flag=True, help="Show the cached state without contacting the clusters.")
def jobs(cluster_names: tuple, show_all: bool, no_refresh: bool):
    """List your vLLM jobs (including ones submitted from other machines)."""
    from datetime import datetime, timedelta

    from rich.table import Table

    from .job import JobManager

    job_manager = JobManager()
    if no_refresh:
        all_jobs = job_manager.list_jobs(refresh=False)
    else:
        loader = _loader()
        clients = _connected_clients(loader, list(_select_clusters(loader, cluster_names)))
        job_manager.list_jobs(ssh_clients=clients, refresh=True)
        all_jobs = job_manager.discover_and_cache_vllm_jobs(clients, config_loader=loader)

    cutoff = datetime.now() - timedelta(minutes=20)

    def visible(job) -> bool:
        if show_all or job.status in ("RUNNING", "PENDING", "CONFIGURING"):
            return True
        try:
            return datetime.fromisoformat(job.submit_time).replace(tzinfo=None) > cutoff
        except ValueError:
            return True

    shown = {k: j for k, j in all_jobs.items() if visible(j)}
    if not shown:
        console.print("No jobs" + ("" if show_all else " (use --all to include older finished jobs)"))
        return
    table = Table(title="sllm jobs")
    for column, style in (("Job", "cyan"), ("Model", "green"), ("Cluster", "blue"), ("Status", "yellow"),
                          ("Node", "magenta"), ("Time left", None), ("Submitted", "dim")):
        table.add_column(column, style=style)
    for job_id, job in shown.items():
        model = f"{job.model_name}:{job.instance}" if job.instance else job.model_name
        table.add_row(job_id, model, job.cluster_name, job.status, job.node or "-", job.time_left or "-",
                      job.submit_time[:19].replace("T", " "))
    console.print(table)
    if len(all_jobs) > len(shown):
        console.print(f"[dim]{len(all_jobs) - len(shown)} older finished job(s) hidden; use --all[/dim]")


def _find_job(job_manager, job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        fail(f"Job {job_id} is not in the local job list (see `sllm jobs`)")
    return job


@main.command()
@click.option("--job", "-j", "job_id", help="Job id.")
@click.option("--latest", is_flag=True, help="Use the most recently submitted running job.")
@click.option("--wait", is_flag=True, help="Wait until the server is ready.")
@click.option("--timeout", "-t", default=600, show_default=True, help="Seconds to wait with --wait.")
def health(job_id: str | None, latest: bool, wait: bool, timeout: int):
    """Check whether a job's vLLM server is ready."""
    import time as _time

    from .health import detect_vllm_port, query_model_info
    from .job import JobManager

    if bool(job_id) == latest:
        raise click.UsageError("Give either --job ID or --latest")
    job_manager = JobManager()
    if latest:
        running = [j for j in job_manager.list_jobs(refresh=False).values() if j.status == "RUNNING"]
        if not running:
            fail("No running jobs (see `sllm jobs`)")
        job = max(running, key=lambda j: j.submit_time)
    else:
        job = _find_job(job_manager, job_id)

    client = _connect(_loader().get_cluster(job.cluster_name))
    status, node, _ = client.get_job_status(job.job_id)
    if status != "RUNNING" or not node:
        fail(f"Job {job.job_id} is not running (status: {status})")
    job.node = node
    port = detect_vllm_port(job, client, timeout=timeout if wait else 10)
    deadline = _time.time() + timeout
    while True:
        info = query_model_info(client, node, port)
        if info:
            console.print(f"[green]✓[/green] {info.get('id')} is ready at {node}:{port} (job {job.job_id})")
            return
        if not wait or _time.time() > deadline:
            fail(f"vLLM at {node}:{port} is not ready yet" + ("" if wait else " (use --wait to wait)"))
        _time.sleep(5)


@main.command()
@click.argument("job_ids", nargs=-1)
@click.option("--job", "-j", "job_opts", multiple=True, help="Job id (same as the argument).")
@click.option("--cluster", "-c", help="Cluster of a job that is not in the local job list.")
def cancel(job_ids: tuple, job_opts: tuple, cluster: str | None):
    """Cancel jobs (stops their vLLM servers)."""
    from .job import JobManager

    ids = list(job_ids) + list(job_opts)
    if not ids:
        fail("Name the job(s) to cancel")
    loader = _loader()
    job_manager = JobManager()
    failed = False
    for job_id in ids:
        job = job_manager.get_job(job_id)
        if not job and not cluster:
            fail(f"Job {job_id} is not in the local job list; pass --cluster")
        client = _connect(loader.get_cluster(cluster or job.cluster_name))
        if client.cancel_job(job_id):
            console.print(f"[green]✓[/green] cancelled {job_id}")
            job_manager.update_job_status(job_id, client)
        else:
            err_console.print(f"[red]✗[/red] could not cancel {job_id}")
            failed = True
    if failed:
        sys.exit(1)


@main.command(name="clean")
@click.option("--all", "clean_all", is_flag=True, help="Forget every job, not only finished ones.")
@click.option("--older-than", type=int, metavar="DAYS", help="Also forget jobs submitted more than DAYS ago.")
def clean_jobs(clean_all: bool, older_than: int | None):
    """Forget finished jobs in the local job list (does not touch the clusters)."""
    from datetime import datetime, timedelta

    from .job import ACTIVE_STATES, JobManager

    job_manager = JobManager()
    all_jobs = job_manager._load_jobs()
    cutoff = datetime.now() - timedelta(days=older_than) if older_than is not None else None

    def forget(job) -> bool:
        if clean_all or job.status not in ACTIVE_STATES:
            return True
        try:
            return cutoff is not None and datetime.fromisoformat(job.submit_time) < cutoff
        except ValueError:
            return False

    keep = {k: j for k, j in all_jobs.items() if not forget(j)}
    job_manager._save_jobs(keep)
    console.print(f"Removed {len(all_jobs) - len(keep)} job(s); {len(keep)} left")


@main.command()
@click.option("--cluster", "-c", help="Also show which models are downloaded on this cluster.")
@click.option("--filter", "filters", multiple=True, metavar="KEY:VALUES",
              help="size:VALUES or tags:VALUES, values comma-separated (repeatable).")
@click.option("--full", is_flag=True, help="Show full model ids.")
def models(cluster: str | None, filters: tuple, full: bool):
    """List the models in the registry.

    ```bash
    sllm models --filter size:small,tiny --filter tags:reasoning
    ```
    """
    from rich.table import Table

    from .ssh import shell_path

    loader = _loader()
    registry = {name: m for name, m in loader.load_models().items() if name != m.id}
    wanted = {}
    for f in filters:
        key, sep, values = f.partition(":")
        if not sep or key not in ("size", "tags"):
            fail(f"Invalid filter '{f}': use size:VALUES or tags:VALUES")
        wanted[key] = set(values.split(","))
    if "size" in wanted:
        registry = {n: m for n, m in registry.items() if m.size in wanted["size"]}
    if "tags" in wanted:
        registry = {n: m for n, m in registry.items() if wanted["tags"] & set(m.tags)}

    downloaded = None
    if cluster:
        cfg = loader.get_cluster(cluster)
        client = _connect(cfg)
        _, out, _ = client.execute_command(f"ls -1 {shell_path(_require_model_path(cfg))} 2>/dev/null")
        downloaded = set(out.split())

    table = Table(title="Models")
    table.add_column("Name", style="cyan")
    if full:
        table.add_column("Id", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Tags", style="magenta")
    if downloaded is not None:
        table.add_column(f"On {cluster}", style="green")
    for name, m in registry.items():
        row = [name] + ([m.id] if full else []) + [m.size, ", ".join(m.tags)]
        if downloaded is not None:
            row.append("✓" if m.id in downloaded else "")
        table.add_row(*row)
    console.print(table)


@main.command()
@click.option("--port", "-p", "ports", multiple=True, type=int, help="Stop what listens on this port (repeatable).")
@click.option("--all", "kill_all", is_flag=True, help="Stop every `sllm proxy` process.")
def kill(ports: tuple, kill_all: bool):
    """Stop running proxies (the cluster jobs keep running; see `cancel`)."""
    import psutil

    if not ports and not kill_all:
        kill_all = True
    targets = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
            if kill_all and "proxy" in cmd and any(name in c for c in cmd for name in ("sllm", "slurm_llm")):
                targets[proc.pid] = proc
            elif ports and any(c.status == psutil.CONN_LISTEN and c.laddr.port in ports for c in proc.net_connections()):
                targets[proc.pid] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    targets.pop(os.getpid(), None)
    if not targets:
        console.print("No proxy running")
        return
    for pid, proc in targets.items():
        proc.terminate()  # the proxy cancels its port forwards on SIGTERM
        try:
            proc.wait(timeout=20)
        except psutil.TimeoutExpired:
            proc.kill()
        console.print(f"[green]✓[/green] stopped process {pid}")

    from .tunnels import TunnelRegistry

    cleaned = TunnelRegistry().cleanup_stale_tunnels()  # forwards of processes that did not clean up
    if cleaned:
        console.print(f"[green]✓[/green] cancelled {cleaned} leftover port forward(s)")


# --- configuration ---------------------------------------------------------------------------------


@main.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """View, check and edit the configuration.

    ```bash
    sllm config view
    sllm config validate --check-ssh
    sllm config cluster cluster-a
    ```
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(config_view)
        console.print("\n[dim]Subcommands: view, validate, cluster, model, preferences, hf-token[/dim]")


@config.command("view")
def config_view():
    """Show where the configuration lives and what it contains."""
    from .config import read_hf_token, tilde

    loader = _loader()
    overlay = loader.overlay_path()
    console.print(
        f"[bold]Clusters[/bold] ({tilde(loader.rclust_config_path())} + {tilde(overlay) if overlay else 'no clusters.yaml'})"
    )
    try:
        for name, cfg in loader.load_clusters().items():
            gpus = ", ".join(f"{g.type} {g.memory}GB" for g in cfg.gpu_types) or "no GPU types with memory"
            console.print(f"  {name}: {cfg.host}; model_path {cfg.model_path or '[red]not set[/red]'}; {gpus}")
    except Exception as e:
        console.print(f"  [red]{e}[/red]")
    try:
        registry = loader.load_models()
        console.print(f"[bold]Models[/bold] ({tilde(loader.config_file('model_registry.yaml'))}): "
                      f"{len({m.id for m in registry.values()})}")
        console.print(f"[bold]Resources[/bold] ({tilde(loader.config_file('resources.yaml'))}): "
                      f"{', '.join(loader.load_resources())}")
    except Exception as e:
        console.print(f"  [red]{e}[/red]")
    console.print(f"[bold]Hugging Face token[/bold]: {'set' if read_hf_token() else 'not set'}")


@config.command("validate")
@click.option("--check-ssh", is_flag=True, help="Also check the SSH connection to each cluster.")
def config_validate(check_ssh: bool):
    """Check the configuration for mistakes (exits 1 if there are errors)."""
    from .config import read_hf_token
    from .template_source import resolve_template

    loader = _loader()
    errors, warnings, templates = [], [], []
    clusters = {}
    try:
        clusters = loader.load_clusters()
    except Exception as e:
        errors.append(str(e))
    overlay_file = loader.overlay_path()
    if clusters and overlay_file:
        import yaml

        from .config import tilde

        try:
            overlay = (yaml.safe_load(overlay_file.read_text()) or {}).get("clusters") or {}
        except (OSError, yaml.YAMLError, AttributeError):
            overlay = {}
        for name in overlay:
            if name not in clusters:
                warnings.append(
                    f"clusters.yaml: cluster '{name}' is not in rclust's config ({tilde(loader.rclust_config_path())}), "
                    "so it is ignored; add it with `rclust config` or check the spelling"
                )
        for name in clusters:
            if name not in overlay:
                warnings.append(f"cluster '{name}' (from rclust's config) has no entry in clusters.yaml; "
                                f"`sllm config cluster {name}` adds one")
    for name, cfg in clusters.items():
        if not cfg.model_path:
            errors.append(f"cluster '{name}': model_path is not set")
        if not cfg.gpu_types:
            warnings.append(
                f"cluster '{name}': no GPU type with a memory (GB) in clusters.yaml, so every job asks for one GPU"
            )
        if cfg.gpu_allocation_mode not in ("per-node", "flexible"):
            errors.append(f"cluster '{name}': gpu_allocation_mode must be per-node or flexible")
        for kind, template in (("job", cfg.slurm_template), ("setup", cfg.setup_template)):
            try:
                source = resolve_template(template, update=False)
            except Exception as e:
                errors.append(f"cluster '{name}': {e}")
                continue
            templates.append(f"{name}: {kind} template {template}: {source.describe()}")
    try:
        resources = loader.load_resources()
        for model in {m.id: m for m in loader.load_models().values()}.values():
            if model.size not in resources:
                errors.append(f"model {model.id}: size '{model.size}' is not in resources.yaml")
    except Exception as e:
        errors.append(str(e))
    try:
        for size, names in loader.load_cluster_preferences().items():
            for name in names or []:
                if name != "any" and clusters and name not in clusters:
                    warnings.append(f"cluster_preferences.yaml: unknown cluster '{name}' for size {size}")
    except Exception as e:
        errors.append(str(e))
    if not read_hf_token():
        warnings.append("no Hugging Face token (needed only for gated models): `sllm config hf-token`")
    if check_ssh:
        from .ssh import SSHClient

        for name, cfg in clusters.items():
            try:
                client = SSHClient(cfg)
                client.connect(interactive=False)
                console.print(f"[green]✓[/green] {name}: SSH works")
            except Exception as e:
                errors.append(str(e))

    for t in templates:
        console.print(f"[dim]{t}[/dim]", highlight=False)
    for w in warnings:
        console.print(f"[yellow]warning:[/yellow] {w}", highlight=False)
    for e in errors:
        console.print(f"[red]error:[/red] {e}", highlight=False)
    if errors:
        sys.exit(1)
    console.print("[green]✓ Configuration OK[/green]")


@config.command("cluster")
@click.argument("cluster_name", required=False)
def config_cluster(cluster_name: str | None):
    """Set a cluster's slurm-llm settings interactively (written to clusters.yaml)."""
    import yaml
    from rich.prompt import Prompt

    loader = _loader()
    clusters = loader.load_clusters()
    if not cluster_name:
        cluster_name = Prompt.ask("Cluster", choices=list(clusters), default=next(iter(clusters)))
    if cluster_name not in clusters:
        fail(f"Unknown cluster '{cluster_name}' (configured: {', '.join(clusters)}); add clusters with `rclust config`")

    overlay_file = loader.overlay_path()
    data = (yaml.safe_load(overlay_file.read_text()) or {}) if overlay_file else {}
    data.setdefault("clusters", {})
    current = data["clusters"].get(cluster_name) or {}
    cfg = clusters[cluster_name]

    console.print(f"[bold]{cluster_name}[/bold] ({cfg.host}). `~` and $VARIABLES are expanded on the cluster.")
    entry = dict(current)
    entry["remote_root"] = Prompt.ask("Directory for job scripts and logs", default=cfg.remote_root)
    entry["model_path"] = Prompt.ask(
        "Directory for model weights (needs lots of space)", default=cfg.model_path or None
    )
    setup_line = Prompt.ask(
        "Setup commands, e.g. module load python/3.12 cuda/12.4 (empty for none)",
        default="; ".join(cfg.setup_commands),
    )
    entry["setup_commands"] = [setup_line] if setup_line.strip() else []
    entry["gpu_allocation_mode"] = Prompt.ask(
        "Request GPUs per node (--gpus-per-node) or anywhere (--gpus)",
        choices=["per-node", "flexible"], default=cfg.gpu_allocation_mode,
    )
    from .config import rclust_gpu_types

    rclust_cluster = (yaml.safe_load(loader.rclust_config_path().read_text()) or {})["clusters"][cluster_name]
    known = {g.type: g.memory for g in cfg.gpu_types}
    gpu_types = []
    for g in rclust_gpu_types(rclust_cluster):
        memory = Prompt.ask(f"Memory of one {g['type']} GPU in GB", default=str(known.get(g["type"], "")) or None)
        if memory and memory.isdigit():
            gpu_types.append({"type": g["type"], "memory": int(memory)})
    if gpu_types:
        entry["gpu_types"] = gpu_types
    data["clusters"][cluster_name] = entry

    path = loader.config_dir / "clusters.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False))
    console.print(f"[green]✓[/green] saved to {path}")


@config.command("model")
@click.argument("model_name")
def config_model(model_name: str):
    """Set a model's context window and GPU memory override."""
    import shutil

    import yaml
    from rich.prompt import Prompt

    loader = _loader()
    model = loader.require_model(model_name)
    path = loader.config_dir / "model_registry.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(loader.config_file("model_registry.yaml"), path)
    registry = yaml.safe_load(path.read_text()) or {}
    entry = next(m for m in registry.get("models", []) if m.get("id") == model.id)

    updates = {}
    for key, question in (("max_model_len", "Context window, max_model_len (empty: the model's own)"),
                          ("model_memory_gb", "Total GPU memory to request in GB (empty: estimate)")):
        value = Prompt.ask(question, default=str(entry.get(key) or ""))
        if value and not value.isdigit():
            fail(f"{key} must be a whole number")
        updates[key] = int(value) if value else None
    for key, value in updates.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    path.write_text(yaml.dump(registry, default_flow_style=False, sort_keys=False))
    console.print(f"[green]✓[/green] updated {entry.get('short_name', model.id)} in {path}")


@config.command("preferences")
def config_preferences():
    """Choose which clusters to consider for each model size."""
    import yaml
    from rich.prompt import Prompt

    loader = _loader()
    clusters = list(loader.load_clusters())
    sizes = list(loader.load_resources())
    prefs = loader.load_cluster_preferences()
    console.print(f"Clusters: {', '.join(clusters)}. Enter names in order of preference, `any`, or leave empty.")
    for size in sizes:
        answer = Prompt.ask(f"  {size}", default=" ".join(prefs.get(size) or []))
        names = answer.replace(",", " ").split()
        unknown = [n for n in names if n != "any" and n not in clusters]
        if unknown:
            fail(f"Unknown cluster(s): {', '.join(unknown)}")
        prefs[size] = names
    path = loader.config_dir / "cluster_preferences.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({"cluster_preferences": {k: v for k, v in prefs.items() if v}}, sort_keys=False))
    console.print(f"[green]✓[/green] saved to {path}")


@config.command("hf-token")
def config_hf_token():
    """Save your Hugging Face token (used by `setup` to log in on the clusters).

    The token is stored in `~/.config/slurm-llm/.hf_token`, readable only by you. Alternatively
    set the HF_TOKEN environment variable. Get a token at https://huggingface.co/settings/tokens.
    """
    import getpass

    token = getpass.getpass("Hugging Face token (input hidden): ").strip()
    if not token:
        fail("No token entered")
    if not token.startswith("hf_"):
        err_console.print("[yellow]Warning: Hugging Face tokens usually start with 'hf_'[/yellow]")
    path = _loader().config_dir / ".hf_token"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    os.chmod(path, 0o600)
    console.print(f"[green]✓[/green] saved to {path}; run `sllm setup` to log in on the clusters")


if __name__ == "__main__":
    main()
