"""Waiting for a vLLM job: Slurm allocation, port discovery and server health."""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager

from rich.console import Console

from .config import ConfigError, env
from .job import JobInfo
from .ssh import SSHClient, StatusUnknownError, shell_path

logger = logging.getLogger(__name__)
console = Console()

# States in which a job may still start running.
WAITING_STATES = {"SUBMITTED", "PENDING", "CONFIGURING", "REQUEUED", "RESIZING", "SUSPENDED", "UNKNOWN"}

OOM_PATTERN = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory|\bOOM\b", re.IGNORECASE)

KV_CACHE_PATTERN = re.compile(r"KV cache|max seq len|max_model_len|maximum model length", re.IGNORECASE)


DEFAULT_BUSY_TIMEOUT = 120


def default_busy_timeout() -> int:
    """How long (seconds) a job's status may stay unknown, e.g. while Slurm is busy, before giving
    up: $SLURM_LLM_BUSY_TIMEOUT, or 2 minutes."""
    value = env("BUSY_TIMEOUT")
    if value is None:
        return DEFAULT_BUSY_TIMEOUT
    try:
        seconds = int(value)
    except ValueError:
        seconds = -1
    if seconds <= 0:
        raise ConfigError(f"SLURM_LLM_BUSY_TIMEOUT must be a number of seconds, not {value!r}")
    return seconds


def _duration(seconds: int) -> str:
    if seconds % 60:
        return f"{seconds} seconds"
    return "1 minute" if seconds == 60 else f"{seconds // 60} minutes"


class _StatusTimer:
    """Gives up once a job's status has been UNKNOWN for `limit` seconds in a row."""

    def __init__(self, job: JobInfo, ssh_client: SSHClient, limit: int | None):
        self.job, self.ssh_client, self.since = job, ssh_client, None
        self.limit = limit or default_busy_timeout()

    def update(self, status: str) -> None:
        if status != "UNKNOWN":
            self.since = None
            return
        now = time.time()
        self.since = self.since or now
        if now - self.since >= self.limit:
            reason = getattr(self.ssh_client, "last_status_error", None) or "no answer"
            raise StatusUnknownError(
                f"Could not check job {self.job.job_id} on {self.ssh_client.config.name} for "
                f"{_duration(self.limit)} ({reason}). The job has not been cancelled; check it later "
                f"with `sllm jobs`. To wait longer, set SLURM_LLM_BUSY_TIMEOUT (seconds)."
            )


@contextmanager
def _spinner(show: bool, text: str):
    if not show:
        yield None
        return
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(
        TextColumn("    "),
        SpinnerColumn(spinner_name="arc", style="white"),
        TextColumn(text),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(text, total=None)
        yield progress


def _check_job_errors(ssh_client: SSHClient, job_id: str, job: JobInfo | None = None) -> str:
    """Explain why a job failed, from the end of its Slurm logs."""
    err_path = ssh_client.get_remote_slurm_error_path(job_id)
    out_path = ssh_client.get_remote_slurm_log_path(job_id)
    exit_code, log, _ = ssh_client.execute_command(
        f"tail -100 {shell_path(err_path)} 2>/dev/null; tail -100 {shell_path(out_path)} 2>/dev/null", timeout=15
    )
    if not log.strip():
        return f"Could not read the job's logs ({out_path})"
    lines = log.strip().splitlines()
    where = f"  Check logs: {err_path} and {out_path}\n"

    if any(OOM_PATTERN.search(line) for line in lines):
        msg = "CUDA Out of Memory error detected. The model needs more GPU memory than it was given.\n" + where
        if job and job.gpu_info:
            g = job.gpu_info
            msg += f"  Assigned GPU: {g.get('num_gpus', 1)}x {g.get('type', 'unknown')} ({g.get('memory', '?')}GB each)\n"
        return msg + (
            "  Fixes: set model_memory_gb (total GPU memory to request) or a smaller max_model_len in "
            "model_registry.yaml, or use --force-cluster with a cluster that has larger GPUs.\n"
        )

    kv_lines = [line for line in lines if KV_CACHE_PATTERN.search(line)]
    if kv_lines:
        msg = "KV cache memory error: the context window does not fit in the GPU memory left.\n" + where
        suggested = re.search(r"estimated maximum model length is (\d+)", "\n".join(kv_lines), re.IGNORECASE)
        if suggested:
            msg += f"  Suggested max_model_len: {suggested.group(1)}\n"
        msg += "\n".join(f"    {line.strip()}" for line in kv_lines[-5:]) + "\n"
        return msg + "  Fix: set max_model_len in model_registry.yaml, or use a GPU with more memory.\n"

    for pattern, label in ((r"CUDA error|cuda runtime error", "CUDA error"), (r"Error:|Exception:|Traceback", "Error")):
        matched = [line for line in lines if re.search(pattern, line, re.IGNORECASE)]
        if matched:
            return f"{label} in the job's logs.\n{where}" + "\n".join(matched[-5:])
    return where + "\nLast lines:\n" + "\n".join(lines[-10:])


PORT_PATTERNS = [
    # written by the job template before vLLM starts, so it is the earliest reliable signal
    ("grep -E 'Port selection:.*actual=[0-9]+'", r"\bactual=(\d+)\b"),
    # vLLM's own messages: "Starting vLLM API server 0 on http://..." (<= 0.11),
    # "Starting vLLM server on http://..." (>= 0.19)
    ("grep -iE 'Starting vLLM (API )?server.*on http'", r"on http://[^:\s]+:(\d+)"),
    ("grep APIServer", r"['\"]port['\"]:\s*(\d+)"),
]


def detect_vllm_port(
    job: JobInfo,
    ssh_client: SSHClient,
    timeout: int = 480,
    interval: int = 5,
    show_progress: bool = False,
) -> int:
    """The port of this job's vLLM server, read from the job's own Slurm log.

    Other ports on the node are never probed: another model may be serving there. Falls back to
    the port already recorded for the job; raises RuntimeError if none is found in time.
    """
    log_path = shell_path(ssh_client.get_remote_slurm_log_path(job.job_id))
    start = time.time()
    with _spinner(show_progress, "Detecting vLLM port from logs..."):
        while True:
            for grep, pattern in PORT_PATTERNS:
                code, output, _ = ssh_client.execute_command(f"{grep} {log_path} 2>/dev/null | tail -1", timeout=10)
                ssh_client.check_connection(code, f"reading the log of job {job.job_id} (the job has not been cancelled)")
                match = re.search(pattern, output) if code == 0 else None
                if match:
                    return int(match.group(1))
            if time.time() - start >= timeout:
                break
            time.sleep(interval)

    if job.local_port:
        logger.warning(f"Could not re-detect the port of job {job.job_id}; using its recorded port {job.local_port}")
        return job.local_port
    raise RuntimeError(
        f"Could not find the vLLM port of job {job.job_id} in its log after {timeout}s "
        f"({ssh_client.get_remote_slurm_log_path(job.job_id)})"
    )


def wait_for_vllm_health(
    job: JobInfo,
    ssh_client: SSHClient,
    initial_interval: int = 5,
    max_interval: int = 30,
    show_progress: bool = True,
    busy_timeout: int | None = None,
) -> bool:
    """Wait until the job's vLLM server answers /health.

    Waits as long as the job is running (loading a large model can take a while); raises
    RuntimeError, with the likely cause from the logs, if the job ends or vLLM runs out of memory.
    """
    health_url = f"http://{job.node}:{job.local_port}/health"
    log_path = shell_path(ssh_client.get_remote_slurm_log_path(job.job_id))
    start = last_check = time.time()
    interval = initial_interval
    timer = _StatusTimer(job, ssh_client, busy_timeout)
    with _spinner(show_progress, f"Loading vLLM (job {job.job_id})..."):
        while True:
            if time.time() - last_check >= 30:
                last_check = time.time()
                status, _, _ = ssh_client.get_job_status(job.job_id)
                timer.update(status)
                if status not in {"RUNNING", "COMPLETING", "UNKNOWN"}:
                    raise RuntimeError(
                        f"Job {job.job_id} ended (status: {status}). {_check_job_errors(ssh_client, job.job_id, job)}"
                    )
                # vLLM can crash from OOM while the job keeps running
                code, output, _ = ssh_client.execute_command(
                    f"grep -iE 'out of memory|OutOfMemoryError' {log_path} 2>/dev/null | tail -1", timeout=10
                )
                if code == 0 and output.strip():
                    raise RuntimeError(
                        f"Job {job.job_id} ran out of GPU memory. {_check_job_errors(ssh_client, job.job_id, job)}"
                    )

            code, _, _ = ssh_client.execute_command(f"curl -sf -m 5 {health_url} >/dev/null", timeout=15)
            ssh_client.check_connection(code, f"waiting for vLLM to start (job {job.job_id} has not been cancelled)")
            if code == 0:
                if show_progress:
                    console.print(f"    ✓ vLLM server is healthy (loaded in {int(time.time() - start)}s)")
                return True
            time.sleep(interval)
            interval = min(interval * 1.5, max_interval)


def wait_for_job_running(
    job: JobInfo,
    ssh_client: SSHClient,
    job_manager,
    estimated_wait: int | None = 3600,
    poll_interval: int = 5,
    show_progress: bool = True,
    busy_timeout: int | None = None,
) -> bool:
    """Wait until Slurm starts the job. Returns False if the job ends without running.

    `estimated_wait` is informational only; this waits as long as the job is queued (Ctrl+C to stop).
    `busy_timeout` (default: `default_busy_timeout()`) limits how long the status may stay unknown.
    """
    previous = job.status
    timer = _StatusTimer(job, ssh_client, busy_timeout)
    with _spinner(show_progress, f"Waiting for Slurm (job {job.job_id})..."):
        while True:
            status, node, _ = ssh_client.get_job_status(job.job_id)
            timer.update(status)
            if status == "UNKNOWN":  # the check failed; keep the last known state (UNKNOWN is not cached)
                time.sleep(poll_interval)
                continue
            job.status = status
            if node:
                job.node = node
            if status != previous or (status == "RUNNING" and node):
                job_manager.save_job(job)
                previous = status
            if status == "RUNNING" and node:
                break
            if status not in WAITING_STATES | {"RUNNING"}:
                logger.info(f"Job {job.job_id} ended before running: {status}")
                return False
            time.sleep(poll_interval)
    if show_progress:
        console.print(f"    ✓ Job running on {job.node}")
    return True


def query_model_info(ssh_client: SSHClient, node: str, port: int, timeout: int = 10) -> dict | None:
    """The first entry of the server's /v1/models (includes max_model_len), or None."""
    code, output, _ = ssh_client.execute_command(f"curl -s -m {timeout} http://{node}:{port}/v1/models", timeout=timeout + 5)
    if code != 0:
        return None
    try:
        data = json.loads(output).get("data") or []
    except (json.JSONDecodeError, AttributeError):
        return None
    return data[0] if data else None
