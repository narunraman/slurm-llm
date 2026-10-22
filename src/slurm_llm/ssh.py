"""SSH access to clusters through OpenSSH connection sharing.

`sllm connect` (or `rclust connect`) logs in once, answering any password or two-factor
prompt, and leaves a background master connection (ssh ControlMaster). Every other command runs
over that connection without prompting. The socket path is the one rclust uses for the same host,
so a connection opened by either tool is reused by the other.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import ClusterConfig, ConfigError, RemoteLLMError

logger = logging.getLogger(__name__)

DEFAULT_PERSIST = "1h"  # how long an idle master connection stays open (ssh ControlPersist)
# Printed before a command in a login shell; anything a site profile prints before it is dropped
OUTPUT_MARKER = "SLURM_LLM_OUTPUT_START"


class NotConnectedError(RemoteLLMError):
    """No usable SSH connection to a cluster."""


def shell_path(path: str) -> str:
    """Quote a remote path for the shell, keeping `~` and `$VARS` (e.g. $SCRATCH) working."""
    if path == "~" or path.startswith("~/"):
        path = "$HOME" + path[1:]
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


class SSHClient:
    """Runs commands on one cluster over a shared SSH connection."""

    def __init__(self, cluster_config: ClusterConfig):
        from rclust.ssh import SSHClient as RclustSSHClient

        self.config = cluster_config
        try:
            transport = RclustSSHClient(cluster_config.host, cluster_config.user, name=cluster_config.name)
        except ValueError as e:
            raise ConfigError(f"cluster '{cluster_config.name}': {e}") from None
        self.destination = transport.destination
        self.ssh_master_socket = str(transport.socket_path)
        self._status_warning: str | None = None  # last "could not check" warning, so polls don't repeat it

    def _get_ssh_target(self) -> str:
        """The ssh destination ([user@]host or an alias from ~/.ssh/config)."""
        return self.destination

    def _ssh(self, *options: str) -> list[str]:
        return ["ssh", "-o", "ConnectTimeout=10", *options, "-S", self.ssh_master_socket]

    def has_active_connection(self) -> bool:
        """Whether a master connection is open (checked locally, without contacting the cluster)."""
        if not Path(self.ssh_master_socket).exists():
            return False
        try:
            result = subprocess.run(
                self._ssh("-O", "check") + [self.destination], capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def connect(self, interactive: bool = True, persist: str | None = None) -> None:
        """Open a master connection unless one is already open.

        Interactive: the terminal is passed through so password and two-factor prompts can be
        answered. Otherwise the login must succeed without prompting (e.g. SSH keys), and
        NotConnectedError tells the user to run `sllm connect`.
        """
        if self.has_active_connection():
            return
        socket = Path(self.ssh_master_socket)
        socket.unlink(missing_ok=True)  # a socket left behind by a master that has exited
        socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        cmd = self._ssh("-M", "-o", f"ControlPersist={persist or DEFAULT_PERSIST}", "-f", "-N")
        name = self.config.name
        try:
            if interactive:
                print(
                    f"Connecting to {name} ({self.destination}); answer any password or two-factor prompts...",
                    file=sys.stderr,
                )
                result = subprocess.run(cmd + [self.destination], timeout=300)
                detail = f"ssh exit code {result.returncode}"
            else:
                # stderr goes to a file, not a pipe: a pipe could stay open in the backgrounded master
                with tempfile.TemporaryFile("w+") as err:
                    result = subprocess.run(
                        cmd[:1] + ["-o", "BatchMode=yes"] + cmd[1:] + [self.destination],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, timeout=60,
                    )
                    err.seek(0)
                    lines = err.read().strip().splitlines()
                detail = lines[-1] if lines else f"ssh exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            result, detail = None, "timed out"
        except OSError as e:
            raise NotConnectedError(f"Could not run ssh: {e}") from None
        if result is None or result.returncode != 0 or not self.has_active_connection():
            hint = (
                f"Check that `ssh {self.destination}` works."
                if interactive
                else f"Run `sllm connect {name}` to log in (it can answer password or two-factor prompts)."
            )
            raise NotConnectedError(f"Could not connect to {name} ({self.destination}): {detail}. {hint}")

    def execute_command(
        self,
        command: str,
        timeout: int = 30,
        use_login_shell: bool = True,
        input: str | None = None,
    ) -> tuple[int, str, str]:
        """Run a command on the login node. Returns (exit code, stdout, stderr); never prompts.

        `input` is sent on stdin (use it for secrets, which must not appear in command lines).
        With `use_login_shell` (the default), the command runs under `bash -l` so site profiles are
        loaded: on some clusters only they put Slurm on the PATH and set SLURM_CONF and $SCRATCH.
        """
        if use_login_shell:
            command = f"bash -l -c {shlex.quote(f'echo {OUTPUT_MARKER}; {command}')}"
        cmd = self._ssh("-o", "BatchMode=yes", "-o", "ControlMaster=no") + [self.destination, command]
        try:
            result = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except OSError as e:
            return -1, "", str(e)
        stdout, stderr = result.stdout, result.stderr
        if use_login_shell:
            _, found, rest = stdout.partition(OUTPUT_MARKER + "\n")
            stdout = rest if found else stdout
        if result.returncode == 255 and not self.has_active_connection():
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else "ssh failed"
            stderr = f"not connected to {self.config.name} ({detail}); run `sllm connect {self.config.name}`"
        return result.returncode, stdout, stderr

    def execute_command_streaming(self, command: str, timeout: int = 300) -> int:
        """Run a command with its output shown live in the terminal. Returns the exit code.

        LogLevel=ERROR hides ssh's "Shared connection to ... closed." line (printed at the INFO
        level after every session with a terminal) but still shows connection errors.
        """
        cmd = self._ssh("-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "LogLevel=ERROR", "-tt")
        cmd += [self.destination, command]
        try:
            return subprocess.run(cmd, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"\nCommand timed out after {timeout}s", file=sys.stderr)
            return -1

    def resolve_dir(self, path: str) -> str:
        """Create a remote directory if needed and return its absolute path."""
        code, out, err = self.execute_command(f"mkdir -p {shell_path(path)} && cd {shell_path(path)} && pwd")
        lines = out.strip().splitlines()
        if code != 0 or not lines:
            raise RemoteLLMError(f"Could not create {path} on {self.config.name}: {err.strip() or f'exit code {code}'}")
        return lines[-1]

    def submit_slurm_job(self, script_content: str, remote_script_path: str) -> str:
        """Upload a batch script and submit it with sbatch. Returns the Slurm job id."""
        code, _, err = self.execute_command(f"cat > {shell_path(remote_script_path)}", input=script_content)
        if code != 0:
            raise RemoteLLMError(f"Could not upload the job script to {self.config.name}: {err.strip()}")
        code, out, err = self.execute_command(f"sbatch {shell_path(remote_script_path)}")
        for line in out.splitlines():
            if "Submitted batch job" in line:
                return line.split()[-1]
        raise RemoteLLMError(f"sbatch failed on {self.config.name}: {err.strip() or out.strip() or f'exit code {code}'}")

    def get_job_status(self, job_id: str) -> tuple[str, str | None, str | None]:
        """(state, node, time left) of a Slurm job.

        State is Slurm's (e.g. PENDING, RUNNING, COMPLETED, OUT_OF_MEMORY), NOT_FOUND when Slurm no
        longer knows the job, or UNKNOWN when the cluster could not be asked (including when squeue
        or sacct fail, e.g. because Slurm is not on the PATH: a failed check is not a finished job).
        """
        job = shlex.quote(str(job_id))
        code, out, err = self.execute_command(f"squeue -j {job} -h -o '%T|%N|%L'")
        if code == 0:
            self._status_warning = None
        if code == 0 and out.strip():
            parts = out.strip().splitlines()[0].split("|")
            node = parts[1] if len(parts) > 1 and parts[1] else None
            time_left = parts[2] if len(parts) > 2 and parts[2] else None
            return parts[0], node, time_left
        # squeue exits 1 with "Invalid job id" once the job has left the queue
        if code != 0 and "invalid job id" not in err.lower():
            return self._unknown_status(f"squeue failed: {err.strip() or f'exit code {code}'}")
        code, out, err = self.execute_command(f"sacct -j {job} -X -n -P -o State")
        if code == 0 and out.strip():
            return out.split()[0], None, None
        if code != 0:
            return self._unknown_status(f"sacct failed: {err.strip() or f'exit code {code}'}")
        return "NOT_FOUND", None, None

    def _unknown_status(self, reason: str) -> tuple[str, None, None]:
        """UNKNOWN, warning once per distinct reason (callers poll, and the reason rarely changes)."""
        message = f"Could not check job status on {self.config.name} ({reason}); will keep trying"
        if message != getattr(self, "_status_warning", None):
            logger.warning(message)
            self._status_warning = message
        return "UNKNOWN", None, None

    def cancel_job(self, job_id: str) -> bool:
        code, _, _ = self.execute_command(f"scancel {shlex.quote(str(job_id))}")
        return code == 0

    def get_remote_slurm_log_path(self, job_id: str) -> str:
        return f"{self.config.remote_root or '~'}/slurm-{job_id}.out"

    def get_remote_slurm_error_path(self, job_id: str) -> str:
        return f"{self.config.remote_root or '~'}/slurm-{job_id}.err"
