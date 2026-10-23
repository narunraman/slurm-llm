"""SSH access to clusters through OpenSSH connection sharing.

`sllm connect` (or `rclust connect`) logs in once, answering any password or two-factor
prompt, and leaves a background master connection (ssh ControlMaster). Every other command runs
over that connection without prompting. The socket path is the one rclust uses for the same host,
so a connection opened by either tool is reused by the other.
"""

from __future__ import annotations

import logging
import re
import shlex
import time

from .config import ClusterConfig, ConfigError, SlurmLLMError

logger = logging.getLogger(__name__)

DEFAULT_PERSIST = "1h"  # how long an idle master connection stays open (ssh ControlPersist)
SLURM_HINT = "Check that `bash -lc {command}` works on {name}."


# Slurm's own messages when it is missing or not configured (failures that retrying cannot fix)
SLURM_MISSING = ("command not found", "configuration source")
SLURM_COMMANDS = ("sbatch", "squeue", "sacct", "scancel")
# ...and when the controller is busy or unreachable (sbatch may or may not have submitted the job)
SLURM_BUSY = ("timed out", "unable to contact", "connection", "temporarily unavailable", "try again")


class NotConnectedError(SlurmLLMError):
    """No usable SSH connection to a cluster."""

    def __init__(self, message: str, cluster: str | None = None):
        super().__init__(message)
        self.cluster = cluster


class SlurmUnavailableError(SlurmLLMError):
    """Slurm's commands are missing or not configured on a cluster's login node."""


class StatusUnknownError(SlurmLLMError):
    """A job's status could not be checked for too long (the job itself may be fine)."""


def shell_path(path: str) -> str:
    """Quote a remote path for the shell, keeping `~` and `$VARS` (e.g. $SCRATCH) working."""
    if path == "~" or path.startswith("~/"):
        path = "$HOME" + path[1:]
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


class SSHClient:
    """Runs commands on one cluster over a shared SSH connection.

    The connection itself (logging in once, keeping it alive, noticing when it is gone) is
    rclust's: slurm-llm and rclust share one transport and one connection per cluster.
    """

    def __init__(self, cluster_config: ClusterConfig):
        from rclust.ssh import SSHClient as Transport

        self.config = cluster_config
        try:
            self.transport = Transport(cluster_config.host, cluster_config.user, name=cluster_config.name)
        except ValueError as e:
            raise ConfigError(f"cluster '{cluster_config.name}': {e}") from None
        self.transport.connect_command = "sllm connect"  # what its messages tell you to run
        self.destination = self.transport.destination
        self.ssh_master_socket = str(self.transport.socket_path)
        self._status_warning: str | None = None  # last "could not check" warning, so polls don't repeat it
        self.last_status_error: str | None = None  # why the last status check gave UNKNOWN

    def _get_ssh_target(self) -> str:
        """The ssh destination ([user@]host or an alias from ~/.ssh/config)."""
        return self.destination

    def has_active_connection(self) -> bool:
        """Whether a master connection is open (checked locally, without contacting the cluster)."""
        return self.transport.is_connected()

    def connect(self, interactive: bool = True, persist: str | None = None) -> None:
        """Open a master connection unless one is already open.

        Interactive: the terminal is passed through so password and two-factor prompts can be
        answered. Otherwise the login must succeed without prompting (e.g. SSH keys), and
        NotConnectedError tells the user to run `sllm connect`.
        """
        from rclust.ssh import NotConnectedError as TransportNotConnected

        try:
            self.transport.connect(persist=persist or DEFAULT_PERSIST, interactive=interactive)
        except TransportNotConnected as e:
            raise NotConnectedError(str(e), self.config.name) from None
        except OSError as e:
            raise NotConnectedError(f"Could not run ssh: {e}", self.config.name) from None

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
        A lost connection, including one that stopped responding, gives exit code 255.
        """
        return self.transport.execute_command(command, timeout=timeout, use_login_shell=use_login_shell, input=input)

    def execute_command_streaming(self, command: str, timeout: int = 300) -> int:
        """Run a command with its output shown live in the terminal. Returns the exit code."""
        return self.transport.execute_streaming(command, timeout=timeout)

    def resolve_dir(self, path: str) -> str:
        """Create a remote directory if needed and return its absolute path."""
        code, out, err = self.execute_command(f"mkdir -p {shell_path(path)} && cd {shell_path(path)} && pwd")
        lines = out.strip().splitlines()
        if code != 0 or not lines:
            raise SlurmLLMError(f"Could not create {path} on {self.config.name}: {err.strip() or f'exit code {code}'}")
        return lines[-1]

    def lost_connection(self, doing: str) -> NotConnectedError:
        """The error for a command that failed because the connection is gone."""
        name = self.config.name
        return NotConnectedError(
            f"Lost the connection to {name} while {doing}. Run `sllm connect {name}`, then run the command again.",
            name,
        )

    def check_connection(self, code: int, doing: str) -> None:
        """Raise NotConnectedError if a command failed (`code`) because the connection is gone."""
        if self._connection_lost(code):
            raise self.lost_connection(doing)

    def _connection_lost(self, code: int) -> bool:
        return code == 255 and not self.has_active_connection()

    def slurm_problem(self) -> str | None:
        """Why Slurm's commands cannot be used on the login node, or None if they can."""
        commands = " ".join(SLURM_COMMANDS)
        code, out, err = self.execute_command(f"for c in {commands}; do command -v $c >/dev/null || echo $c; done")
        if self._connection_lost(code):
            raise self.lost_connection("checking Slurm")
        if code != 0:
            return f"could not check ({err.strip() or f'exit code {code}'})"
        missing = out.split()
        if missing:
            hint = SLURM_HINT.format(command=missing[0], name=self.config.name)
            return f"{', '.join(missing)} not found, even in a login shell. {hint}"
        return None

    def _job_ids_named(self, job_name: str) -> set[str] | None:
        """Your queued jobs with this name, or None if the queue could not be read."""
        code, out, _ = self.execute_command(f"squeue -u $USER -h -n {shlex.quote(job_name)} -o %i")
        return set(out.split()) if code == 0 else None

    def submit_slurm_job(self, script_content: str, remote_script_path: str) -> str:
        """Upload a batch script and submit it with sbatch. Returns the Slurm job id.

        When sbatch's answer is lost (a timeout or dropped connection), the job may still have
        been submitted, so the queue is checked before reporting a failure: retrying blindly
        would start a second job.
        """
        name = self.config.name
        code, _, err = self.execute_command(f"cat > {shell_path(remote_script_path)}", input=script_content)
        if self._connection_lost(code):
            raise self.lost_connection("uploading the job script (no job was submitted)")
        if code != 0:
            raise SlurmLLMError(f"Could not upload the job script to {name}: {err.strip()}")
        job_name = re.search(r"^#SBATCH\s+(?:--job-name[= ]|-J\s*)(\S+)", script_content, re.MULTILINE)
        before = self._job_ids_named(job_name.group(1)) if job_name else None

        code, out, err = self.execute_command(f"sbatch {shell_path(remote_script_path)}", timeout=60)
        for line in out.splitlines():
            if "Submitted batch job" in line:
                return line.split()[-1]
        reason = err.strip() or out.strip() or f"exit code {code}"
        if any(s in reason.lower() for s in SLURM_MISSING):
            raise SlurmUnavailableError(f"sbatch failed on {name}: {reason}. {SLURM_HINT.format(command='sbatch', name=name)}")
        busy = any(s in reason.lower() for s in SLURM_BUSY)
        # Slurm refused the job. sbatch exits 255 for some refusals (e.g. an invalid #SBATCH line),
        # like ssh when the connection drops, so its own "sbatch: error:" decides.
        if not busy and ("sbatch: error" in reason.lower() or code not in (0, -1, 255)):
            raise SlurmLLMError(f"sbatch failed on {name}: {reason}")

        # The answer was lost: look for the job in the queue
        if self._connection_lost(code):
            raise NotConnectedError(
                f"Lost the connection to {name} while submitting; the job may have been submitted. "
                f"Run `sllm connect {name}`, then run the command again: it will find the job if it was.",
                name,
            )
        if before is not None:
            for _ in range(3):
                time.sleep(5)
                after = self._job_ids_named(job_name.group(1))
                if after is None:
                    break
                new = after - before
                if len(new) == 1:
                    job_id = new.pop()
                    logger.warning(f"sbatch on {name} did not answer ({reason}), but job {job_id} is in the queue; using it")
                    return job_id
        raise SlurmLLMError(
            f"sbatch on {name} did not answer ({reason}) and the job is not in the queue. "
            f"Check `squeue -u $USER` on {name} before running the command again, to avoid a second job."
        )

    def get_job_status(self, job_id: str) -> tuple[str, str | None, str | None]:
        """(state, node, time left) of a Slurm job.

        State is Slurm's (e.g. PENDING, RUNNING, COMPLETED, OUT_OF_MEMORY), NOT_FOUND when Slurm no
        longer knows the job, or UNKNOWN when the cluster could not be asked (including when squeue
        or sacct fail, e.g. because Slurm is not on the PATH: a failed check is not a finished job).
        """
        job = shlex.quote(str(job_id))
        code, out, err = self.execute_command(f"squeue -j {job} -h -o '%T|%N|%L'")
        self._check_status_failure(job_id, "squeue", code, err)
        if code == 0:
            self._status_warning = self.last_status_error = None
        if code == 0 and out.strip():
            parts = out.strip().splitlines()[0].split("|")
            node = parts[1] if len(parts) > 1 and parts[1] else None
            time_left = parts[2] if len(parts) > 2 and parts[2] else None
            return parts[0], node, time_left
        # squeue exits 1 with "Invalid job id" once the job has left the queue
        if code != 0 and "invalid job id" not in err.lower():
            return self._unknown_status(f"squeue failed: {err.strip() or f'exit code {code}'}")
        code, out, err = self.execute_command(f"sacct -j {job} -X -n -P -o State")
        self._check_status_failure(job_id, "sacct", code, err)
        if code == 0 and out.strip():
            return out.split()[0], None, None
        if code != 0:
            return self._unknown_status(f"sacct failed: {err.strip() or f'exit code {code}'}")
        return "NOT_FOUND", None, None

    def _check_status_failure(self, job_id: str, command: str, code: int, err: str) -> None:
        """Raise for failures that retrying cannot fix: a lost connection, or Slurm missing."""
        if self._connection_lost(code):
            raise self.lost_connection(f"checking job {job_id} (the job has not been cancelled)")
        if code != 0 and any(s in err.lower() for s in SLURM_MISSING):
            name = self.config.name
            raise SlurmUnavailableError(
                f"Could not check job {job_id} on {name}: {err.strip()}. The job has not been cancelled. "
                + SLURM_HINT.format(command=command, name=name)
            )

    def _unknown_status(self, reason: str) -> tuple[str, None, None]:
        """UNKNOWN, warning once per distinct reason (callers poll, and the reason rarely changes)."""
        self.last_status_error = reason
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
