"""Local port forwards through a cluster's shared SSH connection."""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import state_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TunnelHandle:
    """Description of a local forward owned by an SSH control master."""

    local_port: int
    forward_spec: str
    ssh_target: str
    control_socket: str
    ssh_options: tuple[str, ...]
    owner_pid: int

    def command(self, operation: str) -> list[str]:
        """Build an OpenSSH multiplexing control command."""
        cmd = ["ssh", *self.ssh_options, "-S", self.control_socket]
        if operation == "forward":
            cmd.extend(["-o", "ExitOnForwardFailure=yes"])
        cmd.extend(["-O", operation, "-L", self.forward_spec, self.ssh_target])
        return cmd


class TunnelRegistry:
    """Track multiplexed forwards and clean up only their owning process's entries."""

    _instance: "TunnelRegistry | None" = None

    def __init__(self, registry_path: Path | None = None):
        self._registry_path = registry_path or state_dir() / "tunnels.json"
        # Tunnel PIDs and listening ports are host-local. Keep the advisory lock
        # on a local filesystem as well: shared (NFS) home directories often
        # return ENOLCK for flock().
        self._lock_path = (
            Path(tempfile.gettempdir())
            / f"slurm-llm-tunnels-{os.getuid()}-{socket.gethostname()}.lock"
        )
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._atexit_registered = False
        self._thread_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "TunnelRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_unlocked(self) -> dict[str, Any]:
        """Load registry from disk."""
        if not self._registry_path.exists():
            return {"tunnels": {}}
        try:
            with open(self._registry_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"tunnels": {}}

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        """Save registry to disk."""
        try:
            with open(self._registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.warning(f"Could not save tunnel registry: {e}")

    @contextmanager
    def _locked_data(self):
        """Serialize registry read-modify-write cycles across client processes."""
        with self._thread_lock:
            with open(self._lock_path, "a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                data = self._load_unlocked()
                try:
                    yield data
                finally:
                    self._save_unlocked(data)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def register_tunnel(
        self, handle: TunnelHandle, model: str, cluster: str
    ) -> None:
        """Register an active tunnel."""
        with self._locked_data() as data:
            data["tunnels"][str(handle.local_port)] = {
                **asdict(handle),
                "ssh_options": list(handle.ssh_options),
                "model": model,
                "cluster": cluster,
                "created": time.time(),
            }

        # Register atexit handler on first tunnel registration
        if not self._atexit_registered:
            atexit.register(self.cleanup_owned, os.getpid())
            self._atexit_registered = True

    def unregister_tunnel(self, local_port: int, owner_pid: int) -> None:
        """Remove a tunnel only if it still belongs to the requested owner."""
        with self._locked_data() as data:
            info = data["tunnels"].get(str(local_port))
            if info and info.get("owner_pid") == owner_pid:
                del data["tunnels"][str(local_port)]

    def cleanup_stale_tunnels(self) -> int:
        """
        Clean up orphaned tunnels from previous sessions.

        Returns:
            Number of stale tunnels cleaned up
        """
        stale: list[dict[str, Any]] = []
        with self._locked_data() as data:
            for port_str, info in list(data["tunnels"].items()):
                owner_pid = info.get("owner_pid", info.get("pid"))
                if not owner_pid or not self._is_process_alive(owner_pid):
                    stale.append(info)
                    del data["tunnels"][port_str]

        for info in stale:
            self._cancel_registered_forward(info)
        return len(stale)

    def cleanup_owned(self, owner_pid: int | None = None) -> None:
        """Cancel forwards registered by one process without touching siblings."""
        owner_pid = owner_pid or os.getpid()
        owned: list[dict[str, Any]] = []
        with self._locked_data() as data:
            for port_str, info in list(data["tunnels"].items()):
                if info.get("owner_pid") == owner_pid:
                    owned.append(info)
                    del data["tunnels"][port_str]

        for info in owned:
            self._cancel_registered_forward(info)

    def cleanup_all(self) -> None:
        """Backward-compatible alias that now cleans only the current owner."""
        self.cleanup_owned(os.getpid())

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _cancel_registered_forward(self, info: dict[str, Any]) -> None:
        """Best-effort cancellation for a registry entry using its control socket."""
        required = {"local_port", "forward_spec", "ssh_target", "control_socket"}
        if not required.issubset(info):
            # Legacy PID-only records cannot safely identify a multiplexed forward.
            return

        handle = TunnelHandle(
            local_port=int(info["local_port"]),
            forward_spec=info["forward_spec"],
            ssh_target=info["ssh_target"],
            control_socket=info["control_socket"],
            ssh_options=tuple(info.get("ssh_options", ())),
            owner_pid=int(info["owner_pid"]),
        )
        try:
            result = subprocess.run(
                handle.command("cancel"),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug(
                    "Could not cancel stale tunnel on port %s: %s",
                    handle.local_port,
                    result.stderr.strip(),
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug(
                "Could not cancel stale tunnel on port %s: %s",
                handle.local_port,
                exc,
            )
