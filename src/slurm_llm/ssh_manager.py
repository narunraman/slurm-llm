"""Shared SSH clients for several clusters."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import ConfigLoader
from .ssh import SSHClient

logger = logging.getLogger(__name__)


class SSHConnectionManager:
    """Hands out one SSHClient per cluster, opening its master connection on first use.

    With `interactive=False` (the default, and what parallel code needs) connections are only
    opened when no prompt is required; otherwise `get_client` raises NotConnectedError telling
    the user to run `sllm connect`.
    """

    def __init__(self, config_loader: ConfigLoader, interactive: bool = False):
        self.config_loader = config_loader
        self.interactive = interactive
        self._connections: dict[str, SSHClient] = {}
        self._lock = threading.Lock()
        self._cluster_locks: dict[str, threading.Lock] = {}

    def get_client(self, cluster_name: str) -> SSHClient:
        """A connected client for the cluster (raises ConfigError or NotConnectedError)."""
        with self._lock:
            cluster_lock = self._cluster_locks.setdefault(cluster_name, threading.Lock())
        with cluster_lock:
            client = self._connections.get(cluster_name)
            if client is None:
                client = SSHClient(self.config_loader.get_cluster(cluster_name))
            if not client.has_active_connection():
                logger.info(f"Opening SSH connection to {cluster_name}")
                client.connect(interactive=self.interactive)
            self._connections[cluster_name] = client
            return client

    def reconnect(self, cluster_name: str) -> SSHClient:
        """Open a new master connection interactively (answering two-factor prompts)."""
        with self._lock:
            cluster_lock = self._cluster_locks.setdefault(cluster_name, threading.Lock())
        with cluster_lock:
            client = self._connections.get(cluster_name) or SSHClient(self.config_loader.get_cluster(cluster_name))
            client.connect(interactive=True)
            self._connections[cluster_name] = client
            return client

    def close_all(self) -> None:
        """Forget cached clients. Master connections stay open until their ControlPersist expires."""
        self._connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


def execute_parallel(
    ssh_clients: dict[str, SSHClient],
    command: str | dict[str, str],
    timeout: int = 30,
) -> dict[str, tuple[int, str, str]]:
    """Run one command (or one per cluster, given a dict) on several clusters at once."""
    if not ssh_clients:
        return {}
    commands = command if isinstance(command, dict) else {name: command for name in ssh_clients}

    def run(name: str, client: SSHClient):
        try:
            return name, client.execute_command(commands[name], timeout=timeout)
        except Exception as e:
            logger.warning(f"Command failed on {name}: {e}")
            return name, (1, "", str(e))

    results = {}
    with ThreadPoolExecutor(max_workers=len(ssh_clients)) as executor:
        futures = [executor.submit(run, name, client) for name, client in ssh_clients.items()]
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
    return results
