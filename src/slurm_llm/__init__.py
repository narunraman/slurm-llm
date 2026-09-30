"""slurm-llm: serve open LLMs with vLLM on Slurm clusters and use them like a local API.

Public names are imported on first use, so the CLI starts quickly.
"""

from importlib import import_module

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("slurm-llm")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

_EXPORTS = {
    "RemoteLLM": "client",
    "ConfigLoader": "config",
    "ClusterConfig": "config",
    "ModelConfig": "config",
    "ResourceConfig": "config",
    "GPUType": "config",
    "RemoteLLMError": "config",
    "JobManager": "job",
    "JobInfo": "job",
    "SSHClient": "ssh",
    "SSHConnectionManager": "ssh_manager",
    "ClusterScheduler": "scheduler",
    "JobOrchestrator": "orchestration",
    "OrchestratorConfig": "orchestration",
    "detect_vllm_port": "health",
    "wait_for_vllm_health": "health",
    "wait_for_job_running": "health",
    "submit_model": "api",
    "SubmitResult": "api",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
