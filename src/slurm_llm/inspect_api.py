"""Inspect model provider for models that slurm-llm serves on Slurm: `--model sllm/<model>`.

On the first request, the model is made ready as `sllm prepare` would (a running or queued job is
reused; otherwise a cluster and GPUs are chosen and a job is submitted), a port is forwarded to
it, and requests then go to its vLLM server through Inspect's vLLM provider. `<model>` is a name
from slurm-llm's model registry (`sllm models`), optionally with an instance: `sllm/qwen-3-8b:1`.

Model args (`-M`): `cluster` (run there), `time` (time limit of a new job, e.g. 4:00:00), and
`keep` (default true: leave the job running when the evaluation ends, so the next one reuses it;
false: cancel it).
"""

from __future__ import annotations

import atexit
from typing import Any

from inspect_ai.model import GenerateConfig
from inspect_ai.model._providers.vllm import VLLMAPI


def _flag(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() not in ("false", "0", "no", "off")


class SlurmLLMAPI(VLLMAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        cluster: str | None = None,
        time: str | None = None,
        keep: bool | str = True,
        **model_args: Any,
    ) -> None:
        # "model:1" is slurm-llm's instance; the vLLM provider would read it as a LoRA adapter
        self._sllm_args = {"model": model_name, "cluster": cluster, "walltime": time}
        self._keep = _flag(keep)
        self._sllm = None
        name, _, instance = model_name.rpartition(":")
        super().__init__(
            model_name=name if instance.isdigit() else model_name,
            base_url=base_url,
            api_key=api_key or "unused",  # slurm-llm's vLLM servers take any key
            config=config,
            **model_args,
        )

    def _resolve_server(self) -> None:
        """Make the model ready on its cluster (on the first request), then connect to it."""
        if self._init_base_url is None:
            from .client import SlurmLLM

            client = SlurmLLM(**self._sllm_args)
            client.ensure_ready()
            self._sllm = client
            # Inspect does not always close models: release the job when the process ends too
            atexit.register(self._release)
            self._init_base_url = client.base_url
            self.base_model = client.served_model  # the name vLLM serves the model under
        super()._resolve_server()

    def close(self) -> None:
        super().close()
        self._release()

    def _release(self) -> None:
        """Close the port forward and, with keep=false, cancel the job."""
        if self._sllm is not None:
            if self._keep:
                self._sllm.close()  # the job keeps serving, for the next evaluation
            else:
                self._sllm.cancel_job()  # also closes the port forward
            self._sllm = None
            self._init_base_url = None
