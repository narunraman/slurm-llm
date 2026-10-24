"""Registers slurm-llm with Inspect (https://inspect.aisi.org.uk): `--model sllm/<model>`."""

from inspect_ai.model import modelapi


@modelapi(name="sllm")
def sllm():
    from .inspect_api import SlurmLLMAPI

    return SlurmLLMAPI
