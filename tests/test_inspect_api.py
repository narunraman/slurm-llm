"""`--model sllm/<model>` in Inspect: ready the model through slurm-llm, then use Inspect's vLLM client."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("httpx2")

from inspect_ai.model import get_model  # noqa: E402

from slurm_llm import inspect_api  # noqa: E402


@pytest.fixture(autouse=True)
def no_failures_from_other_tests():
    inspect_api._failed.clear()


@pytest.fixture
def sllm(monkeypatch):
    """A SlurmLLM that is "ready" at once, serving Qwen3-8B on localhost:43210."""
    client = MagicMock(base_url="http://localhost:43210/v1", served_model="Qwen3-8B")
    cls = MagicMock(return_value=client)
    monkeypatch.setattr("slurm_llm.client.SlurmLLM", cls)
    monkeypatch.setattr(inspect_api.VLLMAPI, "_resolve_server", lambda self: None)  # no HTTP client here
    return cls, client


def test_nothing_happens_on_the_cluster_until_the_first_request(sllm):
    cls, _ = sllm
    api = get_model("sllm/qwen-3-8b:1", cluster="cluster-a", time="2:00:00").api
    assert isinstance(api, inspect_api.SlurmLLMAPI)
    assert api.base_model == "qwen-3-8b"  # ":1" is slurm-llm's instance, not a LoRA adapter
    cls.assert_not_called()


def test_the_first_request_readies_the_model_and_uses_its_served_name(sllm):
    cls, client = sllm
    api = get_model("sllm/qwen-3-8b:1", cluster="cluster-a", time="2:00:00").api
    api._resolve_server()
    cls.assert_called_once_with(model="qwen-3-8b:1", cluster="cluster-a", walltime="2:00:00")
    client.ensure_ready.assert_called_once()
    assert api._init_base_url == "http://localhost:43210/v1"
    assert api.base_model == "Qwen3-8B"  # what vLLM serves it as
    api._resolve_server()
    cls.assert_called_once()  # ready once


@pytest.mark.parametrize("keep, cancelled", [(None, False), ("false", True), (False, True), ("true", False)])
def test_closing_keeps_or_cancels_the_job(sllm, keep, cancelled):
    _, client = sllm
    args = {} if keep is None else {"keep": keep}
    api = get_model("sllm/qwen-3-8b", **args).api
    api._resolve_server()
    api.close()
    assert client.cancel_job.called is cancelled
    assert client.close.called is not cancelled  # the port forward goes either way
    assert api._init_base_url is None  # a later request readies it again


def test_the_job_is_released_when_the_process_ends(sllm, monkeypatch):
    _, client = sllm
    hooks = []
    monkeypatch.setattr(inspect_api.atexit, "register", hooks.append)
    api = get_model("sllm/qwen-3-8b", keep="false").api
    api._resolve_server()
    assert hooks == [api._release]
    hooks[0]()  # the process ends without Inspect closing the model
    client.cancel_job.assert_called_once()
    api.close()  # and a later close does nothing more
    client.cancel_job.assert_called_once()


def test_a_failed_setup_is_not_tried_again(sllm):
    # the job crashed: requests waiting for the same model must not each submit another one
    cls, client = sllm
    client.ensure_ready.side_effect = RuntimeError("Job 7 ended (status: FAILED)")
    first = get_model("sllm/qwen-3-8b", cluster="cluster-a").api
    with pytest.raises(RuntimeError, match="FAILED"):
        first._resolve_server()
    for _ in range(3):
        waiting = get_model("sllm/qwen-3-8b", cluster="cluster-a").api
        with pytest.raises(RuntimeError, match="could not be started earlier in this run: Job 7 ended"):
            waiting._resolve_server()
    cls.assert_called_once()  # one attempt, one job
