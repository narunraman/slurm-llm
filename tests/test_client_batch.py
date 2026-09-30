"""Regression tests for load-balanced batch timeout behavior."""

import concurrent.futures
import time
from unittest.mock import patch

import pytest

from slurm_llm.client import RemoteLLM


def _client(endpoint_count: int, generate) -> RemoteLLM:
    client = RemoteLLM.__new__(RemoteLLM)
    client._openai_client = object()
    client._tunnel_handle = None
    client._tunnel_registry = None
    client.local_port = 0
    client._endpoints = [
        (f"node-{index}", object()) for index in range(endpoint_count)
    ]
    client.generate = generate
    return client


def _requests(count: int, timeout: float = 900) -> list[dict]:
    return [
        {
            "messages": [{"role": "user", "content": f"request {index}"}],
            "timeout": timeout,
        }
        for index in range(count)
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("endpoint_count", "expected_timeout"),
    [(1, 1800.0), (2, 900.0), (10, 900.0)],
)
def test_default_batch_timeout_only_scales_for_sequential_worker_waves(
    endpoint_count, expected_timeout
):
    client = _client(
        endpoint_count,
        lambda **kwargs: object(),
    )
    observed_timeouts = []
    real_as_completed = concurrent.futures.as_completed

    def recording_as_completed(futures, timeout):
        observed_timeouts.append(timeout)
        return real_as_completed(futures, timeout=timeout)

    with patch(
        "concurrent.futures.as_completed",
        side_effect=recording_as_completed,
    ):
        results = client.generate_batch(_requests(20))

    assert len(results) == 20
    assert observed_timeouts == [expected_timeout]


@pytest.mark.unit
def test_batch_timeout_returns_promptly_and_notifies_each_request_once():
    def slow_generate(**kwargs):
        time.sleep(0.1)
        return object()

    client = _client(1, slow_generate)
    callbacks = []

    started = time.monotonic()
    results = client.generate_batch(
        _requests(1, timeout=1),
        max_concurrent_per_endpoint=1,
        batch_timeout=0.01,
        per_request_callback=lambda idx, result: callbacks.append((idx, result)),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.08
    assert len(results) == 1
    assert isinstance(results[0], TimeoutError)
    assert len(callbacks) == 1
    assert callbacks[0][0] == 0
    assert isinstance(callbacks[0][1], TimeoutError)

    # The synchronous worker finishes later, but its result must not overwrite
    # the timeout or fire a second callback.
    time.sleep(0.15)
    assert len(callbacks) == 1
    assert isinstance(results[0], TimeoutError)


@pytest.mark.unit
def test_invalid_endpoint_concurrency_is_rejected():
    client = _client(1, lambda **kwargs: object())

    with pytest.raises(
        ValueError,
        match="max_concurrent_per_endpoint must be at least 1",
    ):
        client.generate_batch(
            _requests(1),
            max_concurrent_per_endpoint=0,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requests", "detail"),
    [
        ([{"role": "user", "content": "Q1"}], "looks like one message"),
        ([[{"role": "user", "content": "Q1"}]], "a list of messages"),
        (["Q1"], "a string"),
        ([{"prompt": "Q1"}], 'without a "messages" list'),
        ({"messages": []}, "got dict"),
        ("Q1", "got str"),
    ],
)
def test_generate_batch_explains_the_expected_format(requests, detail):
    client = _client(1, lambda **kwargs: pytest.fail("no request may be sent"))
    with pytest.raises(TypeError, match="expects a list of requests, each a dict with a \"messages\" list") as info:
        client.generate_batch(requests)
    assert detail in str(info.value)


@pytest.mark.unit
def test_generate_batch_accepts_the_documented_format():
    client = _client(1, lambda **kwargs: kwargs["messages"][0]["content"])
    assert client.generate_batch([{"messages": [{"role": "user", "content": "Q1"}]}]) == ["Q1"]
