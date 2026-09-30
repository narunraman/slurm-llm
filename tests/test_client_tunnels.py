"""Regression tests for SSH ControlMaster tunnel lifecycle."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from slurm_llm.client import (
    RemoteLLM,
    TunnelHandle,
    TunnelRegistry,
    _model_ids_match,
)


def _client(local_port: int = 43123) -> RemoteLLM:
    client = RemoteLLM.__new__(RemoteLLM)
    client.local_port = local_port
    client.model = "test-model"
    client._job = SimpleNamespace(
        node="gpu001",
        local_port=8123,
        cluster_name="cluster-a",
    )
    client._ssh_client = SimpleNamespace(
        ssh_master_socket="/tmp/ssh-master",
        _get_ssh_target=lambda: "-p 22 user@cluster-a.example",
    )
    client._tunnel_registry = Mock()
    client._tunnel_handle = None
    return client


@pytest.mark.unit
def test_mux_forward_accepts_successful_immediate_exit():
    client = _client()

    with patch("slurm_llm.client.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        handle = client._create_tunnel()

    assert isinstance(handle, TunnelHandle)
    assert handle.forward_spec == "43123:gpu001:8123"
    assert run.call_args.args[0] == [
        "ssh",
        "-p",
        "22",
        "-S",
        "/tmp/ssh-master",
        "-o",
        "ExitOnForwardFailure=yes",
        "-O",
        "forward",
        "-L",
        "43123:gpu001:8123",
        "user@cluster-a.example",
    ]
    client._tunnel_registry.register_tunnel.assert_called_once_with(
        handle=handle,
        model="test-model",
        cluster="cluster-a",
    )


@pytest.mark.unit
def test_mux_forward_reports_ssh_stderr():
    client = _client()

    with patch("slurm_llm.client.subprocess.run") as run:
        run.return_value = SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="Control socket connect failed",
        )
        with pytest.raises(RuntimeError, match="Control socket connect failed"):
            client._create_tunnel()

    client._tunnel_registry.register_tunnel.assert_not_called()


@pytest.mark.unit
def test_close_cancels_the_exact_mux_forward():
    client = _client()
    handle = TunnelHandle(
        local_port=43123,
        forward_spec="43123:gpu001:8123",
        ssh_target="user@cluster-a.example",
        control_socket="/tmp/ssh-master",
        ssh_options=("-p", "22"),
        owner_pid=9876,
    )
    client._tunnel_handle = handle

    with patch("slurm_llm.client.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        client.close()

    assert run.call_args.args[0] == [
        "ssh",
        "-p",
        "22",
        "-S",
        "/tmp/ssh-master",
        "-O",
        "cancel",
        "-L",
        "43123:gpu001:8123",
        "user@cluster-a.example",
    ]
    client._tunnel_registry.unregister_tunnel.assert_called_once_with(43123, 9876)
    assert client._tunnel_handle is None


@pytest.mark.unit
def test_registry_cleanup_does_not_touch_sibling_owner(tmp_path):
    registry = TunnelRegistry(tmp_path / "tunnels.json")
    # Avoid registering process-exit callbacks against a temporary test directory.
    registry._atexit_registered = True
    first = TunnelHandle(
        local_port=43123,
        forward_spec="43123:gpu001:8123",
        ssh_target="cluster-a",
        control_socket="/tmp/ssh-master",
        ssh_options=(),
        owner_pid=111,
    )
    sibling = TunnelHandle(
        local_port=43124,
        forward_spec="43124:gpu001:8123",
        ssh_target="cluster-a",
        control_socket="/tmp/ssh-master",
        ssh_options=(),
        owner_pid=222,
    )
    registry.register_tunnel(first, "model-a", "cluster-a")
    registry.register_tunnel(sibling, "model-b", "cluster-a")

    with patch.object(registry, "_cancel_registered_forward") as cancel:
        registry.cleanup_owned(111)

    cancel.assert_called_once()
    remaining = registry._load_unlocked()["tunnels"]
    assert set(remaining) == {"43124"}
    assert remaining["43124"]["owner_pid"] == 222


@pytest.mark.unit
def test_registry_lock_is_host_local_even_when_registry_is_shared(tmp_path):
    registry = TunnelRegistry(tmp_path / "shared-home" / "tunnels.json")

    assert registry._lock_path.parent == Path(tempfile.gettempdir())
    assert registry._lock_path.name.startswith("slurm-llm-tunnels-")


@pytest.mark.unit
def test_model_id_match_tolerates_alias_punctuation_and_case():
    assert _model_ids_match("qwen-3-14b", "Qwen3-14B")
    assert not _model_ids_match("qwen-3-14b", "Qwen2.5-72B-Instruct")


@pytest.mark.unit
def test_remote_handshake_rejects_an_unexpected_served_model():
    client = RemoteLLM.__new__(RemoteLLM)
    client.model = "qwen-3-14b"
    client.served_model_name = "Qwen3-14B"
    client.cluster = "cluster-b"
    client.instance = None
    client.walltime = None
    client.local_port = 43123
    client.max_model_len = None
    client._job = SimpleNamespace(
        job_id="18108772",
        node="g1",
        local_port=38772,
        cluster_name="cluster-b",
        max_model_len=32768,
    )
    client.orchestrator = Mock()
    client.orchestrator.ensure_models_ready.return_value = {
        "qwen-3-14b": client._job
    }
    client.orchestrator.conn_manager.get_client.return_value = Mock()
    client._tunnel_registry = Mock()
    client._tunnel_handle = None
    client._create_tunnel = Mock(
        return_value=TunnelHandle(
            local_port=43123,
            forward_spec="43123:g1:38772",
            ssh_target="cluster-b",
            control_socket="/tmp/ssh-master",
            ssh_options=(),
            owner_pid=123,
        )
    )

    wrong_models = SimpleNamespace(
        data=[SimpleNamespace(id="Qwen2.5-72B-Instruct")]
    )
    with patch("slurm_llm.client.OpenAI") as openai_cls, patch.object(
        client, "close"
    ) as close:
        openai_cls.return_value.models.list.return_value = wrong_models
        with pytest.raises(RuntimeError, match="unexpected model"):
            client._ensure_ready_remote()

    close.assert_called_once()
