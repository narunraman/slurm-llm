"""Tests for cross-process model preparation serialization."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from slurm_llm.orchestration import JobOrchestrator


@pytest.mark.unit
def test_ensure_model_ready_locks_normalized_model_then_rechecks():
    orchestrator = JobOrchestrator.__new__(JobOrchestrator)
    orchestrator.config_loader = Mock()
    orchestrator.config_loader.require_model.return_value = SimpleNamespace(
        id="models--org--Model-7B"
    )
    expected = object()
    orchestrator._ensure_model_ready_unlocked = Mock(return_value=expected)
    events = []

    @contextmanager
    def recording_lock(model_name, cluster_name, instance):
        events.append(("acquire", model_name, cluster_name, instance))
        yield
        events.append(("release", model_name, cluster_name, instance))

    with patch("slurm_llm.orchestration._model_readiness_lock", recording_lock):
        result = orchestrator.ensure_model_ready(
            "model-7b",
            cluster_map={"model-7b": "cluster-b"},
            instance=2,
        )

    assert result is expected
    assert events == [
        ("acquire", "models--org--Model-7B", "cluster-b", 2),
        ("release", "models--org--Model-7B", "cluster-b", 2),
    ]
    orchestrator._ensure_model_ready_unlocked.assert_called_once_with(
        "model-7b",
        cluster_name=None,
        cluster_map={"model-7b": "cluster-b"},
        time_override=None,
        exclude_clusters=None,
        instance=2,
    )
