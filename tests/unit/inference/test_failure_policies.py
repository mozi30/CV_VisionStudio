"""Tests for ensemble failure-policy behavior."""

# ruff: noqa: D103

from __future__ import annotations

import pytest
import torch

from vision_studio.inference.simple import EnsembleConfig, SimpleInference
from vision_studio.types import ConfigurationError


class _OkModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        return self._logits[: x.shape[0]].clone()


class _FailModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("boom")


def _batch(batch_size: int = 1):
    return torch.randn(batch_size, 3, 8, 8), {
        "label": torch.zeros(batch_size, dtype=torch.long)
    }


def test_fail_fast_raises_on_first_failure() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [_FailModel(), _OkModel(torch.tensor([[0.9, 0.1]]))]
    with pytest.raises(RuntimeError):
        inf.run_ensemble(models, _batch(), EnsembleConfig(failure_policy="fail-fast"))


def test_continue_with_warning_requires_at_least_one_success() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [_FailModel(), _FailModel()]
    with pytest.raises(ConfigurationError):
        inf.run_ensemble(
            models, _batch(), EnsembleConfig(failure_policy="continue-with-warning")
        )


def test_continue_with_warning_sets_status_and_failed_models() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [_OkModel(torch.tensor([[0.9, 0.1]])), _FailModel()]
    out = inf.run_ensemble(
        models, _batch(), EnsembleConfig(failure_policy="continue-with-warning")
    )
    assert out["status"] == "completed_with_warnings"
    assert out["failed_models"] == [1]
