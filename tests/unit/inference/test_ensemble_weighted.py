"""Tests for weighted ensemble mode validation and aggregation."""

# ruff: noqa: D103

from __future__ import annotations

import pytest
import torch

from vision_studio.inference.simple import EnsembleConfig, SimpleInference
from vision_studio.types import ConfigurationError


class _Model(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        return self._logits[: x.shape[0]].clone()


def _batch(batch_size: int = 2):
    return torch.randn(batch_size, 3, 8, 8), {
        "label": torch.zeros(batch_size, dtype=torch.long)
    }


def test_weighted_averaging_applies_weights() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.8, 0.2]], dtype=torch.float32)),
        _Model(torch.tensor([[0.2, 0.8]], dtype=torch.float32)),
    ]
    cfg = EnsembleConfig(mode="weighted", weights=[0.75, 0.25])
    out = inf.run_ensemble(models, _batch(batch_size=1), cfg)
    assert torch.allclose(
        out["aggregated"], torch.tensor([[0.65, 0.35]], dtype=torch.float32)
    )
    assert torch.equal(out["preds"], torch.tensor([0]))


def test_weighted_averaging_rejects_missing_weights() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.8, 0.2]], dtype=torch.float32)),
        _Model(torch.tensor([[0.2, 0.8]], dtype=torch.float32)),
    ]
    with pytest.raises(ConfigurationError):
        inf.run_ensemble(
            models, _batch(batch_size=1), EnsembleConfig(mode="weighted", weights=None)
        )


def test_weighted_averaging_rejects_length_mismatch() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.8, 0.2]], dtype=torch.float32)),
        _Model(torch.tensor([[0.2, 0.8]], dtype=torch.float32)),
    ]
    with pytest.raises(ConfigurationError):
        inf.run_ensemble(
            models, _batch(batch_size=1), EnsembleConfig(mode="weighted", weights=[1.0])
        )
