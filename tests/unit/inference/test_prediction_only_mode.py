"""Tests for prediction-only default inference mode behavior."""

# ruff: noqa: D103

from __future__ import annotations

import torch

from vision_studio.inference.simple import SimpleInference


class _Model(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.9, 0.1]], dtype=torch.float32).repeat(x.shape[0], 1)


def test_prediction_only_default_has_no_metrics() -> None:
    inf = SimpleInference()
    model = _Model()
    loader = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]
    out = inf.predict(model, loader)
    assert out["metrics"] == {}
