"""Integration tests for multi-model soft-voting workflows."""

# ruff: noqa: D103

from __future__ import annotations

import torch

from vision_studio.inference.simple import SimpleInference


class _Model(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        batch = x.shape[0]
        return self._logits[:batch].clone()


def _batch(batch_size: int = 2):
    return torch.randn(batch_size, 3, 8, 8), {
        "label": torch.zeros(batch_size, dtype=torch.long)
    }


def test_multi_model_soft_voting_workflow_returns_consolidated_output() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.70, 0.30], [0.20, 0.80]], dtype=torch.float32)),
        _Model(torch.tensor([[0.90, 0.10], [0.40, 0.60]], dtype=torch.float32)),
        _Model(torch.tensor([[0.80, 0.20], [0.60, 0.40]], dtype=torch.float32)),
    ]

    out = inf.run_ensemble(models, _batch(batch_size=2))

    expected_agg = torch.tensor([[0.80, 0.20], [0.40, 0.60]], dtype=torch.float32)
    expected_preds = torch.tensor([0, 1], dtype=torch.long)

    assert out["mode"] == "soft"
    assert out["status"] == "completed"
    assert out["failed_models"] == []
    assert torch.allclose(out["aggregated"], expected_agg)
    assert torch.equal(out["preds"], expected_preds)
