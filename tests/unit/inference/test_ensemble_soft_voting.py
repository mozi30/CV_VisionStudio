"""Tests for default soft-voting ensemble behavior."""

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
        if self._logits.shape[0] == batch:
            return self._logits.clone()
        return self._logits[:batch].clone()


def _batch(batch_size: int = 2):
    return torch.randn(batch_size, 3, 8, 8), {
        "label": torch.zeros(batch_size, dtype=torch.long)
    }


def test_soft_voting_default_mode_aggregates_mean_logits() -> None:
    inf = SimpleInference()
    m1 = _Model(torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32))
    m2 = _Model(torch.tensor([[0.6, 0.4], [0.9, 0.1]], dtype=torch.float32))

    out = inf.run_ensemble([m1, m2], _batch())

    expected_agg = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32)
    expected_labels = torch.tensor([0, 0], dtype=torch.long)
    assert out["mode"] == "soft"
    assert torch.allclose(out["aggregated"], expected_agg)
    assert torch.equal(out["preds"], expected_labels)


def test_soft_voting_emits_expected_default_policy_metadata() -> None:
    inf = SimpleInference()
    m1 = _Model(torch.tensor([[0.8, 0.2]], dtype=torch.float32))
    m2 = _Model(torch.tensor([[0.6, 0.4]], dtype=torch.float32))

    out = inf.run_ensemble([m1, m2], _batch(batch_size=1))

    assert out["mode"] == "soft"
    assert out["tie_policy"] == "index-priority"
    assert out["failure_policy"] == "fail-fast"
    assert out["failed_models"] == []
    assert out["status"] == "completed"
