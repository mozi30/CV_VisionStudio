"""Tests for hard-voting ensemble mode and tie policies."""

# ruff: noqa: D103

from __future__ import annotations

import torch

from vision_studio.inference.simple import EnsembleConfig, SimpleInference


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


def test_hard_voting_majority_selects_most_common_label() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)),
        _Model(torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)),
        _Model(torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)),
    ]
    out = inf.run_ensemble(models, _batch(), EnsembleConfig(mode="hard"))
    assert torch.equal(out["preds"], torch.tensor([0, 0]))


def test_hard_voting_default_tie_policy_index_priority() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.9, 0.1]], dtype=torch.float32)),
        _Model(torch.tensor([[0.1, 0.9]], dtype=torch.float32)),
    ]
    out = inf.run_ensemble(models, _batch(batch_size=1), EnsembleConfig(mode="hard"))
    assert out["tie_policy"] == "index-priority"
    assert torch.equal(out["preds"], torch.tensor([0]))


def test_hard_voting_random_tie_policy_returns_valid_class() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.9, 0.1]], dtype=torch.float32)),
        _Model(torch.tensor([[0.1, 0.9]], dtype=torch.float32)),
    ]
    out = inf.run_ensemble(
        models,
        _batch(batch_size=1),
        EnsembleConfig(mode="hard", tie_policy="random"),
    )
    assert int(out["preds"][0].item()) in {0, 1}
