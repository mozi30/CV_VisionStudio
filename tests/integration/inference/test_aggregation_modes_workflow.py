"""Integration tests for comparing ensemble aggregation modes."""

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


def test_aggregation_modes_produce_policy_conformant_outputs() -> None:
    inf = SimpleInference()
    models: list[torch.nn.Module] = [
        _Model(torch.tensor([[0.9, 0.1]], dtype=torch.float32)),
        _Model(torch.tensor([[0.1, 0.9]], dtype=torch.float32)),
        _Model(torch.tensor([[0.2, 0.8]], dtype=torch.float32)),
    ]
    batch = _batch(batch_size=1)

    soft = inf.run_ensemble(models, batch, EnsembleConfig(mode="soft"))
    hard = inf.run_ensemble(models, batch, EnsembleConfig(mode="hard"))
    weighted = inf.run_ensemble(
        models,
        batch,
        EnsembleConfig(mode="weighted", weights=[0.8, 0.1, 0.1]),
    )

    assert soft["mode"] == "soft"
    assert hard["mode"] == "hard"
    assert weighted["mode"] == "weighted"

    assert int(soft["preds"][0].item()) == 1
    assert int(hard["preds"][0].item()) == 1
    assert int(weighted["preds"][0].item()) == 0
