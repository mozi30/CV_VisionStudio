"""Tests for basic ensemble validation constraints."""

# ruff: noqa: D103

from __future__ import annotations

import pytest
import torch

from vision_studio.inference.simple import SimpleInference
from vision_studio.types import ConfigurationError


class _Model(torch.nn.Module):
    def __init__(self, out_dim: int = 3):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        return torch.ones((x.shape[0], self.out_dim), dtype=torch.float32)


def _batch(batch_size: int = 2):
    return torch.randn(batch_size, 3, 8, 8), {
        "label": torch.zeros(batch_size, dtype=torch.long)
    }


def test_empty_model_list_rejected() -> None:
    inf = SimpleInference()
    with pytest.raises(ConfigurationError):
        inf.run_ensemble([], _batch())


def test_mismatched_class_count_rejected() -> None:
    inf = SimpleInference()
    m1 = _Model(3)
    m2 = _Model(4)
    with pytest.raises(ConfigurationError):
        inf.run_ensemble([m1, m2], _batch())
