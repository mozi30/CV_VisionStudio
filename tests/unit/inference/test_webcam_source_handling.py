"""Tests for webcam source validation error handling."""

# ruff: noqa: D103

from __future__ import annotations

import pytest
import torch

from vision_studio.inference.simple import SimpleInference
from vision_studio.types import ConfigurationError, RealtimeInferenceConfig


class _Model(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.8, 0.2]], dtype=torch.float32).repeat(x.shape[0], 1)


def _frames():
    return [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]


def test_webcam_source_rejects_negative_index() -> None:
    inf = SimpleInference()
    cfg = RealtimeInferenceConfig(source=-1)
    with pytest.raises(ConfigurationError):
        inf.predict_webcam_stream(_Model(), _frames(), cfg)


def test_webcam_source_rejects_empty_string() -> None:
    inf = SimpleInference()
    cfg = RealtimeInferenceConfig(source="")
    with pytest.raises(ConfigurationError):
        inf.predict_webcam_stream(_Model(), _frames(), cfg)
