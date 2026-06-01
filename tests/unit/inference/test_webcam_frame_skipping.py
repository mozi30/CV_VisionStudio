"""Tests for webcam frame-skipping behavior in realtime mode."""

# ruff: noqa: D103

from __future__ import annotations

import torch

from vision_studio.inference.simple import SimpleInference
from vision_studio.types import RealtimeInferenceConfig


class _Model(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.8, 0.2]], dtype=torch.float32).repeat(x.shape[0], 1)


def _frames(count: int):
    return [
        (torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])}) for _ in range(count)
    ]


def test_webcam_frame_skipping_applies_sampling_rule() -> None:
    inf = SimpleInference()
    model = _Model()
    cfg = RealtimeInferenceConfig(frame_skip=1)
    out = inf.predict_webcam_stream(model, _frames(6), cfg)
    assert out["mode"] == "webcam"
    assert out["preds"].shape[0] == 3
