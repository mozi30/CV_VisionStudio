"""Tests for webcam output-recording behavior in realtime mode."""

# ruff: noqa: D103

from __future__ import annotations

from pathlib import Path

import torch

from vision_studio.inference.simple import SimpleInference
from vision_studio.types import RealtimeInferenceConfig


class _Model(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.8, 0.2]], dtype=torch.float32).repeat(x.shape[0], 1)


def test_webcam_recording_writes_artifact(tmp_path: Path) -> None:
    inf = SimpleInference()
    model = _Model()
    out_file = tmp_path / "annotated_output.pt"
    cfg = RealtimeInferenceConfig(record_output=True, output_path=out_file)
    frames = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    out = inf.predict_webcam_stream(model, frames, cfg)

    assert out["artifact"] == str(out_file)
    assert out_file.exists()
