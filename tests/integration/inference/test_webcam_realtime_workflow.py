"""Integration tests for realtime webcam inference workflow behavior."""

# ruff: noqa: D103

from __future__ import annotations

from pathlib import Path

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


def test_webcam_realtime_workflow_sampling_and_recording(tmp_path: Path) -> None:
    inf = SimpleInference()
    model = _Model()
    out_path = tmp_path / "webcam.pt"
    cfg = RealtimeInferenceConfig(
        frame_skip=1, record_output=True, output_path=out_path
    )

    out = inf.predict_webcam_stream(model, _frames(6), cfg)

    assert out["mode"] == "webcam"
    assert out["preds"].shape[0] == 3
    assert out["artifact"] == str(out_path)
    assert out_path.exists()
