"""Integration tests for unified image and video inference paths."""

# ruff: noqa: D103

from __future__ import annotations

import torch

from vision_studio.inference.simple import SimpleInference


class _Model(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.7, 0.3]], dtype=torch.float32).repeat(x.shape[0], 1)


def test_unified_image_and_video_paths_return_expected_modes() -> None:
    inf = SimpleInference()
    model = _Model()

    image_batch = (torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})
    image_out = inf.predict_image(model, image_batch)
    assert image_out["mode"] == "image"
    assert image_out["metrics"] == {}

    video_frames = [
        (torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])}),
        (torch.randn(1, 3, 8, 8), {"label": torch.tensor([1])}),
    ]
    video_out = inf.predict_video_stream(model, video_frames)
    assert video_out["mode"] == "video"
    assert video_out["metrics"] == {}
    assert video_out["preds"].shape[0] == 2
