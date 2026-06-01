"""Tests for base inference validation contract helpers."""

# ruff: noqa: D103

from __future__ import annotations

import pytest

from vision_studio.inference.base import Inference
from vision_studio.types import ConfigurationError


class _DummyInference(Inference):
    def predict(self, model, data_loader):  # pragma: no cover
        return {}

    def predict_batch(self, model, batch):  # pragma: no cover
        return {}


def test_validate_non_empty_models_rejects_empty() -> None:
    with pytest.raises(ConfigurationError):
        _DummyInference.validate_non_empty_models([])


def test_validate_frame_skip_rejects_negative() -> None:
    with pytest.raises(ConfigurationError):
        _DummyInference.validate_frame_skip(-1)
