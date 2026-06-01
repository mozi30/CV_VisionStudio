"""Legacy classification inference entrypoint routed through unified flow."""

# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.nn import Module

from .base import Batch
from .simple import SimpleInference


class ClassificationInference(SimpleInference):
    """Legacy classification entrypoint routed through unified inference."""

    @torch.no_grad()
    def predict(
        self,
        model: Module,
        data_loader: Iterable[Batch],
    ) -> dict[str, Any]:
        return super().predict(model, data_loader)

    @torch.no_grad()
    def predict_batch(
        self,
        model: Module,
        batch: Batch,
    ) -> dict[str, Any]:
        return super().predict_batch(model, batch)
