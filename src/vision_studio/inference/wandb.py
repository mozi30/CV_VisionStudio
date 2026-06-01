"""Legacy wandb inference entrypoint routed through unified inference flow."""

# ruff: noqa: D101,D102,D107

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.nn import Module

from .base import Batch
from .simple import SimpleInference


class WandbInference(SimpleInference):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        project: str = "my-inference-project",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        log_every_n_steps: int = 10,
        use_wandb: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.log_every_n_steps = log_every_n_steps
        self.use_wandb = use_wandb
        self.global_step = 0
        self._wandb_initialized = False

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
