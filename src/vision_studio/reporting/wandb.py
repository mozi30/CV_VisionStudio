"""WandB reporter implementation."""

from __future__ import annotations

from typing import Any

import wandb

from vision_studio.types import ReportingError

from .base import BaseReporter, ReporterContext


class WandbReporter(BaseReporter):
    """Reporter that publishes metrics to Weights & Biases."""

    def __init__(self, context: ReporterContext | None = None) -> None:
        """Create a WandB reporter with optional run context."""
        super().__init__(context=context)
        self._started = False

    def start(self) -> None:
        """Initialize the WandB run or raise a reporting error."""
        try:
            wandb.init(
                project=self.context.project,
                name=self.context.run_name,
                config=self.context.config,
            )
        except Exception as exc:  # noqa: BLE001
            raise ReportingError("Failed to initialize WandB reporter") from exc
        self._started = True

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Send metrics to the active WandB run."""
        if not self._started:
            raise ReportingError("WandB reporter must be started before logging")
        try:
            wandb.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            raise ReportingError("Failed to log metrics to WandB") from exc

    def finish(self) -> None:
        """Finish the active WandB run if it was started."""
        if self._started:
            wandb.finish()
            self._started = False
