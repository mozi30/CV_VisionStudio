"""Local live-plot reporter implementation."""

from __future__ import annotations

from typing import Any

from .logging import LoggingReporter


class LivePlotReporter(LoggingReporter):
    """Reporter that falls back to logging when live plotting is unavailable."""

    def __init__(self) -> None:
        """Create a live-plot reporter with logging fallback behavior."""
        super().__init__()

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics locally while standing in for live plotting."""
        super().log(metrics, step=step)
