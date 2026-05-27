"""Logging-based reporter implementation."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseReporter


class LoggingReporter(BaseReporter):
    """Reporter that writes metrics through the standard logger."""

    def __init__(self) -> None:
        """Create a logging reporter instance."""
        super().__init__()
        self.logger = logging.getLogger("vision_studio.reporting")

    def start(self) -> None:
        """Start the logging reporter lifecycle."""
        self.logger.debug("Logging reporter started")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Emit metric payloads to the logger."""
        payload = {"step": step, **metrics} if step is not None else metrics
        self.logger.info("metrics=%s", payload)

    def finish(self) -> None:
        """Finish the logging reporter lifecycle."""
        self.logger.debug("Logging reporter finished")
