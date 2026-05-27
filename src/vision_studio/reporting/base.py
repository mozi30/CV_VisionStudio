"""Reporter abstractions shared by training and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ReporterContext:
    """Configuration describing a reporting session."""

    mode: str
    project: str | None = None
    run_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


class BaseReporter(ABC):
    """Base interface for reporting metric events."""

    def __init__(self, context: ReporterContext | None = None) -> None:
        """Create a reporter with optional reporting context."""
        self.context = context or ReporterContext(mode="logging")

    @abstractmethod
    def start(self) -> None:
        """Initialize any reporter resources before logging begins."""
        raise NotImplementedError

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Record metrics for an optional training or evaluation step."""
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> None:
        """Finalize reporter state after logging completes."""
        raise NotImplementedError
