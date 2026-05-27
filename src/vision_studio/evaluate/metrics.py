"""Metric aggregation interfaces for evaluation workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vision_studio.types import EvaluatorOutput


class EvaluationMetrics(ABC):
    """Base contract for metric aggregation implementations."""

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated metric state before a new evaluation run."""
        raise NotImplementedError

    @abstractmethod
    def update(self, predictions: Any, targets: Any, loss: Any) -> None:
        """Consume one evaluated batch of predictions, targets, and loss."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> EvaluatorOutput:
        """Return the aggregated metric output for the current evaluation run."""
        raise NotImplementedError
