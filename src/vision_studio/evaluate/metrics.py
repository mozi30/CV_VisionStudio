"""Metric aggregation interfaces for evaluation workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

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


class LossEvaluationMetrics(EvaluationMetrics):
    """Aggregate explicit loss-only evaluation metrics."""

    def __init__(self) -> None:
        """Create a loss-only evaluation metrics accumulator."""
        self.reset()

    def reset(self) -> None:
        """Reset stored loss statistics."""
        self.loss_sum = 0.0
        self.loss_count = 0

    def update(self, predictions: Any, targets: Any, loss: Any) -> None:
        """Add one batch loss value into the aggregated loss metrics."""
        if loss is None:
            raise ValueError("LossEvaluationMetrics requires a loss value.")
        batch_size = self._infer_batch_size(predictions, targets)
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = float(loss)
        self.loss_sum += loss_value * batch_size
        self.loss_count += batch_size

    def compute(self) -> EvaluatorOutput:
        """Return averaged loss for all accumulated batches."""
        return {"loss": self.loss_sum / self.loss_count if self.loss_count else 0.0}

    @staticmethod
    def _infer_batch_size(predictions: Any, targets: Any) -> int:
        """Infer batch size from prediction or target structures."""
        if isinstance(targets, torch.Tensor):
            return (
                int(targets.shape[0]) if targets.ndim > 0 else int(targets.numel() or 1)
            )
        if isinstance(targets, dict):
            for value in targets.values():
                if isinstance(value, torch.Tensor):
                    return (
                        int(value.shape[0])
                        if value.ndim > 0
                        else int(value.numel() or 1)
                    )
                if isinstance(value, (list, tuple)):
                    return len(value)
        if isinstance(targets, (list, tuple)):
            return len(targets)
        if isinstance(predictions, torch.Tensor):
            return (
                int(predictions.shape[0])
                if predictions.ndim > 0
                else int(predictions.numel() or 1)
            )
        return 1
