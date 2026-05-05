from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..types import EvaluatorOutput


class BaseEvaluator(ABC):
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0

    def update_loss(self, loss: torch.Tensor | float, batch_size: int) -> None:
        if loss is None:
            return
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = float(loss)

        self.loss_sum += loss_value * batch_size
        self.loss_count += batch_size

    def base_metrics(self) -> EvaluatorOutput:
        return {
            "loss": self.loss_sum / self.loss_count if self.loss_count else 0.0,
        }

    @abstractmethod
    def update(self, predictions: Any, targets: Any, loss: Any) -> None:
        pass

    @abstractmethod
    def compute(self) -> EvaluatorOutput:
        pass

class LossEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def update(self, predictions: Any, targets: Any, loss: Any) -> None:
        batch_size = self._infer_batch_size(predictions, targets)
        self.update_loss(loss=loss, batch_size=batch_size)

    def compute(self) -> EvaluatorOutput:
        return self.base_metrics()

    @staticmethod
    def _infer_batch_size(predictions: Any, targets: Any) -> int:
        if isinstance(targets, torch.Tensor):
            if targets.ndim > 0:
                return int(targets.shape[0])
            return int(targets.numel() or 1)

        if isinstance(targets, dict):
            label = targets.get("label")
            if isinstance(label, torch.Tensor):
                if label.ndim > 0:
                    return int(label.shape[0])
                return int(label.numel() or 1)

            for value in targets.values():
                if isinstance(value, torch.Tensor):
                    if value.ndim > 0:
                        return int(value.shape[0])
                    return int(value.numel() or 1)
                if isinstance(value, (list, tuple)):
                    return len(value)
            return 1

        if isinstance(targets, (list, tuple)):
            return len(targets)

        if isinstance(predictions, torch.Tensor):
            if predictions.ndim > 0:
                return int(predictions.shape[0])
            return int(predictions.numel() or 1)

        return 1
