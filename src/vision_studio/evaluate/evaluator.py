"""Evaluator interfaces and loop implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from vision_studio.models.base import BaseModel
from vision_studio.types import EvaluatorOutput

Batch = tuple[Tensor, dict[str, Any]]


class Evaluator(ABC):
    """Base interface for evaluation-loop implementations."""

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        dataset: Iterable[Batch],
    ) -> EvaluatorOutput:
        """Run evaluation for a model on a dataset and return metrics."""
        raise NotImplementedError


class LoopEvaluator(Evaluator):
    """Default evaluator that iterates a dataset and updates metrics."""

    def __init__(self, metrics: Any, device: torch.device | str = "cpu") -> None:
        """Create a loop evaluator for the provided metrics implementation."""
        self.metrics = metrics
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModel,
        dataset: Iterable[Batch],
    ) -> EvaluatorOutput:
        """Evaluate a model over a dataset and return aggregated metrics."""
        self.metrics.reset()
        was_training = model.training
        model.eval()

        for batch in dataset:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            moved_targets: dict[str, Any] = {}
            for key, value in targets.items():
                moved_targets[key] = (
                    value.to(self.device) if isinstance(value, Tensor) else value
                )

            logits = model(inputs)
            losses = model.compute_loss(logits, moved_targets)
            outputs = model.postprocess(logits)
            self.metrics.update(outputs, moved_targets, losses["loss"])

        if was_training:
            model.train()
        return self.metrics.compute()
