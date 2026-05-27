"""General Trainer implementation for training-loop orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from vision_studio.models.base import BaseModel
from vision_studio.types import EvaluatorOutput

from .base import Trainer

Batch = tuple[Tensor, dict[str, Any]]


class VisionTrainer(Trainer):
    """Concrete Trainer implementation for the MVP training workflow."""

    def __init__(
        self,
        optimizer,
        evaluator: Any | None = None,
        device: torch.device | str = "cpu",
        settings=None,
    ) -> None:
        """Initialize a Trainer with an optional default Evaluator."""
        super().__init__(optimizer=optimizer, device=device, settings=settings)
        self.default_evaluator = evaluator

    def fit(
        self,
        model: BaseModel,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
        evaluator: Any | None = None,
    ) -> dict[str, Any]:
        """Train a model and optionally run evaluation after each epoch."""
        self.validate_settings()
        chosen_evaluator = (
            evaluator if evaluator is not None else self.default_evaluator
        )

        model.to(self.device)
        history: dict[str, list[EvaluatorOutput]] = {"train": [], "evaluation": []}

        for epoch in range(self.current_epoch, self.settings.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(model, train_loader)
            history["train"].append(train_metrics)

            if chosen_evaluator is None or val_loader is None:
                if chosen_evaluator is None:
                    self.warn("No Evaluator configured; evaluation skipped.")
                continue

            evaluation_metrics = chosen_evaluator.evaluate(model, val_loader)
            history["evaluation"].append(evaluation_metrics)

        return {
            "history": history,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "warnings": self.warnings,
        }

    def train_epoch(
        self,
        model: BaseModel,
        train_loader: Iterable[Batch],
    ) -> EvaluatorOutput:
        """Run a single training epoch and return aggregated loss metrics."""
        model.train()

        loss_sum = 0.0
        loss_count = 0
        for batch in train_loader:
            inputs, targets = self.move_batch_to_device(batch)
            self.optimizer.zero_grad()
            logits = model(inputs)
            metrics = model.compute_loss(logits, targets)
            loss = metrics["loss"]
            loss.backward()
            self.optimizer.step()

            batch_size = int(inputs.shape[0]) if hasattr(inputs, "shape") else 1
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            loss_count += batch_size
            self.global_step += 1

        return {"loss": loss_sum / loss_count if loss_count else 0.0}
