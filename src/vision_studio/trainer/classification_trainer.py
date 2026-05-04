from __future__ import annotations

from typing import Any
import torch
from torch.optim import Optimizer

from .base import Trainer
from ..evaluate import ClassificationEvaluator


class ClassificationTrainer(Trainer):
    """Trainer for image classification tasks.

    Provides training with classification-specific metrics:
    - Accuracy
    - Precision (macro/micro)
    - Recall (macro/micro)
    - F1 Score (macro/micro)
    - Confusion Matrix
    - Top-K Accuracy
    """

    def __init__(
        self,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
        num_classes: int | None = None,
        topk: tuple[int, ...] = (1, 5),
    ) -> None:
        """Initialize the classification trainer.

        Args:
            optimizer: PyTorch optimizer
            device: Device to use for training (default: "cpu")
            num_classes: Number of classes for evaluation metrics
            topk: Tuple of k values for top-k accuracy computation

        """
        super().__init__(optimizer=optimizer, device=device)
        self.num_classes = num_classes
        self.topk = topk
        self.evaluator = (
            ClassificationEvaluator(num_classes=num_classes, topk=topk)
            if num_classes is not None
            else None
        )

    def fit(
        self,
        model,
        train_loader,
        val_loader=None,
        num_epochs: int = 1,
    ) -> dict[str, list[dict[str, Any]]]:
        """Train the model for multiple epochs.

        Args:
            model: Classification model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train

        Returns:
            Training history with train and validation metrics

        """
        model.to(self.device)

        history: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(model, train_loader)
            history["train"].append(train_metrics)

            if val_loader is not None:
                val_metrics = self.validate(model, val_loader)
                history["val"].append(val_metrics)

        return history

    def train_epoch(
        self,
        model,
        train_loader,
    ) -> dict[str, Any]:
        """Train for one epoch.

        Args:
            model: Classification model
            train_loader: Training data loader

        Returns:
            Dictionary with epoch metrics (loss, accuracy, etc.)

        """
        model.train()

        total_loss = 0.0
        num_batches = 0

        # Reset evaluator if available
        if self.evaluator is not None:
            self.evaluator.reset()

        for batch in train_loader:
            inputs, targets = self.move_batch_to_device(batch)

            self.optimizer.zero_grad()

            losses = model.training_step((inputs, targets))
            loss = losses["loss"]

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            self.global_step += 1

            # Collect predictions and labels for metrics
            with torch.no_grad():
                logits = model.forward(inputs)
                outputs = model.postprocess(logits)
                preds = outputs["logits"].detach().cpu()
                labels = targets["label"].detach().cpu()

                if self.evaluator is not None:
                    self.evaluator.update(preds, labels)

        # Compute metrics
        metrics: dict[str, Any] = {}
        if self.evaluator is not None:
            metrics = dict(self.evaluator.compute())
        metrics["loss"] = total_loss / max(1, num_batches)

        return metrics

    @torch.no_grad()
    def validate(
        self,
        model,
        val_loader,
    ) -> dict[str, Any]:
        """Validate the model.

        Args:
            model: Classification model
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics

        """
        model.eval()

        total_loss = 0.0
        num_batches = 0

        # Reset evaluator if available
        if self.evaluator is not None:
            self.evaluator.reset()

        for batch in val_loader:
            inputs, targets = self.move_batch_to_device(batch)
            losses = model.validation_step((inputs, targets))
            loss = losses["loss"]

            total_loss += float(loss.item())
            num_batches += 1

            # Collect predictions and labels for metrics
            logits = model.forward(inputs)
            outputs = model.postprocess(logits)
            preds = outputs["logits"].detach().cpu()
            labels = targets["label"].detach().cpu()

            if self.evaluator is not None:
                self.evaluator.update(preds, labels)

        # Compute metrics
        metrics: dict[str, Any] = {}
        if self.evaluator is not None:
            metrics = dict(self.evaluator.compute())
        metrics["loss"] = total_loss / max(1, num_batches)

        return metrics
