from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Optimizer

from .base import Trainer


class ClassificationTrainer(Trainer):
    """Trainer for image classification tasks.

    Provides training with classification-specific metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    """

    def __init__(
        self,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the classification trainer.

        Args:
            optimizer: PyTorch optimizer
            device: Device to use for training (default: "cpu")

        """
        super().__init__(optimizer=optimizer, device=device)

    def fit(
        self,
        model,
        train_loader,
        val_loader=None,
        num_epochs: int = 1,
    ) -> dict[str, Any]:
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

        history: dict[str, list[dict[str, float]]] = {
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
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            model: Classification model
            train_loader: Training data loader

        Returns:
            Dictionary with epoch metrics (loss, accuracy, etc.)

        """
        model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0

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
                outputs = model.forward(inputs, targets=None)
                preds = outputs["labels"].detach().cpu().numpy()
                labels = targets["label"].detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / max(1, num_batches)

        return metrics

    @torch.no_grad()
    def validate(
        self,
        model,
        val_loader,
    ) -> dict[str, float]:
        """Validate the model.

        Args:
            model: Classification model
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics

        """
        model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0

        for batch in val_loader:
            inputs, targets = self.move_batch_to_device(batch)
            losses = model.validation_step((inputs, targets))
            loss = losses["loss"]

            total_loss += float(loss.item())
            num_batches += 1

            # Collect predictions and labels for metrics
            outputs = model.forward(inputs, targets=None)
            preds = outputs["labels"].detach().cpu().numpy()
            labels = targets["label"].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / max(1, num_batches)

        return metrics

    @staticmethod
    def _compute_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
        """Compute classification metrics.

        Args:
            labels: True labels
            preds: Predicted labels

        Returns:
            Dictionary with accuracy, precision, recall, and f1 score

        """
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
