from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor
from torch.nn import Module

from .base import Batch, Inference


class ClassificationInference(Inference):
    """Inference engine for image classification with metrics."""

    @torch.no_grad()
    def predict(
        self,
        model: Module,
        data_loader: Iterable[Batch],
    ) -> dict[str, Any]:
        """Run inference on entire dataset and compute metrics.

        Args:
            model: Classification model
            data_loader: Data loader with batches

        Returns:
            Dictionary containing predictions, targets, and metrics

        """
        model.to(self.device)
        model.eval()

        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_probs: list[Tensor] = []

        for batch in data_loader:
            batch_output = self.predict_batch(model, batch)

            preds = batch_output["preds"]
            labels = batch_output["labels"]
            probs = batch_output.get("probs")

            all_preds.append(preds)
            all_labels.append(labels)
            if probs is not None:
                all_probs.append(probs.detach().cpu())

        # Concatenate all predictions and labels
        preds_array = np.concatenate(all_preds, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)

        # Compute metrics
        metrics = self._compute_metrics(labels_array, preds_array)

        result = {
            "preds": torch.from_numpy(preds_array),
            "labels": torch.from_numpy(labels_array),
            "metrics": metrics,
        }

        if all_probs:
            result["probs"] = torch.cat(all_probs, dim=0)

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        model: Module,
        batch: Batch,
    ) -> dict[str, Any]:
        """Run inference on a single batch.

        Args:
            model: Classification model
            batch: Input batch

        Returns:
            Dictionary with predictions, labels, and optionally probabilities

        """
        inputs, targets = self.move_batch_to_device(batch)
        logits = model(inputs)
        output = model.postprocess(logits)

        preds = output["labels"].detach().cpu().numpy()
        labels = targets["label"].detach().cpu().numpy()
        probs = output.get("probs")

        return {
            "preds": preds,
            "labels": labels,
            "probs": probs,
        }

    @staticmethod
    def _compute_metrics(
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> dict[str, float | np.ndarray]:
        """Compute classification metrics.

        Args:
            labels: Ground truth labels
            preds: Predicted labels

        Returns:
            Dictionary with accuracy, precision, recall, f1, and confusion matrix

        """
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(labels, preds)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": conf_matrix,
        }
