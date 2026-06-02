"""Classification evaluation metric implementations."""

from typing import Any, cast

import torch

from ..types import ClassificationEvaluatorOutput
from .base import BaseEvaluator
from .metrics import EvaluationMetrics


class ClassificationEvaluator(BaseEvaluator):
    """Aggregate classification evaluation metrics."""

    def __init__(
        self,
        num_classes: int,
        topk: tuple[int, ...] = (1, 5),
        average: str = "macro",
    ):
        """Create a classification evaluator for the configured classes."""
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.average = average
        self.reset()

    def reset(self) -> None:
        """Reset accumulated classification statistics."""
        super().reset()
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.long,
        )
        self.total = 0
        self.topk_correct = {k: 0 for k in self.topk}

    def update(
        self,
        predictions: torch.Tensor | dict[str, Any],
        targets: torch.Tensor | dict[str, Any],
        loss: torch.Tensor,
    ) -> None:
        """Accumulate one batch of classification predictions and targets."""
        if isinstance(predictions, dict):
            if "logits" in predictions:
                predictions = predictions["logits"]
            elif "probs" in predictions:
                predictions = predictions["probs"]
            else:
                raise KeyError(
                    "Classification predictions dict must include 'logits' or 'probs'."
                )

        if isinstance(targets, dict):
            if "label" in targets:
                targets = targets["label"]
            elif "labels" in targets:
                targets = targets["labels"]
            else:
                raise KeyError(
                    "Classification targets dict must include 'label' or 'labels'."
                )

        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()
        batch_size = targets.numel()
        self.update_loss(loss, batch_size)

        self.total += targets.numel()

        pred_classes = predictions.argmax(dim=1)

        for true, pred in zip(targets, pred_classes, strict=False):
            self.confusion_matrix[true.long(), pred.long()] += 1

        max_k = min(max(self.topk), predictions.shape[1])
        _, topk_preds = predictions.topk(max_k, dim=1)

        for k in self.topk:
            if k <= predictions.shape[1]:
                correct = topk_preds[:, :k].eq(targets.view(-1, 1)).any(dim=1)
                self.topk_correct[k] += int(correct.sum().item())

    def compute(self) -> ClassificationEvaluatorOutput:
        """Compute aggregated classification metrics for all batches."""
        cm = self.confusion_matrix.float()

        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)
        f1_per_class = (
            2
            * precision_per_class
            * recall_per_class
            / (precision_per_class + recall_per_class + 1e-8)
        )

        accuracy = tp.sum() / (cm.sum() + 1e-8)

        metrics = cast(ClassificationEvaluatorOutput, self.base_metrics())
        metrics.update(
            {
                "accuracy": accuracy.item(),
                "precision_macro": precision_per_class.mean().item(),
                "recall_macro": recall_per_class.mean().item(),
                "f1_macro": f1_per_class.mean().item(),
            }
        )

        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()

        precision_micro = total_tp / (total_tp + total_fp + 1e-8)
        recall_micro = total_tp / (total_tp + total_fn + 1e-8)
        f1_micro = (
            2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)
        )

        metrics.update(
            {
                "precision_micro": precision_micro.item(),
                "recall_micro": recall_micro.item(),
                "f1_micro": f1_micro.item(),
            }
        )

        for k in self.topk:
            if self.total > 0:
                metrics[f"top_{k}_accuracy"] = self.topk_correct[k] / self.total

        return metrics

    def get_confusion_matrix(self) -> torch.Tensor:
        """Return the accumulated confusion matrix."""
        return self.confusion_matrix


class ClassificationEvaluationMetrics(ClassificationEvaluator, EvaluationMetrics):
    """Compatibility adapter exposing classification metrics under the new name."""


class ClassificationEvaluationPerClassMetrics(EvaluationMetrics):
    """Aggregate per-class classification metrics across evaluated batches."""

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> None:
        """Create a per-class metrics accumulator."""
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero.")
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("class_names length must match num_classes.")
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Reset accumulated per-class statistics."""
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.long,
        )
        self.loss_sum = 0.0
        self.loss_count = 0

    def update(
        self,
        predictions: torch.Tensor | dict[str, Any],
        targets: torch.Tensor | dict[str, Any],
        loss: torch.Tensor,
    ) -> None:
        """Accumulate one batch for per-class metrics."""
        prediction_classes = _classification_prediction_classes(predictions)
        target_classes = _classification_target_classes(targets)

        if prediction_classes.numel() != target_classes.numel():
            raise ValueError(
                "ClassificationEvaluationPerClassMetrics requires prediction and "
                "target class counts to match."
            )

        _validate_class_ids(prediction_classes, self.num_classes, "prediction")
        _validate_class_ids(target_classes, self.num_classes, "target")
        self._update_loss(loss, int(target_classes.numel()))

        for true, pred in zip(target_classes, prediction_classes, strict=True):
            self.confusion_matrix[true.long(), pred.long()] += 1

    def compute(self) -> dict[str, Any]:
        """Return per-class precision, recall, F1, support, and confusion matrix."""
        cm = self.confusion_matrix.float()
        true_positive = torch.diag(cm)
        false_positive = cm.sum(dim=0) - true_positive
        false_negative = cm.sum(dim=1) - true_positive
        support = cm.sum(dim=1)

        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        per_class = {
            name: {
                "precision": precision[index].item(),
                "recall": recall[index].item(),
                "f1": f1[index].item(),
                "support": int(support[index].item()),
            }
            for index, name in enumerate(self.class_names)
        }

        return {
            "loss": self.loss_sum / self.loss_count if self.loss_count else 0.0,
            "per_class": per_class,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.long().tolist(),
            "confusion_matrix": self.confusion_matrix.clone(),
        }

    def get_confusion_matrix(self) -> torch.Tensor:
        """Return the accumulated confusion matrix."""
        return self.confusion_matrix.clone()

    def _update_loss(self, loss: torch.Tensor | float, batch_size: int) -> None:
        if loss is None:
            return
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = float(loss)
        self.loss_sum += loss_value * batch_size
        self.loss_count += batch_size


class ConfusionMatrixEvaluationMetrics(EvaluationMetrics):
    """Aggregate a classification confusion matrix across evaluated batches."""

    def __init__(self, num_classes: int, normalize: bool = False) -> None:
        """Create a confusion-matrix accumulator for classification outputs."""
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero.")
        self.num_classes = num_classes
        self.normalize = normalize
        self.reset()

    def reset(self) -> None:
        """Reset accumulated confusion-matrix and loss state."""
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.long,
        )
        self.loss_sum = 0.0
        self.loss_count = 0

    def update(
        self,
        predictions: torch.Tensor | dict[str, Any],
        targets: torch.Tensor | dict[str, Any],
        loss: torch.Tensor,
    ) -> None:
        """Accumulate one batch of predicted and target classes."""
        prediction_classes = _classification_prediction_classes(predictions)
        target_classes = _classification_target_classes(targets)

        if prediction_classes.numel() != target_classes.numel():
            raise ValueError(
                "ConfusionMatrixEvaluationMetrics requires prediction and target "
                "class counts to match."
            )

        _validate_class_ids(prediction_classes, self.num_classes, "prediction")
        _validate_class_ids(target_classes, self.num_classes, "target")
        self._update_loss(loss, int(target_classes.numel()))

        for true, pred in zip(target_classes, prediction_classes, strict=True):
            self.confusion_matrix[true.long(), pred.long()] += 1

    def compute(self) -> dict[str, Any]:
        """Return averaged loss and the accumulated confusion matrix."""
        result: dict[str, Any] = {
            "loss": self.loss_sum / self.loss_count if self.loss_count else 0.0,
            "confusion_matrix": self.confusion_matrix.clone(),
        }
        if self.normalize:
            result["normalized_confusion_matrix"] = self._normalized_matrix()
        return result

    def get_confusion_matrix(self) -> torch.Tensor:
        """Return the accumulated confusion matrix."""
        return self.confusion_matrix.clone()

    def _update_loss(self, loss: torch.Tensor | float, batch_size: int) -> None:
        if loss is None:
            return
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().cpu().item())
        else:
            loss_value = float(loss)
        self.loss_sum += loss_value * batch_size
        self.loss_count += batch_size

    def _normalized_matrix(self) -> torch.Tensor:
        matrix = self.confusion_matrix.float()
        row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        return matrix / row_sums


def _classification_prediction_classes(
    predictions: torch.Tensor | dict[str, Any],
) -> torch.Tensor:
    if isinstance(predictions, dict):
        if "labels" in predictions:
            predictions = predictions["labels"]
        elif "label" in predictions:
            predictions = predictions["label"]
        elif "logits" in predictions:
            predictions = predictions["logits"]
        elif "probs" in predictions:
            predictions = predictions["probs"]
        else:
            raise KeyError(
                "Classification predictions dict must include 'labels', 'label', "
                "'logits', or 'probs'."
            )

    predictions = predictions.detach().cpu()
    if predictions.ndim > 1:
        return predictions.argmax(dim=1).view(-1)
    return predictions.long().view(-1)


def _classification_target_classes(
    targets: torch.Tensor | dict[str, Any],
) -> torch.Tensor:
    if isinstance(targets, dict):
        if "label" in targets:
            targets = targets["label"]
        elif "labels" in targets:
            targets = targets["labels"]
        else:
            raise KeyError(
                "Classification targets dict must include 'label' or 'labels'."
            )

    return targets.detach().cpu().long().view(-1)


def _validate_class_ids(values: torch.Tensor, num_classes: int, name: str) -> None:
    if values.numel() == 0:
        return
    if values.min().item() < 0 or values.max().item() >= num_classes:
        raise ValueError(
            f"Classification {name} classes must be in [0, {num_classes - 1}]."
        )
