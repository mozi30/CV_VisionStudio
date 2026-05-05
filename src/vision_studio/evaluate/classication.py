from typing import cast

import torch

from .base import BaseEvaluator
from ..types import ClassificationEvaluatorOutput

class ClassificationEvaluator(BaseEvaluator):
    def __init__(
        self,
        num_classes: int,
        topk: tuple[int, ...] = (1, 5),
        average: str = "macro",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.average = average
        self.reset()

    def reset(self) -> None:
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
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        predictions: Tensor [B, C] logits/probabilities
        targets: Tensor [B] class indices
        """
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()
        batch_size = targets.numel()
        self.update_loss(loss, batch_size)

        self.total += targets.numel()

        pred_classes = predictions.argmax(dim=1)

        for true, pred in zip(targets, pred_classes):
            self.confusion_matrix[true.long(), pred.long()] += 1

        max_k = min(max(self.topk), predictions.shape[1])
        _, topk_preds = predictions.topk(max_k, dim=1)

        for k in self.topk:
            if k <= predictions.shape[1]:
                correct = topk_preds[:, :k].eq(targets.view(-1, 1)).any(dim=1)
                self.topk_correct[k] += correct.sum().item()

    def compute(self) -> ClassificationEvaluatorOutput:
        cm = self.confusion_matrix.float()

        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)
        f1_per_class = (
            2 * precision_per_class * recall_per_class
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
            2 * precision_micro * recall_micro
            / (precision_micro + recall_micro + 1e-8)
        )

        metrics.update({
            "precision_micro": precision_micro.item(),
            "recall_micro": recall_micro.item(),
            "f1_micro": f1_micro.item(),
        })

        for k in self.topk:
            if self.total > 0:
                metrics[f"top_{k}_accuracy"] = self.topk_correct[k] / self.total

        return metrics

    def get_confusion_matrix(self) -> torch.Tensor:
        return self.confusion_matrix
