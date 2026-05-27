"""Unit tests for evaluation metric adapters."""

import torch

from vision_studio.evaluate import (
    ClassificationEvaluationMetrics,
    DetectionEvaluationMetrics,
    LossEvaluationMetrics,
)


def test_placeholder_classification_evaluation_metrics() -> None:
    """ClassificationEvaluationMetrics should compute basic loss and accuracy."""
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    loss = torch.tensor(0.5)

    metrics.update(predictions, targets, loss)
    result = metrics.compute()

    assert result["loss"] == 0.5
    assert result["accuracy"] == 1.0


def test_placeholder_detection_evaluation_metrics() -> None:
    """DetectionEvaluationMetrics should expose loss and AP-style metrics."""
    metrics = DetectionEvaluationMetrics(num_classes=1, iou_thresholds=[0.5])
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([0]),
        }
    ]
    loss = torch.tensor(0.25)

    metrics.update(predictions, targets, loss)
    result = metrics.compute()

    assert result["loss"] == 0.25
    assert "ap50" in result


def test_placeholder_loss_evaluation_metrics() -> None:
    """LossEvaluationMetrics should average loss values across batches."""
    metrics = LossEvaluationMetrics()
    predictions = torch.ones(2, 2)
    targets = {"label": torch.ones(2, dtype=torch.long)}
    metrics.update(predictions, targets, torch.tensor(0.5))
    metrics.update(predictions, targets, torch.tensor(1.0))

    result = metrics.compute()

    assert result["loss"] == 0.75
