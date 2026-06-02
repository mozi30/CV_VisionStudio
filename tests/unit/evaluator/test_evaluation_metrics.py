"""Unit tests for evaluation metric adapters."""

import pytest
import torch

from vision_studio.evaluate import (
    ClassificationEvaluationMetrics,
    ClassificationEvaluationPerClassMetrics,
    ConfusionMatrixEvaluationMetrics,
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


def test_classification_evaluation_metrics_accept_postprocess_dicts() -> None:
    """Classification metrics should consume model postprocess outputs."""
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    predictions = {
        "logits": torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        "probs": torch.tensor([[0.3, 0.7], [0.7, 0.3]]),
        "labels": torch.tensor([1, 0]),
    }
    targets = {"label": torch.tensor([1, 0])}

    metrics.update(predictions, targets, torch.tensor(0.5))
    result = metrics.compute()

    assert result["loss"] == 0.5
    assert result["accuracy"] == 1.0


def test_confusion_matrix_evaluation_metrics_from_logits() -> None:
    """ConfusionMatrixEvaluationMetrics should accumulate class counts."""
    metrics = ConfusionMatrixEvaluationMetrics(num_classes=3)
    predictions = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    targets = torch.tensor([0, 2, 1, 2])

    metrics.update(predictions, targets, torch.tensor(0.25))
    result = metrics.compute()

    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )
    assert result["loss"] == 0.25
    assert torch.equal(result["confusion_matrix"], expected)


def test_confusion_matrix_evaluation_metrics_accept_dicts_and_normalize() -> None:
    """ConfusionMatrixEvaluationMetrics should accept postprocess-style dicts."""
    metrics = ConfusionMatrixEvaluationMetrics(num_classes=2, normalize=True)

    metrics.update(
        {"labels": torch.tensor([0, 1, 1])},
        {"label": torch.tensor([0, 0, 1])},
        torch.tensor(0.5),
    )
    result = metrics.compute()

    assert torch.equal(
        result["confusion_matrix"],
        torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
    )
    assert torch.allclose(
        result["normalized_confusion_matrix"],
        torch.tensor([[0.5, 0.5], [0.0, 1.0]]),
    )


def test_confusion_matrix_evaluation_metrics_reset_clears_state() -> None:
    """ConfusionMatrixEvaluationMetrics should reset matrix and loss."""
    metrics = ConfusionMatrixEvaluationMetrics(num_classes=2)
    metrics.update(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor(1.0))

    metrics.reset()
    result = metrics.compute()

    assert result["loss"] == 0.0
    assert torch.equal(result["confusion_matrix"], torch.zeros(2, 2, dtype=torch.long))


def test_classification_evaluation_per_class_metrics_from_logits() -> None:
    """ClassificationEvaluationPerClassMetrics should report class metrics."""
    metrics = ClassificationEvaluationPerClassMetrics(
        num_classes=3,
        class_names=["red", "green", "blue"],
    )
    predictions = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    targets = torch.tensor([0, 2, 1, 2])

    metrics.update(predictions, targets, torch.tensor(0.25))
    result = metrics.compute()

    assert result["loss"] == 0.25
    assert result["support_per_class"] == [1, 1, 2]
    assert result["precision_per_class"] == pytest.approx([1.0, 0.5, 1.0])
    assert result["recall_per_class"] == pytest.approx([1.0, 1.0, 0.5])
    assert result["f1_per_class"] == pytest.approx([1.0, 2 / 3, 2 / 3])
    assert result["per_class"]["red"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1,
    }


def test_classification_evaluation_per_class_metrics_accept_dict_labels() -> None:
    """ClassificationEvaluationPerClassMetrics should accept label dictionaries."""
    metrics = ClassificationEvaluationPerClassMetrics(num_classes=2)

    metrics.update(
        {"labels": torch.tensor([0, 1, 1])},
        {"label": torch.tensor([0, 0, 1])},
        torch.tensor(0.5),
    )
    result = metrics.compute()

    assert result["support_per_class"] == [2, 1]
    assert result["per_class"]["0"]["precision"] == 1.0
    assert result["per_class"]["0"]["recall"] == 0.5
    assert result["per_class"]["1"]["precision"] == 0.5
    assert result["per_class"]["1"]["recall"] == 1.0


def test_classification_evaluation_per_class_metrics_validates_class_names() -> None:
    """ClassificationEvaluationPerClassMetrics should validate class name count."""
    try:
        ClassificationEvaluationPerClassMetrics(num_classes=2, class_names=["only-one"])
    except ValueError as exc:
        assert "class_names length" in str(exc)
    else:
        raise AssertionError("Expected class_names validation to fail.")


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
