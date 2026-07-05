"""Integration tests for ensemble evaluator workflows."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vision_studio.evaluate import ClassificationEvaluationMetrics, LoopEvaluator
from vision_studio.inference.simple import EnsembleConfig
from vision_studio.models.base import BaseModel
from vision_studio.types import LossOutput, PostprocessOutput


class _WorkflowModel(BaseModel):
    def __init__(self, logits: torch.Tensor, loss: float):
        super().__init__()
        self._logits = logits
        self._loss = loss

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._logits[: inputs.shape[0]].clone()

    def postprocess(self, logits: torch.Tensor) -> PostprocessOutput:
        return {"logits": logits}

    def compute_loss(self, logits: torch.Tensor, targets: dict[str, Any]) -> LossOutput:
        return {"loss": torch.tensor(self._loss, dtype=torch.float32)}


def test_ensemble_evaluation_workflow_over_dataset() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models: list[BaseModel] = [
        _WorkflowModel(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), loss=0.2),
        _WorkflowModel(torch.tensor([[0.7, 0.3], [0.1, 0.9]]), loss=0.4),
    ]
    dataset = [
        (
            torch.randn(2, 3, 8, 8),
            {"label": torch.tensor([0, 1], dtype=torch.long)},
        ),
        (
            torch.randn(1, 3, 8, 8),
            {"label": torch.tensor([0], dtype=torch.long)},
        ),
    ]

    result = evaluator.evaluate_ensemble(
        models,
        dataset,
        EnsembleConfig(mode="soft"),
    )

    assert result["loss"] == pytest.approx(0.3)
    assert result["accuracy"] == 1.0
    assert result["top_1_accuracy"] == 1.0
    assert result["status"] == "completed"
    assert result["aggregation_metadata"]["mode"] == "soft"
