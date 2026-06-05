"""Unit tests for evaluator exceptions and contracts."""

from collections.abc import Iterable
from typing import Any

import torch

from vision_studio.evaluate.evaluator import LoopEvaluator
from vision_studio.models.base import BaseModel
from vision_studio.types import (
    EvaluationError,
    InputSpec,
    LossOutput,
    OutputSpec,
    PostprocessOutput,
)

# ruff: noqa: D103


class _StubMetrics:
    def __init__(self) -> None:
        self.reset_called = False
        self.updates = 0

    def reset(self) -> None:
        self.reset_called = True

    def update(self, predictions, targets, loss) -> None:
        self.updates += 1

    def compute(self):
        return {"loss": 0.5}


class _StubModel(BaseModel):
    @property
    def input_spec(self) -> InputSpec:
        return {}

    @property
    def output_spec(self) -> OutputSpec:
        return {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def postprocess(self, logits: torch.Tensor) -> PostprocessOutput:
        return {"logits": logits}

    def compute_loss(self, logits: torch.Tensor, targets: dict[str, Any]) -> LossOutput:
        return {"loss": logits.mean()}


def test_evaluation_error_is_exception() -> None:
    """EvaluationError should remain an exception subtype."""
    assert issubclass(EvaluationError, Exception)


def test_loop_evaluator_resets_updates_and_computes_metrics() -> None:
    """LoopEvaluator should reset, update, and compute through metrics."""
    metrics = _StubMetrics()
    evaluator = LoopEvaluator(metrics=metrics)
    model = _StubModel()
    dataset: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]] = [
        (torch.ones(2, 2), {"label": torch.ones(2, dtype=torch.long)})
    ]

    result = evaluator.evaluate(model, dataset)

    assert metrics.reset_called is True
    assert metrics.updates == 1
    assert result == {"loss": 0.5}


def test_loop_evaluator_single_model_result_shape_remains_flat() -> None:
    metrics = _StubMetrics()
    evaluator = LoopEvaluator(metrics=metrics)
    model = _StubModel()
    dataset: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]] = [
        (torch.ones(2, 2), {"label": torch.ones(2, dtype=torch.long)})
    ]

    result = evaluator.evaluate(model, dataset)

    assert result == {"loss": 0.5}
    assert "status" not in result
    assert "aggregation_metadata" not in result


def test_prepare_ensemble_handoff_normalizes_output() -> None:
    metrics = _StubMetrics()
    evaluator = LoopEvaluator(metrics=metrics)

    payload = {
        "preds": torch.tensor([1, 0]),
        "metrics": {"loss": 0.1},
        "status": "completed",
        "aggregation_metadata": {"mode": "soft"},
        "failed_models": [2],
    }

    handoff = evaluator.prepare_ensemble_handoff(payload)

    assert torch.equal(handoff["preds"], torch.tensor([1, 0]))
    assert handoff["metrics"] == {"loss": 0.1}
    assert handoff["status"] == "completed"
    assert handoff["aggregation_metadata"] == {"mode": "soft"}
    assert handoff["failed_models"] == [2]
