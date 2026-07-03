"""Unit tests for ensemble dataset evaluation."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torchvision.transforms import v2

from vision_studio.augmentation import Resize
from vision_studio.evaluate import (
    ClassificationEvaluationMetrics,
    EnsembleMember,
    LoopEvaluator,
)
from vision_studio.inference.simple import EnsembleConfig
from vision_studio.models.base import BaseModel
from vision_studio.reporting import BaseReporter
from vision_studio.types import (
    ConfigurationError,
    LossOutput,
    PostprocessOutput,
)


class _ConstantModel(BaseModel):
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


class _FailingModel(_ConstantModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("model failed")


class _ShapeCheckingModel(_ConstantModel):
    def __init__(self, expected_size: tuple[int, int], logits: torch.Tensor):
        super().__init__(logits=logits, loss=0.2)
        self.expected_size = expected_size
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(tuple(inputs.shape))
        if tuple(inputs.shape[-2:]) != self.expected_size:
            raise RuntimeError(
                f"expected image size {self.expected_size}, got {tuple(inputs.shape[-2:])}"
            )
        return super().forward(inputs)


class _InputRangeCheckingModel(_ConstantModel):
    def __init__(self, logits: torch.Tensor):
        super().__init__(logits=logits, loss=0.2)
        self.seen_min: float | None = None
        self.seen_max: float | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_min = float(inputs.min())
        self.seen_max = float(inputs.max())
        return super().forward(inputs)


class _RecordingReporter(BaseReporter):
    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.finished = 0
        self.logged: list[dict[str, Any]] = []

    def start(self) -> None:
        self.started += 1

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.logged.append(metrics)

    def finish(self) -> None:
        self.finished += 1


def _dataset(batch_size: int = 2):
    return [
        (
            torch.randn(batch_size, 3, 8, 8),
            {"label": torch.tensor([0, 1][:batch_size], dtype=torch.long)},
        )
    ]


def test_evaluate_ensemble_default_soft_voting_metrics_and_metadata() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _ConstantModel(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), loss=0.2),
        _ConstantModel(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), loss=0.4),
    ]

    result = evaluator.evaluate_ensemble(models, _dataset())

    assert result["loss"] == pytest.approx(0.3)
    assert result["accuracy"] == 1.0
    assert result["status"] == "completed"
    assert result["failed_models"] == []
    assert result["aggregation_metadata"] == {
        "mode": "soft",
        "tie_policy": "index-priority",
        "failure_policy": "fail-fast",
        "failed_models": [],
        "model_count": 2,
        "successful_model_count": 2,
    }


def test_evaluate_ensemble_weighted_mode_and_weight_validation() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.2),
        _ConstantModel(torch.tensor([[0.1, 0.9]]), loss=0.4),
    ]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([1])})]

    result = evaluator.evaluate_ensemble(
        models,
        batch,
        EnsembleConfig(mode="weighted", weights=[0.1, 0.9]),
    )

    assert result["accuracy"] == 1.0
    assert result["aggregation_metadata"]["mode"] == "weighted"

    with pytest.raises(ConfigurationError, match="weights length must match"):
        evaluator.evaluate_ensemble(
            models,
            batch,
            EnsembleConfig(mode="weighted", weights=[1.0]),
        )


def test_evaluate_ensemble_loss_is_mean_successful_per_model_loss() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.25),
        _ConstantModel(torch.tensor([[0.8, 0.2]]), loss=0.75),
    ]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble(models, batch)

    assert result["loss"] == pytest.approx(0.5)


def test_evaluate_ensemble_hard_voting_tie_policy_metadata_and_labels() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.2),
        _ConstantModel(torch.tensor([[0.1, 0.9]]), loss=0.2),
    ]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble(
        models,
        batch,
        EnsembleConfig(mode="hard", tie_policy="index-priority"),
    )

    assert result["accuracy"] == 1.0
    assert result["aggregation_metadata"]["mode"] == "hard"
    assert result["aggregation_metadata"]["tie_policy"] == "index-priority"


def test_evaluate_ensemble_continue_with_warning_records_failed_models() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.25),
        _FailingModel(torch.tensor([[0.1, 0.9]]), loss=0.75),
    ]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble(
        models,
        batch,
        EnsembleConfig(failure_policy="continue-with-warning"),
    )

    assert result["status"] == "completed_with_warnings"
    assert result["failed_models"] == [1]
    assert result["aggregation_metadata"]["failed_models"] == [1]
    assert result["aggregation_metadata"]["successful_model_count"] == 1
    assert result["loss"] == pytest.approx(0.25)


def test_evaluate_ensemble_all_model_failure_raises() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _FailingModel(torch.tensor([[0.9, 0.1]]), loss=0.25),
        _FailingModel(torch.tensor([[0.1, 0.9]]), loss=0.75),
    ]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    with pytest.raises(ConfigurationError, match="No successful model outputs"):
        evaluator.evaluate_ensemble(
            models,
            batch,
            EnsembleConfig(failure_policy="continue-with-warning"),
        )


def test_evaluate_ensemble_incompatible_class_counts_raise() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=3, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    models = [
        _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.25),
        _ConstantModel(torch.tensor([[0.1, 0.2, 0.7]]), loss=0.75),
    ]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    with pytest.raises(ConfigurationError, match="identical output class count"):
        evaluator.evaluate_ensemble(models, batch)


def test_evaluate_ensemble_restores_training_states_after_success() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    training_model = _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.25)
    eval_model = _ConstantModel(torch.tensor([[0.8, 0.2]]), loss=0.25)
    training_model.train()
    eval_model.eval()
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    evaluator.evaluate_ensemble([training_model, eval_model], batch)

    assert training_model.training is True
    assert eval_model.training is False


def test_evaluate_ensemble_restores_training_states_after_failure() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    training_model = _ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.25)
    failing_model = _FailingModel(torch.tensor([[0.8, 0.2]]), loss=0.25)
    training_model.train()
    failing_model.eval()
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    with pytest.raises(RuntimeError, match="model failed"):
        evaluator.evaluate_ensemble([training_model, failing_model], batch)

    assert training_model.training is True
    assert failing_model.training is False


def test_evaluate_ensemble_reports_with_configured_reporter() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    reporter = _RecordingReporter()
    evaluator = LoopEvaluator(metrics=metrics, reporter=reporter)
    models = [_ConstantModel(torch.tensor([[0.9, 0.1]]), loss=0.25)]
    batch = [(torch.randn(1, 3, 8, 8), {"label": torch.tensor([0])})]

    evaluator.evaluate_ensemble(models, batch)

    assert reporter.started == 1
    assert reporter.finished == 1
    assert reporter.logged == [{"evaluation/loss": pytest.approx(0.25)}]


def test_evaluate_ensemble_members_applies_member_augmentations() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    small_model = _ShapeCheckingModel(
        expected_size=(8, 8),
        logits=torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
    )
    large_model = _ShapeCheckingModel(
        expected_size=(16, 16),
        logits=torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
    )
    batch = [
        (
            torch.rand(2, 3, 12, 12),
            {"label": torch.tensor([0, 1], dtype=torch.long)},
        )
    ]

    result = evaluator.evaluate_ensemble_members(
        members=[
            EnsembleMember(
                model=small_model,
                augmentation=Resize(8, 8),
                name="small",
            ),
            EnsembleMember(
                model=large_model,
                augmentation=Resize(16, 16),
                name="large",
            ),
        ],
        dataset=batch,
        config=EnsembleConfig(mode="soft"),
    )

    assert result["accuracy"] == 1.0
    assert small_model.seen_shapes == [(2, 3, 8, 8)]
    assert large_model.seen_shapes == [(2, 3, 16, 16)]


def test_evaluate_ensemble_accepts_ensemble_members_for_augmentation_path() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    small_model = _ShapeCheckingModel(
        expected_size=(8, 8),
        logits=torch.tensor([[0.9, 0.1]]),
    )
    large_model = _ShapeCheckingModel(
        expected_size=(16, 16),
        logits=torch.tensor([[0.8, 0.2]]),
    )
    batch = [(torch.rand(1, 3, 12, 12), {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble(
        models=[
            EnsembleMember(model=small_model, augmentation=Resize(8, 8)),
            EnsembleMember(model=large_model, augmentation=Resize(16, 16)),
        ],
        dataset=batch,
        config=EnsembleConfig(mode="soft"),
    )

    assert result["accuracy"] == 1.0
    assert small_model.seen_shapes == [(1, 3, 8, 8)]
    assert large_model.seen_shapes == [(1, 3, 16, 16)]


def test_evaluate_ensemble_members_accepts_torchvision_augmentation() -> None:
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    model = _ShapeCheckingModel(
        expected_size=(8, 8),
        logits=torch.tensor([[0.9, 0.1]]),
    )
    batch = [(torch.rand(1, 3, 12, 12), {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble_members(
        members=[
            EnsembleMember(
                model=model,
                augmentation=v2.Resize((8, 8)),
            )
        ],
        dataset=batch,
        config=EnsembleConfig(mode="soft"),
    )

    assert result["accuracy"] == 1.0
    assert model.seen_shapes == [(1, 3, 8, 8)]


def test_evaluate_ensemble_members_preserves_normalized_tensors_for_torchvision() -> (
    None
):
    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    model = _InputRangeCheckingModel(logits=torch.tensor([[0.9, 0.1]]))
    normalized_input = torch.linspace(-2.0, 2.0, steps=3 * 8 * 8).reshape(1, 3, 8, 8)
    batch = [(normalized_input, {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble_members(
        members=[
            EnsembleMember(
                model=model,
                augmentation=v2.Identity(),
            )
        ],
        dataset=batch,
        config=EnsembleConfig(mode="soft"),
    )

    assert result["accuracy"] == 1.0
    assert model.seen_min == pytest.approx(-2.0)
    assert model.seen_max == pytest.approx(2.0)


def test_evaluate_ensemble_members_treats_augmentation_failure_as_model_failure() -> (
    None
):
    class _FailingAugmentation(Resize):
        def __call__(self, image, target):
            raise RuntimeError("augmentation failed")

    metrics = ClassificationEvaluationMetrics(num_classes=2, topk=(1,))
    evaluator = LoopEvaluator(metrics=metrics)
    good_model = _ShapeCheckingModel(
        expected_size=(8, 8),
        logits=torch.tensor([[0.9, 0.1]]),
    )
    skipped_model = _ShapeCheckingModel(
        expected_size=(16, 16),
        logits=torch.tensor([[0.1, 0.9]]),
    )
    batch = [(torch.rand(1, 3, 12, 12), {"label": torch.tensor([0])})]

    result = evaluator.evaluate_ensemble_members(
        members=[
            EnsembleMember(good_model, Resize(8, 8), "good"),
            EnsembleMember(skipped_model, _FailingAugmentation(16, 16), "bad"),
        ],
        dataset=batch,
        config=EnsembleConfig(failure_policy="continue-with-warning"),
    )

    assert result["status"] == "completed_with_warnings"
    assert result["failed_models"] == [1]
    assert good_model.seen_shapes == [(1, 3, 8, 8)]
    assert skipped_model.seen_shapes == []
