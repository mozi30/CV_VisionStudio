"""Integration tests for training and evaluation workflows."""

import torch
from torch.optim import SGD

from vision_studio.evaluate.evaluator import LoopEvaluator
from vision_studio.models.base import BaseModel
from vision_studio.reporting import BaseReporter
from vision_studio.trainer.trainer import VisionTrainer
from vision_studio.types import TrainerSettings


class _Metrics:
    def reset(self) -> None:
        self.count = 0

    def update(self, predictions, targets, loss) -> None:
        self.count += 1

    def compute(self):
        return {"loss": 0.25}


class _RecordingReporter(BaseReporter):
    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.finished = 0
        self.logged: list[dict[str, float | int]] = []

    def start(self) -> None:
        self.started += 1

    def log(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        self.logged.append(metrics)

    def finish(self) -> None:
        self.finished += 1


class _Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight

    def postprocess(self, logits: torch.Tensor):
        return {"logits": logits}

    def compute_loss(self, logits: torch.Tensor, targets):
        return {"loss": logits.mean()}


def test_placeholder_training_evaluation_workflow() -> None:
    """Trainer should produce one evaluation result per completed epoch."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    trainer = VisionTrainer(
        optimizer=optimizer,
        evaluator=LoopEvaluator(_Metrics()),
        settings=TrainerSettings(
            epochs=2,
            checkpoint_path=None,
            best_checkpoint_count=0,
        ),
    )
    model = _Model()
    train_loader = [(torch.ones(2, 2), {"label": torch.ones(2, dtype=torch.long)})]
    val_loader = [(torch.ones(2, 2), {"label": torch.ones(2, dtype=torch.long)})]

    result = trainer.fit(model, train_loader, val_loader)

    assert len(result["history"]["train"]) == 2
    assert len(result["history"]["evaluation"]) == 2


def test_placeholder_migration_parity_workflow() -> None:
    """Supplying an Evaluator should keep evaluation outputs available."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    evaluator = LoopEvaluator(_Metrics())
    trainer = VisionTrainer(
        optimizer=optimizer,
        settings=TrainerSettings(
            epochs=1,
            checkpoint_path=None,
            best_checkpoint_count=0,
        ),
    )
    model = _Model()
    loader = [(torch.ones(2, 2), {"label": torch.ones(2, dtype=torch.long)})]

    result = trainer.fit(model, loader, loader, evaluator=evaluator)

    assert result["history"]["evaluation"][0]["loss"] == 0.25


def test_trainer_owns_reporting_scope_during_training_with_evaluation() -> None:
    """Trainer should own the reporting session for the full fit lifecycle."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    trainer_reporter = _RecordingReporter()
    evaluator_reporter = _RecordingReporter()
    evaluator = LoopEvaluator(_Metrics(), reporter=evaluator_reporter)
    trainer = VisionTrainer(
        optimizer=optimizer,
        evaluator=evaluator,
        reporter=trainer_reporter,
        settings=TrainerSettings(
            epochs=2,
            checkpoint_path=None,
            best_checkpoint_count=0,
        ),
    )
    model = _Model()
    loader = [(torch.ones(2, 2), {"label": torch.ones(2, dtype=torch.long)})]

    result = trainer.fit(model, loader, loader)

    assert len(result["history"]["evaluation"]) == 2
    assert trainer_reporter.started == 1
    assert trainer_reporter.finished == 1
    assert [entry["epoch"] for entry in trainer_reporter.logged] == [0, 0, 1, 1]
    assert evaluator_reporter.started == 0
    assert evaluator_reporter.finished == 0
    assert evaluator_reporter.logged == []
