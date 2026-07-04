"""Unit tests for trainer delegation and configuration behavior."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import SGD

from vision_studio.trainer.base import Trainer
from vision_studio.types import ConfigurationError, TrainerSettings


class _StubTrainer(Trainer):
    def fit(self, model, train_loader, val_loader=None):  # type: ignore[override]
        return {}

    def train_epoch(self, model, train_loader):  # type: ignore[override]
        return {"loss": 0.0}


def test_trainer_settings_reject_zero_epochs() -> None:
    """Trainer settings should preserve explicitly configured epochs."""
    settings = TrainerSettings(epochs=0)
    assert settings.epochs == 0


def test_configuration_error_is_exception() -> None:
    """ConfigurationError should remain an exception subtype."""
    assert issubclass(ConfigurationError, Exception)


def test_trainer_settings_warn_when_checkpoint_path_missing(tmp_path: Path) -> None:
    """Trainer should warn when checkpoint saving is not configured."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    trainer = _StubTrainer(
        optimizer=optimizer,
        settings=TrainerSettings(checkpoint_path=None),
    )
    trainer.validate_settings()
    assert trainer.warnings


def test_trainer_settings_raise_for_missing_checkpoint_path(tmp_path: Path) -> None:
    """Trainer should fail when a configured checkpoint path does not exist."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    missing = tmp_path / "missing"
    trainer = _StubTrainer(
        optimizer=optimizer,
        settings=TrainerSettings(checkpoint_path=missing),
    )
    with pytest.raises(ConfigurationError):
        trainer.validate_settings()


def test_trainer_settings_require_checkpoint_monitor_for_best_checkpoints(
    tmp_path: Path,
) -> None:
    """Best checkpoint saving should require an explicit monitor metric."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    trainer = _StubTrainer(
        optimizer=optimizer,
        settings=TrainerSettings(
            checkpoint_path=tmp_path,
            best_checkpoint_count=3,
            checkpoint_monitor=None,
        ),
    )
    with pytest.raises(ConfigurationError):
        trainer.validate_settings()


def test_trainer_checkpoint_enabled_matches_checkpoint_path_configuration(
    tmp_path: Path,
) -> None:
    """Checkpoint enablement should track checkpoint path configuration."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)

    disabled_trainer = _StubTrainer(
        optimizer=optimizer,
        settings=TrainerSettings(checkpoint_path=None),
    )
    assert disabled_trainer.checkpoint_enabled() is False

    enabled_trainer = _StubTrainer(
        optimizer=optimizer,
        settings=TrainerSettings(
            checkpoint_path=tmp_path,
            checkpoint_monitor="loss",
        ),
    )
    assert enabled_trainer.checkpoint_enabled() is True


def test_trainer_collates_numpy_hwc_images_to_chw_float_tensors() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    trainer = _StubTrainer(optimizer=optimizer)
    raw_batch = [
        (
            np.full((4, 5, 3), 255, dtype=np.uint8),
            {"label": 0},
        )
    ]

    inputs, targets = trainer.move_batch_to_device(raw_batch)

    assert inputs.shape == (1, 3, 4, 5)
    assert inputs.dtype == torch.float32
    assert torch.equal(inputs, torch.ones((1, 3, 4, 5)))
    assert torch.equal(targets["label"], torch.tensor([0]))


def test_trainer_moves_nested_target_tensors_to_device() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.1)
    trainer = _StubTrainer(optimizer=optimizer, device="cpu")
    batch = (
        torch.ones(1, 3, 4, 4),
        {
            "detections": [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "labels": torch.tensor([1]),
                }
            ],
            "metadata": ("sample-a", torch.tensor([2])),
        },
    )

    _, targets = trainer.move_batch_to_device(batch)

    assert targets["detections"][0]["boxes"].device.type == "cpu"
    assert targets["detections"][0]["labels"].device.type == "cpu"
    assert targets["metadata"][1].device.type == "cpu"
