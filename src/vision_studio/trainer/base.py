"""Base training abstractions and shared trainer helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from vision_studio.types import (
    CheckpointError,
    CheckpointInfo,
    ConfigurationError,
    EvaluatorOutput,
    TrainerSettings,
)
from vision_studio.utils import input_to_tensor, move_to_device

Batch = tuple[Tensor, dict[str, Any]]


class Trainer(ABC):
    """Base training abstraction shared by concrete Trainer implementations."""

    def __init__(
        self,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
        settings: TrainerSettings | None = None,
    ) -> None:
        """Initialize shared Trainer state."""
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.settings = settings or TrainerSettings()
        self.current_epoch = 0
        self.global_step = 0
        self.warnings: list[str] = []

    @abstractmethod
    def fit(
        self,
        model,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
    ) -> dict[str, Any]:
        """Run the full training workflow for a model and training dataset."""
        raise NotImplementedError

    @abstractmethod
    def train_epoch(
        self,
        model,
        train_loader: Iterable[Batch],
    ) -> EvaluatorOutput:
        """Run one training epoch and return epoch-level training metrics."""
        raise NotImplementedError

    def move_batch_to_device(self, batch: Batch) -> Batch:
        """Normalize a batch and move tensors to the configured device."""
        if self._is_collated_batch(batch):
            inputs, targets = batch
        else:
            inputs, targets = self._collate_raw_batch(batch)

        inputs = inputs.to(self.device)

        return inputs, move_to_device(targets, self.device)

    def _is_collated_batch(self, batch: Any) -> bool:
        return (
            isinstance(batch, tuple)
            and len(batch) == 2
            and isinstance(batch[1], dict)
            and isinstance(batch[0], Tensor)
        )

    def _collate_raw_batch(self, batch: Any) -> Batch:
        samples = list(batch)
        if not samples:
            raise ValueError("Batch must not be empty.")

        inputs = [sample[0] for sample in samples]
        targets = [sample[1] for sample in samples]

        collated_inputs: list[Tensor] = []
        for input_item in inputs:
            collated_inputs.append(input_to_tensor(input_item))

        batched_inputs = torch.stack(collated_inputs, dim=0)
        batched_targets: dict[str, Any] = {}

        for key in targets[0].keys():
            values = [target[key] for target in targets]
            if all(isinstance(value, Tensor) for value in values):
                batched_targets[key] = torch.stack(values, dim=0)
            elif all(isinstance(value, (int, float, bool)) for value in values):
                batched_targets[key] = torch.as_tensor(values)
            else:
                batched_targets[key] = values

        return batched_inputs, batched_targets

    def save_checkpoint(
        self,
        path: str | Path,
        model,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint data for the current Trainer state."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, str(path))

    def validate_settings(self) -> None:
        """Validate Trainer settings before a training run begins."""
        if self.settings.epochs <= 0:
            raise ConfigurationError("epochs must be greater than zero")
        if self.settings.best_checkpoint_count < 0:
            raise ConfigurationError("best_checkpoint_count must not be negative")
        if self.settings.early_stopping_patience < 0:
            raise ConfigurationError("early_stopping_patience must not be negative")
        if self.settings.checkpoint_mode not in {"min", "max"}:
            raise ConfigurationError("checkpoint_mode must be 'min' or 'max'")
        if self.settings.checkpoint_path is None:
            self.warn("Checkpoint path not configured; no checkpoints will be saved.")
            return
        if not self.settings.checkpoint_path.exists():
            raise ConfigurationError("Configured checkpoint path does not exist")
        if not self.settings.checkpoint_path.is_dir():
            raise ConfigurationError("Configured checkpoint path must be a directory")
        if (
            self.settings.best_checkpoint_count > 0
            and not self.settings.checkpoint_monitor
        ):
            raise ConfigurationError(
                "checkpoint_monitor must be configured for best checkpoint saving"
            )

    def checkpoint_enabled(self) -> bool:
        """Return whether checkpoint saving is configured for this Trainer."""
        return self.settings.checkpoint_path is not None

    def warn(self, message: str) -> None:
        """Record a non-fatal Trainer warning message."""
        self.warnings.append(message)

    def build_checkpoint_info(
        self,
        *,
        path: Path,
        kind: str,
        monitor_metric: str | None = None,
        monitor_value: float | None = None,
        rank: int | None = None,
    ) -> CheckpointInfo:
        """Build structured metadata for a saved checkpoint."""
        return CheckpointInfo(
            path=path,
            kind=kind,
            epoch=self.current_epoch,
            global_step=self.global_step,
            monitor_metric=monitor_metric,
            monitor_value=monitor_value,
            rank=rank,
        )

    def checkpoint_path_for(self, filename: str) -> Path:
        """Resolve a checkpoint filename under the configured checkpoint path."""
        if self.settings.checkpoint_path is None:
            raise CheckpointError("Checkpoint path is not configured")
        return self.settings.checkpoint_path / filename

    def load_checkpoint(
        self,
        path: str | Path,
        model,
    ) -> dict[str, Any]:
        """Load checkpoint state into the model and optimizer."""
        checkpoint = torch.load(str(path), map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._move_optimizer_state_to_device()
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint

    def _move_optimizer_state_to_device(self) -> None:
        """Move optimizer state tensors to the trainer device.

        This prevents device mismatch errors after loading checkpoints.
        """
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(self.device)
