from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.optim import Optimizer
from torchvision.transforms import functional as F

Batch = tuple[Tensor, dict[str, Any]]


class Trainer(ABC):
    def __init__(
        self,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.current_epoch = 0
        self.global_step = 0

    @abstractmethod
    def fit(
        self,
        model,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train_epoch(
        self,
        model,
        train_loader: Iterable[Batch],
    ) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        model,
        val_loader: Iterable[Batch],
    ) -> dict[str, float]:
        raise NotImplementedError

    def test(
        self,
        model,
        test_loader: Iterable[Batch],
    ) -> dict[str, float]:
        return self.validate(model, test_loader)

    def move_batch_to_device(self, batch: Batch) -> Batch:
        if self._is_collated_batch(batch):
            inputs, targets = batch
        else:
            inputs, targets = self._collate_raw_batch(batch)

        inputs = inputs.to(self.device)

        moved_targets: dict[str, Any] = {}
        for key, value in targets.items():
            if isinstance(value, Tensor):
                moved_targets[key] = value.to(self.device)
            else:
                moved_targets[key] = value

        return inputs, moved_targets

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
            if isinstance(input_item, Tensor):
                collated_inputs.append(input_item)
            elif isinstance(input_item, Image.Image):
                collated_inputs.append(F.to_tensor(input_item))
            else:
                collated_inputs.append(torch.as_tensor(input_item))

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
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, str(path))

    def load_checkpoint(
        self,
        path: str | Path,
        model,
    ) -> dict[str, Any]:
        checkpoint = torch.load(str(path), map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint
