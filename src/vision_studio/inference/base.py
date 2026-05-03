from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import functional as F

Batch = tuple[Tensor, dict[str, Any]]


class Inference(ABC):
    def __init__(
        self,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)

    @abstractmethod
    def predict(
        self,
        model: Module,
        data_loader: Iterable[Batch],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(
        self,
        model: Module,
        batch: Batch,
    ) -> dict[str, Any]:
        raise NotImplementedError

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

    def save_predictions(
        self,
        path: str | Path,
        predictions: dict[str, Any],
    ) -> None:
        torch.save(predictions, str(path))

    def load_predictions(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        return torch.load(str(path), map_location="cpu")
