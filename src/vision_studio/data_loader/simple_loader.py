import random
from collections.abc import Iterator
from typing import Any, TypeVar

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from vision_studio.dataset import Dataset

from .base import DataLoader

T = TypeVar("T")


class SimpleDataLoader(DataLoader[tuple[Tensor, dict[str, Any]]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[Tensor, dict[str, Any]]]:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            samples = [self.dataset[i] for i in batch_indices]
            yield self._collate_batch(samples)

    def __len__(self) -> int:
        # number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate_batch(
        self,
        samples: list[tuple[Any, dict[str, Any]]],
    ) -> tuple[Tensor, dict[str, Any]]:
        inputs = [sample[0] for sample in samples]
        targets = [sample[1] for sample in samples]

        return self._collate_inputs(inputs), self._collate_targets(targets)

    def _collate_inputs(self, inputs: list[Any]) -> Tensor:
        tensors: list[Tensor] = []

        for input_item in inputs:
            if isinstance(input_item, Tensor):
                tensors.append(input_item)
            elif isinstance(input_item, Image.Image):
                tensors.append(F.to_tensor(input_item))
            else:
                tensors.append(torch.as_tensor(input_item))

        return torch.stack(tensors, dim=0)

    def _collate_targets(self, targets: list[dict[str, Any]]) -> dict[str, Any]:
        if not targets:
            return {}

        batched_targets: dict[str, Any] = {}

        for key in targets[0].keys():
            values = [target[key] for target in targets]

            if all(isinstance(value, Tensor) for value in values):
                batched_targets[key] = torch.stack(values, dim=0)
            elif all(isinstance(value, (int, float, bool)) for value in values):
                batched_targets[key] = torch.as_tensor(values)
            elif all(isinstance(value, str) for value in values):
                batched_targets[key] = values
            else:
                batched_targets[key] = values

        return batched_targets
