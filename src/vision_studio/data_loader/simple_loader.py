import random
from collections.abc import Iterator
from typing import Any, TypeVar

import numpy as np
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
        dataset_percentage_per_epoch: int = 100,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        if not 1 <= dataset_percentage_per_epoch <= 100:
            raise ValueError("dataset_percentage_per_epoch must be between 1 and 100.")
        self.dataset = dataset
        self.dataset_percentage_per_epoch = dataset_percentage_per_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[Tensor, dict[str, Any]]]:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        if self.dataset_percentage_per_epoch < 100:
            total = max(
                1,
                int(len(indices) * self.dataset_percentage_per_epoch / 100),
            )
            indices = indices[:total]

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            samples = [self.dataset[i] for i in batch_indices]
            yield self._collate_batch(samples)

    def __len__(self) -> int:
        # number of batches
        total = max(
            1,
            int(len(self.dataset) * self.dataset_percentage_per_epoch / 100),
        )
        return (total + self.batch_size - 1) // self.batch_size

    def _collate_batch(self, samples: list[Any]) -> tuple[Tensor, dict[str, Any]]:
        inputs = []
        targets = []

        for sample in samples:
            input_item, target = self._split_sample(sample)
            inputs.append(input_item)
            targets.append(target)

        return self._collate_inputs(inputs), self._collate_targets(targets)

    def _split_sample(self, sample: Any) -> tuple[Any, Any]:
        if isinstance(sample, dict):
            if "image" in sample:
                image = sample["image"]
            elif "input" in sample:
                image = sample["input"]
            elif "inputs" in sample:
                image = sample["inputs"]
            else:
                raise KeyError(
                    "dict samples must contain one of: 'image', 'input', or 'inputs'"
                )
            input_keys = {"image", "input", "inputs"}
            target = sample.get(
                "target", {k: v for k, v in sample.items() if k not in input_keys}
            )
            return image, target

        if isinstance(sample, (tuple, list)):
            if len(sample) == 2:
                return sample[0], sample[1]
            if len(sample) == 1:
                return sample[0], {}

        return sample, {}

    def _collate_inputs(self, inputs: list[Any]) -> Tensor:
        tensors: list[Tensor] = []

        for input_item in inputs:
            tensors.append(self._input_to_tensor(input_item))

        return torch.stack(tensors, dim=0)

    def _input_to_tensor(self, input_item: Any) -> Tensor:
        if isinstance(input_item, Tensor):
            return input_item
        if isinstance(input_item, Image.Image):
            return F.to_tensor(input_item)
        if isinstance(input_item, np.ndarray):
            return self._numpy_image_to_tensor(input_item)
        return torch.as_tensor(input_item)

    def _numpy_image_to_tensor(self, image: np.ndarray) -> Tensor:
        tensor = torch.as_tensor(image)
        if image.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif image.ndim == 3 and not self._looks_channel_first(image):
            tensor = tensor.permute(2, 0, 1).contiguous()

        if np.issubdtype(image.dtype, np.integer):
            tensor = tensor.float() / 255.0
        return tensor

    def _looks_channel_first(self, image: np.ndarray) -> bool:
        if image.ndim != 3:
            return False

        channel_sizes = {1, 3, 4}
        first_is_channel = image.shape[0] in channel_sizes
        last_is_channel = image.shape[-1] in channel_sizes

        if first_is_channel and not last_is_channel:
            return True
        if last_is_channel and not first_is_channel:
            return False
        if first_is_channel and last_is_channel:
            return image.shape[1] in channel_sizes

        return False

    def _collate_targets(self, targets: list[Any]) -> dict[str, Any]:
        if not targets:
            return {}

        normalized_targets = [self._normalize_target(target) for target in targets]
        batched_targets: dict[str, Any] = {}

        keys = set().union(*(target.keys() for target in normalized_targets))
        for key in keys:
            values = [target.get(key) for target in normalized_targets]

            if all(isinstance(value, Tensor) for value in values) and self._can_stack(
                values
            ):
                batched_targets[key] = torch.stack(values, dim=0)
            elif all(isinstance(value, (int, float, bool)) for value in values):
                batched_targets[key] = torch.as_tensor(values)
            elif all(isinstance(value, str) for value in values):
                batched_targets[key] = values
            else:
                batched_targets[key] = values

        return batched_targets

    def _can_stack(self, values: list[Any]) -> bool:
        if not values or not all(isinstance(value, Tensor) for value in values):
            return False

        first_shape = values[0].shape
        return all(value.shape == first_shape for value in values)

    def _normalize_target(self, target: Any) -> dict[str, Any]:
        if target is None:
            return {}
        if isinstance(target, dict):
            return target
        if isinstance(target, Tensor) and target.ndim == 0:
            return {"label": target}
        if isinstance(target, (int, float, bool)):
            return {"label": target}
        return {"target": target}
