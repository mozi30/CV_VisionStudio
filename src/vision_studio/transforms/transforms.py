"""Image transforms for preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2

from vision_studio.augmentation.base import Augmentation
from vision_studio.augmentation.utils import to_numpy


class ToTensor(Augmentation):
    """Convert PIL/numpy image to a PyTorch tensor with torchvision transforms.

    Converts image to float32 tensor with values in [0, 1].
    Target dict is passed through unchanged.
    """

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __call__(
        self,
        image: Any,
        target: dict[str, Any] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Convert image to tensor.

        Args:
            image: PIL image, numpy array, or tensor image
            target: Target dictionary

        Returns:
            Tuple of (tensor, target) where tensor is (C, H, W) with values in [0, 1]

        """
        tensor = self.transform(image).as_subclass(Tensor)
        if target is None:
            return tensor
        return tensor, target


class Normalize(Augmentation):
    """Normalize tensor using torchvision's implementation.

    Args:
        mean: Mean values for each channel. Default is ImageNet mean.
        std: Std values for each channel. Default is ImageNet std.

    """

    def __init__(
        self,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        super().__init__()
        # ImageNet normalization constants
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.transform = v2.Normalize(mean=self.mean, std=self.std)

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Any] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Normalize tensor.

        Args:
            image: Tensor of shape (C, H, W) with values in [0, 1]
            target: Target dictionary

        Returns:
            Tuple of (normalized_tensor, target)

        """
        normalized = self.transform(image)
        if target is None:
            return normalized
        return normalized, target


class ImageToArray(Augmentation):
    """Convert PIL image to numpy array.

    Useful for applying numpy-based augmentations.
    """

    def __call__(
        self,
        image: Any,
        target: dict[str, Any] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Convert image to numpy array.

        Args:
            image: PIL Image or numpy array
            target: Target dictionary

        Returns:
            Tuple of (numpy_array, target)

        """
        if isinstance(image, np.ndarray):
            array = image
        else:
            array = to_numpy(image)
        if target is None:
            return array
        return array, target
