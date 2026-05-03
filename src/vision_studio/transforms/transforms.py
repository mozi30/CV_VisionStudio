"""Image transforms for preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor
from torchvision.transforms import functional as F

from vision_studio.augmentation.base import Augmentation
from vision_studio.augmentation.utils import to_numpy, to_pil


class ToTensor(Augmentation):
    """Convert PIL/numpy image to PyTorch tensor.

    Converts image to float32 tensor with values in [0, 1].
    Target dict is passed through unchanged.
    """

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Convert image to tensor.

        Args:
            image: Image as numpy array (H, W, C) with values in [0, 255]
            target: Target dictionary

        Returns:
            Tuple of (tensor, target) where tensor is (C, H, W) with values in [0, 1]

        """
        pil = to_pil(image)
        tensor = F.to_tensor(pil)
        return tensor, target


class Normalize(Augmentation):
    """Normalize tensor using ImageNet statistics or custom mean/std.

    Args:
        mean: Mean values for each channel. Default is ImageNet mean.
        std: Std values for each channel. Default is ImageNet std.

    """

    def __init__(
        self,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        # ImageNet normalization constants
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Normalize tensor.

        Args:
            image: Tensor of shape (C, H, W) with values in [0, 1]
            target: Target dictionary

        Returns:
            Tuple of (normalized_tensor, target)

        """
        return F.normalize(image, self.mean, self.std), target


class ImageToArray(Augmentation):
    """Convert PIL image to numpy array.

    Useful for applying numpy-based augmentations.
    """

    def __call__(
        self,
        image: Any,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Convert image to numpy array.

        Args:
            image: PIL Image or numpy array
            target: Target dictionary

        Returns:
            Tuple of (numpy_array, target)

        """
        if isinstance(image, np.ndarray):
            return image, target
        return to_numpy(image), target
