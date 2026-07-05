"""Tensor conversion and device helpers shared across pipelines."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional

NumpyLayout = Literal["auto", "hwc", "chw"]


def move_to_device(value: Any, device: torch.device | str) -> Any:
    """Recursively move tensors nested in common batch containers."""
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def input_to_tensor(input_item: Any, numpy_layout: NumpyLayout = "auto") -> Tensor:
    """Convert common image/input values into model-ready tensors."""
    if isinstance(input_item, Tensor):
        return input_item
    if isinstance(input_item, Image.Image):
        return functional.to_tensor(input_item)
    if isinstance(input_item, np.ndarray):
        return numpy_image_to_tensor(input_item, layout=numpy_layout)
    return torch.as_tensor(input_item)


def numpy_image_to_tensor(image: np.ndarray, layout: NumpyLayout = "auto") -> Tensor:
    """Convert a NumPy image-like array to a tensor, preserving CHW arrays."""
    if layout not in {"auto", "hwc", "chw"}:
        raise ValueError("numpy_layout must be one of: 'auto', 'hwc', or 'chw'")

    tensor = torch.as_tensor(image)
    if image.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif image.ndim == 3 and _should_permute_numpy_image(image, layout):
        tensor = tensor.permute(2, 0, 1).contiguous()

    if np.issubdtype(image.dtype, np.integer):
        tensor = tensor.float() / 255.0
    return tensor


def _should_permute_numpy_image(image: np.ndarray, layout: NumpyLayout) -> bool:
    if layout == "hwc":
        return True
    if layout == "chw":
        return False
    return not looks_channel_first(image)


def looks_channel_first(image: np.ndarray) -> bool:
    """Heuristically detect CHW-style image arrays."""
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
