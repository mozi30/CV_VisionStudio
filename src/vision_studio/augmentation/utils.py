from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def clamp_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def copy_target(target: dict[str, Any]) -> dict[str, Any]:
    return dict(target)


def image_size(image: np.ndarray) -> tuple[int, int]:
    h, w = image.shape[:2]
    return h, w


def is_grayscale(image: np.ndarray) -> bool:
    return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)


def ensure_3d(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[..., None]
    return image


def restore_dims(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    return image


def update_boxes_flip_horizontal(
    boxes: np.ndarray,
    width: int,
) -> np.ndarray:
    boxes = boxes.copy()
    x1 = boxes[:, 0].copy()
    x2 = boxes[:, 2].copy()
    boxes[:, 0] = width - x2
    boxes[:, 2] = width - x1
    return boxes


def update_boxes_flip_vertical(
    boxes: np.ndarray,
    height: int,
) -> np.ndarray:
    boxes = boxes.copy()
    y1 = boxes[:, 1].copy()
    y2 = boxes[:, 3].copy()
    boxes[:, 1] = height - y2
    boxes[:, 3] = height - y1
    return boxes
