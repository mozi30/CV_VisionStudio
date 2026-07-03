from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import tv_tensors


class Augmentation(ABC):
    def __init__(self, p: float = 1.0):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        original_call = cls.__dict__.get("__call__")
        if original_call is None:
            return

        def wrapped(self, image: Any, target: dict[str, Any] | None = None):
            if random.random() < self.p:
                return original_call(self, image, target)
            if target is None:
                return image
            return image, target

        cls.__call__ = wrapped

    @abstractmethod
    def __call__(
        self,
        image: Any,
        target: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError


class Compose(Augmentation):
    def __init__(self, transforms: list[Callable[..., Any]], p: float = 1.0):
        super().__init__(p=p)
        self.transforms = transforms

    def __call__(
        self,
        image: Any,
        target: dict[str, Any] | None = None,
    ) -> Any:
        if target is None:
            for transform in self.transforms:
                image = transform(image)
            return image

        for transform in self.transforms:
            image, target = apply_transform(transform, image, target)
        return image, target


class OneOf(Augmentation):
    def __init__(
        self,
        transforms: list[Callable[..., Any]],
        probs: list[float] | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        if not transforms:
            raise ValueError("transforms must not be empty")
        if probs is not None and len(probs) != len(transforms):
            raise ValueError("probs must match transforms length")
        self.transforms = transforms
        self.probs = probs

    def __call__(
        self,
        image: Any,
        target: dict[str, Any] | None = None,
    ) -> Any:
        transform = random.choices(self.transforms, weights=self.probs, k=1)[0]
        if target is None:
            return transform(image)
        return apply_transform(transform, image, target)


class TorchVisionAugmentation(Augmentation):
    """Adapt a torchvision transform to Vision Studio's ``(image, target)`` API."""

    def __init__(self, transform: Callable[..., Any], p: float = 1.0):
        super().__init__(p=p)
        self.transform = transform

    def __call__(
        self,
        image: Any,
        target: dict[str, Any] | None = None,
    ) -> Any:
        if target is None:
            return self.transform(image)
        return apply_transform(self.transform, image, target)


def apply_transform(
    transform: Callable[..., Any],
    image: Any,
    target: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply a Vision Studio or torchvision transform to image/target pairs."""
    if isinstance(transform, Augmentation):
        result = transform(image, target)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, target

    tv_image = _image_to_torchvision(image)
    tv_target = _target_to_torchvision(tv_image, target)
    result = transform(tv_image, tv_target)

    if isinstance(result, tuple) and len(result) == 2:
        transformed_image, transformed_target = result
    else:
        transformed_image, transformed_target = result, tv_target

    return transformed_image, _target_from_torchvision(transformed_target)


def _target_to_torchvision(image: Any, target: dict[str, Any]) -> dict[str, Any]:
    converted = dict(target)
    boxes = converted.get("boxes")
    if boxes is not None and not isinstance(boxes, tv_tensors.BoundingBoxes):
        converted["boxes"] = tv_tensors.BoundingBoxes(
            _boxes_to_tensor(boxes),
            format="XYXY",
            canvas_size=_canvas_size(image),
        )
    return converted


def _image_to_torchvision(image: Any) -> Any:
    if isinstance(image, tv_tensors.Image | Image.Image):
        return image

    if isinstance(image, np.ndarray):
        tensor = torch.as_tensor(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in {1, 3, 4}:
            tensor = tensor.permute(2, 0, 1)
        return tv_tensors.Image(tensor)

    if isinstance(image, Tensor):
        return tv_tensors.Image(image)

    return image


def _target_from_torchvision(target: dict[str, Any]) -> dict[str, Any]:
    converted = dict(target)
    boxes = converted.get("boxes")
    if isinstance(boxes, tv_tensors.BoundingBoxes):
        converted["boxes"] = boxes.as_subclass(Tensor)
    return converted


def _boxes_to_tensor(boxes: Any) -> Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    return tensor.reshape(-1, 4)


def _canvas_size(image: Any) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width

    if isinstance(image, np.ndarray):
        return int(image.shape[0]), int(image.shape[1])

    if isinstance(image, Tensor):
        return int(image.shape[-2]), int(image.shape[-1])

    raise TypeError(
        "Cannot infer image size for torchvision bounding box transforms from "
        f"{type(image).__name__}."
    )
