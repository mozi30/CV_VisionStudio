from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Augmentation(ABC):
    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class Compose(Augmentation):
    def __init__(self, transforms: list[Augmentation]):
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomApply(Augmentation):
    def __init__(self, transform: Augmentation, p: float = 1.0):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.transform = transform
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if random.random() < self.p:
            return self.transform(image, target)
        return image, target


class OneOf(Augmentation):
    def __init__(
        self,
        transforms: list[Augmentation],
        probs: list[float] | None = None,
    ):
        if not transforms:
            raise ValueError("transforms must not be empty")
        if probs is not None and len(probs) != len(transforms):
            raise ValueError("probs must match transforms length")
        self.transforms = transforms
        self.probs = probs

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        transform = random.choices(self.transforms, weights=self.probs, k=1)[0]
        return transform(image, target)
