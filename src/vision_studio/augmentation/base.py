from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


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

        def wrapped(self, image: np.ndarray, target: dict[str, Any] | None = None):
            if random.random() < self.p:
                return original_call(self, image, target)
            if target is None:
                return image
            return image, target

        cls.__call__ = wrapped

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class Compose(Augmentation):
    def __init__(self, transforms: list[Augmentation], p: float = 1.0):
        super().__init__(p=p)
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class OneOf(Augmentation):
    def __init__(
        self,
        transforms: list[Augmentation],
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
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        transform = random.choices(self.transforms, weights=self.probs, k=1)[0]
        return transform(image, target)
