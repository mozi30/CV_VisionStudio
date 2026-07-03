"""Balanced data loader that samples classes with equal probability."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from torch import Tensor

from vision_studio.dataset import Dataset

from .simple_loader import SimpleDataLoader


class BalancedDataLoader(SimpleDataLoader):
    """Data loader that ensures balanced sampling across all classes.

    Each class has equal probability of being sampled regardless of
    the class distribution in the dataset. Useful for imbalanced datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
    ):
        """Initialize balanced data loader.

        Args:
                dataset: Dataset with get_class_sample_counts() method.
                batch_size: Number of samples per batch.
                shuffle: Whether to shuffle samples within each epoch.

        """
        super().__init__(
            dataset=dataset,
            dataset_percentage_per_epoch=100,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        class_counts = self._get_class_sample_counts()
        if not class_counts:
            raise ValueError("Could not infer class sample counts from dataset labels")

        self.class_counts = class_counts
        self._compute_sample_weights()

    def _get_class_sample_counts(self) -> dict[int, int]:
        get_counts = getattr(self.dataset, "get_class_sample_counts", None)
        if callable(get_counts):
            class_counts = get_counts()
            if class_counts is not None:
                return dict(class_counts)

        counts: dict[int, int] = {}
        for idx in range(len(self.dataset)):
            _, target = self._split_sample(self.dataset[idx])
            label = self._extract_label(target)
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _extract_label(self, target: Any) -> int:
        normalized = self._normalize_target(target)
        if "label" not in normalized:
            raise KeyError(
                "BalancedDataLoader requires each dataset sample to provide a label"
            )
        label = normalized["label"]
        if hasattr(label, "item"):
            label = label.item()
        return int(label)

    def _compute_sample_weights(self) -> None:
        """Compute weights for each sample to balance classes.

        Weight is inversely proportional to class frequency:
        weight[i] = 1 / (class_count[label[i]] * num_classes)
        """
        num_classes = len(self.class_counts)
        self.sample_weights: list[float] = []

        for idx in range(len(self.dataset)):
            _, target = self._split_sample(self.dataset[idx])
            label = self._extract_label(target)
            class_count = self.class_counts.get(label, 1)
            # Weight is inversely proportional to class frequency
            weight = 1.0 / (class_count * num_classes)
            self.sample_weights.append(weight)

    def __iter__(self) -> Iterator[tuple[Tensor, dict[str, Any]]]:
        """Yield batches with balanced class representation."""
        # Sample indices with replacement using computed weights
        # This ensures each class has equal expected representation
        num_samples = len(self.dataset)
        sampled_indices = random.choices(
            range(num_samples),
            weights=self.sample_weights,
            k=num_samples,
        )

        if self.shuffle:
            random.shuffle(sampled_indices)

        # Yield batches
        for start in range(0, len(sampled_indices), self.batch_size):
            batch_indices = sampled_indices[start : start + self.batch_size]
            samples = [self.dataset[i] for i in batch_indices]
            yield self._collate_batch(samples)

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
