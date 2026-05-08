"""Balanced data loader that samples classes with equal probability."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from vision_studio.dataset import Dataset

from .base import DataLoader


class BalancedDataLoader(DataLoader[tuple[Tensor, dict[str, Any]]]):
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
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle

		# Get class sample counts from dataset
		if not hasattr(dataset, "get_class_sample_counts"):
			raise AttributeError(
				f"Dataset must have get_class_sample_counts() method"
			)

		class_counts = dataset.get_class_sample_counts()
		if class_counts is None:
			raise ValueError(
				"Dataset.get_class_sample_counts() returned None"
			)

		self.class_counts = class_counts
		self._compute_sample_weights()

	def _compute_sample_weights(self) -> None:
		"""Compute weights for each sample to balance classes.

		Weight is inversely proportional to class frequency:
		weight[i] = 1 / (class_count[label[i]] * num_classes)
		"""
		num_classes = len(self.class_counts)
		self.sample_weights: list[float] = []

		for idx in range(len(self.dataset)):
			print(f"Computing weight for sample {idx+1}/{len(self.dataset)}", end="\r")
			_, target = self.dataset[idx]
			label = int(target["label"])
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

	def _collate_batch(
		self,
		samples: list[tuple[Any, dict[str, Any]]],
	) -> tuple[Tensor, dict[str, Any]]:
		"""Collate samples into a batch."""
		inputs = [sample[0] for sample in samples]
		targets = [sample[1] for sample in samples]

		return self._collate_inputs(inputs), self._collate_targets(targets)

	def _collate_inputs(self, inputs: list[Any]) -> Tensor:
		"""Stack input tensors."""
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
		"""Stack target tensors."""
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
