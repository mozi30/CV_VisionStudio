"""Visualization helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any
from vision_studio.dataset import Dataset

import numpy as np
import torch
from matplotlib import pyplot as plt


def _to_display_array(image: Any) -> np.ndarray:
	if isinstance(image, torch.Tensor):
		arr = image.detach().cpu().float().numpy()
		if arr.ndim == 3 and arr.shape[0] in (1, 3):
			arr = np.transpose(arr, (1, 2, 0))
		if arr.ndim == 3 and arr.shape[2] == 1:
			arr = arr[:, :, 0]
	else:
		arr = np.array(image)

	if arr.dtype != np.uint8:
		arr = arr.astype(np.float32)
		if arr.max() > 1.0 or arr.min() < 0.0:
			min_val = float(arr.min())
			max_val = float(arr.max())
			if max_val > min_val:
				arr = (arr - min_val) / (max_val - min_val)
		arr = np.clip(arr, 0.0, 1.0)

	return arr


def show_dataset_images(
	dataset: Dataset,
	n: int = 8,
	cols: int = 4,
	shuffle: bool = False,
	seed: int = 0,
) -> None:
	"""Show a grid of images from a dataset.

	Args:
		dataset: Dataset supporting __len__ and __getitem__.
		n: Number of images to display.
		cols: Number of columns in the grid.
		shuffle: Whether to shuffle indices before sampling.
		seed: RNG seed when shuffle is True.
	"""
	if n <= 0:
		raise ValueError("n must be > 0")
	if cols <= 0:
		raise ValueError("cols must be > 0")

	total = len(dataset)
	if total == 0:
		raise ValueError("dataset is empty")

	count = min(n, total)
	indices = list(range(total))
	if shuffle:
		rng = np.random.default_rng(seed)
		rng.shuffle(indices)
	indices = indices[:count]

	rows = (count + cols - 1) // cols
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
	axes = np.array(axes).reshape(-1)

	for ax in axes[count:]:
		ax.axis("off")

	for ax, idx in zip(axes, indices):
		sample = dataset[idx]
		image = sample
		if isinstance(sample, (tuple, list)) and sample:
			image = sample[0]
		elif isinstance(sample, dict):
			image = sample.get("image", next(iter(sample.values())))

		arr = _to_display_array(image)
		ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
		ax.axis("off")

	plt.tight_layout()
	plt.show()


def show_dataset_label_distribution(
	dataset: Dataset,
	class_names: list[str] | dict[int, str] | None = None,
	title: str = "Dataset label distribution",
) -> None:
	"""Show a bar chart of label counts for a dataset.

	Args:
		dataset: Dataset supporting __len__ and __getitem__.
		class_names: Optional label names, either as an indexable list or a mapping
			from class id to class name.
		title: Plot title.
	"""
	total = len(dataset)
	if total == 0:
		raise ValueError("dataset is empty")

	counts: Counter[int] = Counter()
	for idx in range(total):
		sample = dataset[idx]
		if isinstance(sample, (tuple, list)) and len(sample) > 1:
			target = sample[1]
		elif isinstance(sample, dict):
			target = sample.get("target", sample)
		else:
			raise TypeError(
				"dataset samples must be tuples/lists with targets or dicts containing labels"
			)

		if not isinstance(target, dict) or "label" not in target:
			raise KeyError("dataset targets must contain a 'label' entry")

		counts[int(target["label"])] += 1

	labels = sorted(counts)
	values = [counts[label] for label in labels]

	def _resolve_label_name(label: int) -> str:
		if isinstance(class_names, dict):
			return str(class_names.get(label, label))
		if isinstance(class_names, list) and 0 <= label < len(class_names):
			return str(class_names[label])
		if hasattr(dataset, "get_class_name"):
			try:
				return str(dataset.get_class_name(label))
			except Exception:
				pass
		return str(label)

	if not labels:
		raise ValueError("dataset does not contain any labels")

	x_labels = [_resolve_label_name(label) for label in labels]

	fig_width = max(6, len(labels) * 0.75)
	fig, ax = plt.subplots(figsize=(fig_width, 4.5))
	bars = ax.bar(x_labels, values, color="#4C78A8")
	ax.set_title(title)
	ax.set_xlabel("Class")
	ax.set_ylabel("Count")
	ax.tick_params(axis="x", rotation=45)
	ax.bar_label(bars, padding=3)
	ax.set_ylim(0, max(values) * 1.15)
	fig.tight_layout()
	plt.show()
