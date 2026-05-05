"""Visualization helpers."""

from __future__ import annotations

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
