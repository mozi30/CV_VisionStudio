"""Preprocessing augmentations."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from PIL import ImageFilter

from .base import Augmentation
from .utils import clamp_uint8, copy_target, is_grayscale, to_numpy, to_pil


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image.astype(np.float32)
	if image.ndim == 3 and image.shape[2] == 1:
		return image[..., 0].astype(np.float32)
	if image.ndim == 3 and image.shape[2] >= 3:
		arr = image[..., :3].astype(np.float32)
		return 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
	raise ValueError("image must be 2D or 3D with channels")


def _equalize_channel(channel: np.ndarray) -> np.ndarray:
	hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
	cdf = hist.cumsum()
	cdf_mask = np.ma.masked_equal(cdf, 0)
	if cdf_mask.max() == cdf_mask.min():
		return channel.copy()
	cdf_scaled = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())
	cdf_final = np.ma.filled(cdf_scaled, 0).astype(np.uint8)
	return cdf_final[channel]


def _convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	pad = kernel.shape[0] // 2
	padded = np.pad(image, ((pad, pad), (pad, pad)), mode="edge")
	out = np.zeros_like(image, dtype=np.float32)
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			patch = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
			out[i, j] = np.sum(patch * kernel)
	return out


def _non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
	out = np.zeros_like(magnitude, dtype=np.float32)
	angle = angle % 180.0
	h, w = magnitude.shape
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			q = 0.0
			r = 0.0
			if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180.0):
				q = magnitude[i, j + 1]
				r = magnitude[i, j - 1]
			elif 22.5 <= angle[i, j] < 67.5:
				q = magnitude[i + 1, j - 1]
				r = magnitude[i - 1, j + 1]
			elif 67.5 <= angle[i, j] < 112.5:
				q = magnitude[i + 1, j]
				r = magnitude[i - 1, j]
			else:
				q = magnitude[i - 1, j - 1]
				r = magnitude[i + 1, j + 1]

			if magnitude[i, j] >= q and magnitude[i, j] >= r:
				out[i, j] = magnitude[i, j]
	return out


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
	out = np.zeros_like(strong, dtype=np.uint8)
	out[strong] = 255
	stack = list(zip(*np.where(strong)))
	neighbors = [
		(-1, -1),
		(-1, 0),
		(-1, 1),
		(0, -1),
		(0, 1),
		(1, -1),
		(1, 0),
		(1, 1),
	]

	while stack:
		i, j = stack.pop()
		for di, dj in neighbors:
			ni = i + di
			nj = j + dj
			if 0 <= ni < out.shape[0] and 0 <= nj < out.shape[1] and weak[ni, nj]:
				weak[ni, nj] = False
				out[ni, nj] = 255
				stack.append((ni, nj))
	return out


class HistogramEqualization(Augmentation):
	def __init__(self, per_channel: bool = True, keep_channels: bool = True):
		self.per_channel = per_channel
		self.keep_channels = keep_channels

	def __call__(self, image: np.ndarray, target: dict[str, Any]):
		arr = clamp_uint8(image)

		if is_grayscale(arr):
			gray = arr if arr.ndim == 2 else arr[..., 0]
			eq = _equalize_channel(gray)
			if arr.ndim == 3:
				eq = eq[..., None]
			return eq, copy_target(target)

		if self.per_channel:
			out = np.empty_like(arr)
			for c in range(arr.shape[2]):
				out[..., c] = _equalize_channel(arr[..., c])
			return out, copy_target(target)

		gray = _to_grayscale_float(arr).astype(np.uint8)
		eq = _equalize_channel(gray)
		if self.keep_channels:
			out = np.stack([eq, eq, eq], axis=-1)
			return out, copy_target(target)

		return eq, copy_target(target)


class BrightnessNormalization(Augmentation):
	def __init__(self, target_mean: float = 128.0, per_channel: bool = False):
		self.target_mean = target_mean
		self.per_channel = per_channel

	def __call__(self, image: np.ndarray, target: dict[str, Any]):
		arr = image.astype(np.float32)

		if arr.ndim == 2:
			mean = arr.mean()
		elif self.per_channel:
			mean = arr.mean(axis=(0, 1), keepdims=True)
		else:
			mean = arr.mean()

		out = arr + (self.target_mean - mean)
		return clamp_uint8(out), copy_target(target)


class ContrastNormalization(Augmentation):
    def __init__(
        self,
        target_std: float = 50.0,
        per_channel: bool = False,
        eps: float = 1e-6,
    ):
        self.target_std = target_std
        self.per_channel = per_channel
        self.eps = eps

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        arr = image.astype(np.float32, copy=False)

        if arr.ndim == 2:
            mean, std = cv2.meanStdDev(arr)
            mean = mean[0][0]
            std = std[0][0]

            scale = self.target_std / max(std, self.eps)
            out = (arr - mean) * scale + mean

        elif self.per_channel:
            # OpenCV returns per-channel stats efficiently
            mean, std = cv2.meanStdDev(arr)
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)

            scale = self.target_std / np.maximum(std, self.eps)
            out = (arr - mean) * scale + mean

        else:
            mean, std = cv2.meanStdDev(arr)
            mean = float(mean.mean())
            std = float(std.mean())

            scale = self.target_std / max(std, self.eps)
            out = (arr - mean) * scale + mean

        return np.clip(out, 0, 255).astype(np.uint8), copy_target(target)


class Grayscale(Augmentation):
	def __init__(self, keep_channels: bool = True):
		self.keep_channels = keep_channels

	def __call__(self, image: np.ndarray, target: dict[str, Any]):
		pil = to_pil(image).convert("L")
		arr = np.array(pil)
		if self.keep_channels:
			arr = np.stack([arr, arr, arr], axis=-1)
		return arr, copy_target(target)


class GaussianBlurFilter(Augmentation):
	def __init__(self, radius: float = 1.0):
		if radius < 0:
			raise ValueError("radius must be >= 0")
		self.radius = radius

	def __call__(self, image: np.ndarray, target: dict[str, Any]):
		pil = to_pil(image).filter(ImageFilter.GaussianBlur(radius=self.radius))
		return to_numpy(pil), copy_target(target)


class SobelFilter(Augmentation):
    def __init__(self, keep_channels: bool = True, normalize: bool = True):
        self.keep_channels = keep_channels
        self.normalize = normalize

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        gray = _to_grayscale_float(image).astype(np.float32, copy=False)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if self.normalize:
            max_val = mag.max()
            if max_val > 0:
                mag *= 255.0 / max_val
        out = np.clip(mag, 0, 255).astype(np.uint8)
        if self.keep_channels:
            out = np.repeat(out[..., None], 3, axis=-1)
        return out, copy_target(target)


class CannyEdge(Augmentation):
    def __init__(
        self,
        low_threshold: float = 50.0,
        high_threshold: float = 150.0,
        blur_radius: float = 1.0,
        keep_channels: bool = True,
    ):
        if low_threshold < 0 or high_threshold < 0:
            raise ValueError("thresholds must be >= 0")
        if low_threshold > high_threshold:
            raise ValueError("low_threshold must be <= high_threshold")
        if blur_radius < 0:
            raise ValueError("blur_radius must be >= 0")

        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.blur_radius = blur_radius
        self.keep_channels = keep_channels

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        # Convert to grayscale uint8
        if image.ndim == 3:
            if image.shape[-1] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image[..., 0]
        else:
            gray = image

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Optional blur
        if self.blur_radius > 0:
            k = int(2 * round(self.blur_radius) + 1)
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # Fast Canny
        edges = cv2.Canny(
            gray,
            threshold1=self.low_threshold,
            threshold2=self.high_threshold,
            L2gradient=True,
        )

        if self.keep_channels:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges, copy_target(target)


class CustomFilter(Augmentation):
    def __init__(self, kernel: np.ndarray, normalize: bool = False, clip: bool = True):
        if kernel.ndim != 2:
            raise ValueError("kernel must be 2D")

        kernel = kernel.astype(np.float32)

        if normalize:
            total = kernel.sum()
            if total != 0:
                kernel = kernel / total

        self.kernel = kernel
        self.clip = clip

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        arr = image.astype(np.float32, copy=False)

        out = cv2.filter2D(
            arr,
            ddepth=-1,
            kernel=self.kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )

        if self.clip:
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out, copy_target(target)
