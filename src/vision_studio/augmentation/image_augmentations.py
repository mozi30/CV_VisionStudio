from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .base import Augmentation
from .utils import (
    clamp_uint8,
    copy_target,
    image_size,
    to_numpy,
    to_pil,
    update_boxes_flip_horizontal,
    update_boxes_flip_vertical,
)


class HorizontalFlip(Augmentation):
    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image_size(image)
        out = np.flip(image, axis=1).copy()
        target = copy_target(target)

        if "boxes" in target:
            target["boxes"] = update_boxes_flip_horizontal(
                np.asarray(target["boxes"]),
                width=w,
            )
        return out, target


class VerticalFlip(Augmentation):
    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image_size(image)
        out = np.flip(image, axis=0).copy()
        target = copy_target(target)

        if "boxes" in target:
            target["boxes"] = update_boxes_flip_vertical(
                np.asarray(target["boxes"]),
                height=h,
            )
        return out, target


class Rotate(Augmentation):
    def __init__(self, degrees: float, expand: bool = False):
        self.degrees = degrees
        self.expand = expand

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        angle = random.uniform(-self.degrees, self.degrees)
        pil = to_pil(image)
        out = pil.rotate(angle, resample=Image.BILINEAR, expand=self.expand)
        return to_numpy(out), copy_target(target)


class RandomCrop(Augmentation):
    def __init__(self, crop_height: int, crop_width: int):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image.shape[:2]
        if self.crop_height > h or self.crop_width > w:
            raise ValueError("crop size must be <= image size")

        top = random.randint(0, h - self.crop_height)  # noqa: S311
        left = random.randint(0, w - self.crop_width)  # noqa: S311

        out = image[top : top + self.crop_height, left : left + self.crop_width].copy()
        target = copy_target(target)

        if "boxes" in target:
            boxes = np.asarray(target["boxes"]).copy()
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.crop_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.crop_height)
            target["boxes"] = boxes

        return out, target


class Resize(Augmentation):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(
        self,
        image: np.ndarray | Image.Image,
        target: dict[str, Any] | None = None,
    ):
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            pil = to_pil(image)
        else:  # already PIL
            w, h = image.size
            pil = image

        out = pil.resize((self.width, self.height), resample=Image.BILINEAR)

        if target is not None:
            target = copy_target(target)

            if "boxes" in target:
                boxes = np.asarray(target["boxes"]).copy().astype(np.float32)
                boxes[:, [0, 2]] *= self.width / w
                boxes[:, [1, 3]] *= self.height / h
                target["boxes"] = boxes

            return to_numpy(out), target

        return to_numpy(out)


class RandomScale(Augmentation):
    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image.shape[:2]
        scale = random.uniform(self.min_scale, self.max_scale)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        return Resize(new_h, new_w)(image, target)


class Translate(Augmentation):
    def __init__(self, max_dx: int, max_dy: int, fill: int = 0):
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.fill = fill

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = image.shape[:2]
        dx = random.randint(-self.max_dx, self.max_dx)
        dy = random.randint(-self.max_dy, self.max_dy)

        out = np.full_like(image, self.fill)

        src_x1 = max(0, -dx)
        src_x2 = min(w, w - dx) if dx >= 0 else w
        dst_x1 = max(0, dx)
        dst_x2 = min(w, w + dx) if dx < 0 else w

        src_y1 = max(0, -dy)
        src_y2 = min(h, h - dy) if dy >= 0 else h
        dst_y1 = max(0, dy)
        dst_y2 = min(h, h + dy) if dy < 0 else h

        out[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        target = copy_target(target)
        if "boxes" in target:
            boxes = np.asarray(target["boxes"]).copy()
            boxes[:, [0, 2]] += dx
            boxes[:, [1, 3]] += dy
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            target["boxes"] = boxes

        return out, target


class Shear(Augmentation):
    def __init__(self, max_shear_degrees: float = 10.0):
        self.max_shear_degrees = max_shear_degrees

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        shear_deg = random.uniform(-self.max_shear_degrees, self.max_shear_degrees)
        shear = math.tan(math.radians(shear_deg))
        pil = to_pil(image)
        w, h = pil.size
        out = pil.transform(
            (w, h),
            Image.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.BILINEAR,
        )
        return to_numpy(out), copy_target(target)


class PerspectiveTransform(Augmentation):
    def __init__(self, distortion_scale: float = 0.2):
        self.distortion_scale = distortion_scale

    def __call__(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pil = to_pil(image)
        w, h = pil.size
        dx = int(w * self.distortion_scale)
        dy = int(h * self.distortion_scale)

        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [
            (random.randint(0, dx), random.randint(0, dy)),
            (w - random.randint(0, dx), random.randint(0, dy)),
            (w - random.randint(0, dx), h - random.randint(0, dy)),
            (random.randint(0, dx), h - random.randint(0, dy)),
        ]

        coeffs = _find_perspective_coeffs(src, dst)
        out = pil.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        return to_numpy(out), copy_target(target)


class Brightness(Augmentation):
    def __init__(self, min_factor: float = 0.8, max_factor: float = 1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        factor = random.uniform(self.min_factor, self.max_factor)
        pil = ImageEnhance.Brightness(to_pil(image)).enhance(factor)
        return to_numpy(pil), copy_target(target)


class Contrast(Augmentation):
    def __init__(self, min_factor: float = 0.8, max_factor: float = 1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        factor = random.uniform(self.min_factor, self.max_factor)
        pil = ImageEnhance.Contrast(to_pil(image)).enhance(factor)
        return to_numpy(pil), copy_target(target)


class Saturation(Augmentation):
    def __init__(self, min_factor: float = 0.8, max_factor: float = 1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        factor = random.uniform(self.min_factor, self.max_factor)
        pil = ImageEnhance.Color(to_pil(image)).enhance(factor)
        return to_numpy(pil), copy_target(target)


class HueShift(Augmentation):
    def __init__(self, max_delta: int = 20):
        self.max_delta = max_delta

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        pil = to_pil(image).convert("RGB")
        hsv = np.array(pil.convert("HSV"), dtype=np.uint8)
        delta = random.randint(-self.max_delta, self.max_delta)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + delta) % 256
        out = Image.fromarray(hsv, mode="HSV").convert("RGB")
        return np.array(out), copy_target(target)


class ColorJitter(Augmentation):
    def __init__(
        self,
        brightness: tuple[float, float] = (0.8, 1.2),
        contrast: tuple[float, float] = (0.8, 1.2),
        saturation: tuple[float, float] = (0.8, 1.2),
        hue_delta: int = 20,
    ):
        self.brightness = Brightness(*brightness)
        self.contrast = Contrast(*contrast)
        self.saturation = Saturation(*saturation)
        self.hue = HueShift(hue_delta)

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        transforms = [self.brightness, self.contrast, self.saturation, self.hue]
        random.shuffle(transforms)
        for t in transforms:
            image, target = t(image, target)
        return image, target


class Grayscale(Augmentation):
    def __init__(self, keep_channels: bool = True):
        self.keep_channels = keep_channels

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        pil = to_pil(image).convert("L")
        arr = np.array(pil)
        if self.keep_channels:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr, copy_target(target)


class GaussianNoise(Augmentation):
    def __init__(self, std: float = 10.0):
        self.std = std

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        noise = np.random.normal(0, self.std, size=image.shape)
        out = image.astype(np.float32) + noise
        return clamp_uint8(out), copy_target(target)


class GaussianBlur(Augmentation):
    def __init__(self, radius_min: float = 0.1, radius_max: float = 2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        radius = random.uniform(self.radius_min, self.radius_max)
        pil = to_pil(image).filter(ImageFilter.GaussianBlur(radius=radius))
        return to_numpy(pil), copy_target(target)


class MotionBlur(Augmentation):
    def __init__(self, kernel_size: int = 9):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        arr = image.astype(np.float32)
        k = self.kernel_size
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k

        out = _convolve_image(arr, kernel)
        return clamp_uint8(out), copy_target(target)


class Cutout(Augmentation):
    def __init__(self, mask_height: int, mask_width: int, fill: int = 0):
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.fill = fill

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        h, w = image.shape[:2]
        top = random.randint(0, max(0, h - self.mask_height))
        left = random.randint(0, max(0, w - self.mask_width))

        out = image.copy()
        out[top : top + self.mask_height, left : left + self.mask_width] = self.fill
        return out, copy_target(target)


class Mixup(Augmentation):
    def __init__(
        self,
        alpha: float = 0.2,
        source_images: list[np.ndarray] | None = None,
    ):
        self.alpha = alpha
        self.source_images = source_images or []

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        if not self.source_images:
            return image, copy_target(target)

        other = random.choice(self.source_images)
        if other.shape != image.shape:
            raise ValueError("Mixup requires same image shape")

        lam = np.random.beta(self.alpha, self.alpha)
        out = lam * image.astype(np.float32) + (1.0 - lam) * other.astype(np.float32)
        target = copy_target(target)
        target["mixup_lambda"] = lam
        return clamp_uint8(out), target


class CutMix(Augmentation):
    def __init__(
        self,
        alpha: float = 1.0,
        source_images: list[np.ndarray] | None = None,
    ):
        self.alpha = alpha
        self.source_images = source_images or []

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        if not self.source_images:
            return image, copy_target(target)

        other = random.choice(self.source_images)
        if other.shape != image.shape:
            raise ValueError("CutMix requires same image shape")

        lam = np.random.beta(self.alpha, self.alpha)
        h, w = image.shape[:2]

        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)

        out = image.copy()
        out[y1:y2, x1:x2] = other[y1:y2, x1:x2]

        actual_area = (x2 - x1) * (y2 - y1)
        lam_adjusted = 1.0 - actual_area / (w * h)

        target = copy_target(target)
        target["cutmix_lambda"] = lam_adjusted
        return out, target


class Normalize(Augmentation):
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        arr = image.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        out = (arr - self.mean) / self.std
        return out, copy_target(target)


class RandomResizedCrop(Augmentation):
    def __init__(
        self,
        output_height: int,
        output_width: int,
        scale: tuple[float, float] = (0.8, 1.0),
    ):
        self.output_height = output_height
        self.output_width = output_width
        self.scale = scale

    def __call__(self, image: np.ndarray, target: dict[str, Any]):
        h, w = image.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area  # noqa: S311
            aspect = random.uniform(0.75, 1.3333)  # noqa: S311

            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                top = random.randint(0, h - crop_h)  # noqa: S311
                left = random.randint(0, w - crop_w)  # noqa: S311
                crop = image[top : top + crop_h, left : left + crop_w]
                return Resize(self.output_height, self.output_width)(crop, target)

        return Resize(self.output_height, self.output_width)(image, target)


def _convolve_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    if image.ndim == 2:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="edge")
        out = np.zeros_like(image, dtype=np.float32)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                patch = padded[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                out[i, j] = np.sum(patch * kernel)
        return out

    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                patch = padded[i : i + kernel.shape[0], j : j + kernel.shape[1], c]
                out[i, j, c] = np.sum(patch * kernel)
    return out


def _find_perspective_coeffs(
    src: list[tuple[int, int]],
    dst: list[tuple[int, int]],
) -> list[float]:
    matrix = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])

    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(src, dtype=np.float64).reshape(8)
    coeffs = np.linalg.solve(a, b)
    return coeffs.tolist()
