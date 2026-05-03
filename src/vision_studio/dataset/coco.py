from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .base import Dataset


class CocoDataset(Dataset):
    def __init__(self, base_path: str | Path) -> None:
        base_path = Path(base_path)
        images_dir = base_path / "images"
        annotation_file = base_path / "annotations.json"
        self._init_from_files(images_dir, annotation_file)

    def __init__(self, images_dir: str | Path, annotation_file: str | Path) -> None:
        self.images_dir = Path(images_dir)
        self.annotation_file = Path(annotation_file)

        with self.annotation_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images: list[dict[str, Any]] = coco["images"]
        self.categories: list[dict[str, Any]] = coco.get("categories", [])

        self.annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            self.annotations_by_image.setdefault(image_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        image_info = self.images[index]
        image_id = image_info["id"]

        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        annotations = self.annotations_by_image.get(image_id, [])

        target = {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "boxes": [self._coco_to_xyxy(ann["bbox"]) for ann in annotations],
            "labels": [ann["category_id"] for ann in annotations],
            "annotations": annotations,
        }
        return image, target

    @staticmethod
    def _coco_to_xyxy(box: list[float]) -> list[float]:
        x, y, w, h = box
        return [x, y, x + w, y + h]
