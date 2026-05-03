from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .base import Dataset


class ImageClassificationDataset(Dataset):
    """Dataset for image classification tasks.

    Expects a structure with:
    - images_dir/: Directory containing image files
    - annotation_file (JSON): Contains class labels for each image

    Annotation file format:
    {
        "images": [
            {"id": 0, "file_name": "image_0.jpg"},
            ...
        ],
        "categories": [
            {"id": 0, "name": "class_0"},
            {"id": 1, "name": "class_1"},
            ...
        ],
        "annotations": [
            {"image_id": 0, "category_id": 0},
            ...
        ]
    }
    """

    def __init__(self, images_dir: str | Path, annotation_file: str | Path) -> None:
        """Initialize the classification dataset.

        Args:
            images_dir: Path to directory containing image files
            annotation_file: Path to JSON file with annotations

        """
        self.images_dir = Path(images_dir)
        self.annotation_file = Path(annotation_file)

        with self.annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.images: list[dict[str, Any]] = data["images"]
        self.categories: list[dict[str, Any]] = data.get("categories", [])

        # Map image_id to class label
        self.image_labels: dict[int, int] = {}
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            self.image_labels[image_id] = category_id

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        """Return one image and its target label.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (image, target) where target contains 'label' and 'image_id'

        """
        image_info = self.images[index]
        image_id = image_info["id"]

        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        label = self.image_labels.get(image_id, 0)

        target = {
            "label": label,
            "image_id": image_id,
            "file_name": image_info["file_name"],
        }
        return image, target

    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a given class ID.

        Args:
            class_id: The class ID

        Returns:
            The class name, or "unknown" if not found

        """
        for cat in self.categories:
            if cat["id"] == class_id:
                return cat["name"]
        return "unknown"

    def get_num_classes(self) -> int:
        """Get the total number of classes.

        Returns:
            Number of classes

        """
        return len(self.categories)
