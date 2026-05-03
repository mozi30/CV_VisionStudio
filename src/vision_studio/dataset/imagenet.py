import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import Dataset


class ImageNetClassificationDataset(Dataset):
    """ImageNet classification style dataset with class folders in train/val/test directories."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Any = None,
        output_image_size: tuple[int, int] | None = None,
    ) -> None:
        """Args:
        root_dir (str): Root directory of the dataset.
        split (str): One of 'train', 'val', or 'test'.
        transform (callable, optional): A function/transform to apply to each image.
        """
        assert split in ["train", "val", "test"], (
            "Split must be one of 'train', 'val', 'test'"
        )

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.output_image_size = output_image_size

        # Define the path for the specified split
        self.split_dir = os.path.join(root_dir, split)

        if not os.path.isdir(self.split_dir):
            raise ValueError(
                f"Split directory {self.split_dir} does not exist or is not a directory."
            )

        # List of class directories in the split
        self.class_names = os.listdir(self.split_dir)

        if not self.class_names:
            raise ValueError(f"No class directories found in {self.split_dir}")

        self.class_names.sort()  # To ensure consistent order
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        # Create a list of image paths and their corresponding class indices
        self.image_paths = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.split_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")):
                    self.image_paths.append(
                        (
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name],
                        )
                    )
                else:
                    print(f"Warning: Skipping non-image file {img_name} in {class_dir}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path, label = self.image_paths[index]

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        target = {"label": label}

        if self.transform:
            image, target = self.transform(image, target)

        # final conversion: NumPy HWC -> Tensor CHW
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

            if image.ndim == 3:
                image = image.permute(2, 0, 1)

            image = image.float()

            # only divide if image is still 0–255
            if image.max() > 1:
                image = image / 255.0

        return image, target

    def get_num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.class_names)

    def get_image_size(self) -> tuple[int, int] | None:
        return self.output_image_size
