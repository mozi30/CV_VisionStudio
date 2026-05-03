from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import Dataset


class MnistDataset(Dataset):
    """MNIST handwritten digit dataset.

    Loads from the original MNIST binary format files:
    - train-images-idx3-ubyte.gz: training set images (9912422 bytes)
    - train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
    - t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
    - t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

    Also supports uncompressed versions:
    - train-images.idx3-ubyte
    - train-labels.idx1-ubyte
    - t10k-images.idx3-ubyte
    - t10k-labels.idx1-ubyte

    Args:
        data_dir: Path to directory containing MNIST files
        split: Either "train" or "test" to select which split to load

    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
    ) -> None:
        """Initialize MNIST dataset.

        Args:
            data_dir: Directory containing MNIST binary files
            split: "train" or "test" split

        Raises:
            ValueError: If split is not "train" or "test"
            FileNotFoundError: If required MNIST files are not found

        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.data_dir = Path(data_dir)
        self.split = split

        # Set file names based on split - try both .gz and non-.gz versions
        if split == "train":
            # Try gzipped first, then uncompressed
            self.images_file = self._find_file(
                "train-images-idx3-ubyte.gz",
                "train-images.idx3-ubyte",
            )
            self.labels_file = self._find_file(
                "train-labels-idx1-ubyte.gz",
                "train-labels.idx1-ubyte",
            )
        else:  # test
            self.images_file = self._find_file(
                "t10k-images-idx3-ubyte.gz",
                "t10k-images.idx3-ubyte",
            )
            self.labels_file = self._find_file(
                "t10k-labels-idx1-ubyte.gz",
                "t10k-labels.idx1-ubyte",
            )

        # Load data
        self.images = self._load_images()
        self.labels = self._load_labels()

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Mismatch between images ({len(self.images)}) "
                f"and labels ({len(self.labels)})"
            )

    def _find_file(self, *possible_names: str) -> Path:
        """Find the first available file from a list of possible names.

        Args:
            *possible_names: Possible file names to try

        Returns:
            Path to the found file

        Raises:
            FileNotFoundError: If none of the files are found

        """
        for name in possible_names:
            file_path = self.data_dir / name
            if file_path.exists():
                return file_path

        raise FileNotFoundError(
            f"Could not find any of {possible_names} in {self.data_dir}"
        )

    def _open_file(self, file_path: Path):
        """Open a file, handling both gzipped and uncompressed formats.

        Args:
            file_path: Path to the file

        Returns:
            File-like object

        """
        if str(file_path).endswith(".gz"):
            return gzip.open(file_path, "rb")
        else:
            return open(file_path, "rb")

    def _load_images(self) -> np.ndarray:
        """Load images from binary file.

        Returns:
            Array of shape (N, 28, 28) with uint8 values [0, 255]

        """
        with self._open_file(self.images_file) as f:
            # Read header: magic number (4 bytes), num images (4 bytes),
            # height (4 bytes), width (4 bytes)
            magic = struct.unpack(">I", f.read(4))[0]
            if magic != 2051:  # MNIST image magic number
                raise ValueError(f"Invalid magic number for images: {magic}")

            num_images = struct.unpack(">I", f.read(4))[0]
            height = struct.unpack(">I", f.read(4))[0]
            width = struct.unpack(">I", f.read(4))[0]

            # Read image data
            data = f.read(num_images * height * width)
            images = np.frombuffer(data, dtype=np.uint8)
            images = images.reshape(num_images, height, width)

        return images

    def _load_labels(self) -> np.ndarray:
        """Load labels from binary file.

        Returns:
            Array of shape (N,) with uint8 values [0, 9]

        """
        with self._open_file(self.labels_file) as f:
            # Read header: magic number (4 bytes), num items (4 bytes)
            magic = struct.unpack(">I", f.read(4))[0]
            if magic != 2049:  # MNIST label magic number
                raise ValueError(f"Invalid magic number for labels: {magic}")

            num_items = struct.unpack(">I", f.read(4))[0]

            # Read label data
            data = f.read(num_items)
            labels = np.frombuffer(data, dtype=np.uint8)

        return labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        """Return one image and its label.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (image, target) where:
            - image: PIL Image (grayscale, 28x28)
            - target: dict with 'label' (0-9) and 'index'

        """
        # Convert numpy array to PIL Image
        image_array = self.images[index]
        image = Image.fromarray(image_array, mode="L")

        label = int(self.labels[index])
        target = {
            "label": label,
            "index": index,
            "split": self.split,
        }

        return image, target

    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a digit.

        Args:
            class_id: The digit (0-9)

        Returns:
            String representation of the digit

        """
        if not 0 <= class_id <= 9:
            return "unknown"
        return str(class_id)

    def get_num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return 10

    def get_image_size(self) -> tuple[int, int]:
        """Return the image size (height, width) of the dataset."""
        return (28, 28)
