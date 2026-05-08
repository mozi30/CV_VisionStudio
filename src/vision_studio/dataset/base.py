from abc import ABC, abstractmethod
from typing import Any


class Dataset(ABC):
    """Base interface for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Return one sample and its target."""
        raise NotImplementedError
    
    @abstractmethod
    def get_class_sample_counts(self) -> dict[int, int] | None:
        """Return a mapping from class id to sample count, or None if not available."""
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> tuple[int, int] | None:
        """Return the image size (height, width) of the dataset."""
        raise NotImplementedError
