from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class DataLoader(ABC, Iterable[T], Generic[T]):
    """Base interface for objects that yield batches or samples."""

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Return an iterator over items."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of batches/items if known.

        Override only when length is well-defined.
        """
        raise TypeError(f"{self.__class__.__name__} has no defined length")
