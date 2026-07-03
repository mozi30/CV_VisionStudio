"""Tests for PyTorch dataset compatibility."""

from torch.utils.data import Dataset as TorchDataset

from vision_studio.dataset import (
    CocoDataset,
    Dataset,
    ImageClassificationDataset,
    ImageNetClassificationDataset,
    MnistDataset,
)
from vision_studio.dataset.base import Dataset as BaseDataset


def test_public_dataset_type_is_torch_dataset() -> None:
    """Public dataset aliases should point at the classic PyTorch Dataset."""
    assert Dataset is TorchDataset
    assert BaseDataset is TorchDataset


def test_builtin_datasets_subclass_torch_dataset() -> None:
    """Built-in dataset implementations should work anywhere PyTorch expects Dataset."""
    assert issubclass(ImageClassificationDataset, TorchDataset)
    assert issubclass(CocoDataset, TorchDataset)
    assert issubclass(ImageNetClassificationDataset, TorchDataset)
    assert issubclass(MnistDataset, TorchDataset)
