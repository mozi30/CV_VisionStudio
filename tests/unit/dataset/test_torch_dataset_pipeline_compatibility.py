"""Tests for using plain PyTorch datasets in Vision Studio loaders."""

import numpy as np
import torch
from torch.utils.data import Dataset

from vision_studio.data_loader.balanced_loader import BalancedDataLoader
from vision_studio.data_loader.simple_loader import SimpleDataLoader


class _PlainTorchDataset(Dataset):
    """Minimal PyTorch dataset with no Vision Studio helper methods."""

    def __init__(self) -> None:
        self.samples = [
            (torch.tensor([1.0, 0.0]), 0),
            (torch.tensor([0.0, 1.0]), 1),
            (torch.tensor([0.5, 0.5]), 1),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class _PlainClassificationDataset(Dataset):
    """Classification dataset returning the classic ``(image, label)`` shape."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.full((3, 4, 4), float(index))
        label = index
        return image, label


class _PlainDetectionDataset(Dataset):
    """Detection dataset returning variable-length object targets."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.full((3, 4, 4), float(index))
        if index == 0:
            target = {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "labels": torch.tensor([1]),
            }
        else:
            target = {
                "boxes": torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 2.0, 2.0],
                    ]
                ),
                "labels": torch.tensor([1, 2]),
            }
        return image, target


class _PlainSegmentationDataset(Dataset):
    """Segmentation dataset returning image tensors and mask targets."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.full((3, 4, 4), float(index))
        mask = torch.full((4, 4), index, dtype=torch.long)
        return image, {"mask": mask}


class _NumpyHwcDataset(Dataset):
    """Dataset returning a NumPy HWC image."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        image = np.full((4, 5, 3), 255, dtype=np.uint8)
        return image, 0


def test_simple_loader_normalizes_plain_torch_dataset_targets() -> None:
    """A normal PyTorch ``(input, label)`` dataset should produce target dicts."""
    loader = SimpleDataLoader(_PlainTorchDataset(), batch_size=2)

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 2)
    assert torch.equal(targets["label"], torch.tensor([0, 1]))


def test_simple_loader_batches_plain_classification_dataset() -> None:
    """Classification samples should batch to image tensors and label tensors."""
    loader = SimpleDataLoader(_PlainClassificationDataset(), batch_size=2)

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 3, 4, 4)
    assert torch.equal(targets["label"], torch.tensor([0, 1]))


def test_simple_loader_batches_plain_detection_dataset() -> None:
    """Detection targets with variable object counts should stay per-image lists."""
    loader = SimpleDataLoader(_PlainDetectionDataset(), batch_size=2)

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 3, 4, 4)
    assert isinstance(targets["boxes"], list)
    assert isinstance(targets["labels"], list)
    assert torch.equal(targets["boxes"][0], torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert targets["boxes"][1].shape == (2, 4)
    assert torch.equal(targets["labels"][0], torch.tensor([1]))
    assert torch.equal(targets["labels"][1], torch.tensor([1, 2]))


def test_simple_loader_batches_plain_segmentation_dataset() -> None:
    """Segmentation masks should batch as a dense ``[batch, height, width]`` tensor."""
    loader = SimpleDataLoader(_PlainSegmentationDataset(), batch_size=2)

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 3, 4, 4)
    assert targets["mask"].shape == (2, 4, 4)
    assert targets["mask"].dtype == torch.long
    assert torch.equal(targets["mask"][0], torch.zeros((4, 4), dtype=torch.long))
    assert torch.equal(targets["mask"][1], torch.ones((4, 4), dtype=torch.long))


def test_simple_loader_converts_numpy_hwc_images_to_chw_float_tensors() -> None:
    loader = SimpleDataLoader(_NumpyHwcDataset(), batch_size=1)

    inputs, targets = next(iter(loader))

    assert inputs.shape == (1, 3, 4, 5)
    assert inputs.dtype == torch.float32
    assert torch.equal(inputs, torch.ones((1, 3, 4, 5)))
    assert torch.equal(targets["label"], torch.tensor([0]))


def test_balanced_loader_infers_counts_from_plain_torch_dataset_labels() -> None:
    """BalancedDataLoader should not require Vision Studio metadata helpers."""
    loader = BalancedDataLoader(_PlainTorchDataset(), batch_size=2, shuffle=False)

    assert loader.class_counts == {0: 1, 1: 2}

    inputs, targets = next(iter(loader))

    assert inputs.shape == (2, 2)
    assert "label" in targets
