from __future__ import annotations

import torch
from torchvision.transforms import v2

from vision_studio.augmentation import Compose, TorchVisionAugmentation


def test_compose_accepts_torchvision_classification_transforms() -> None:
    image = torch.zeros((3, 8, 8), dtype=torch.uint8)

    transform = Compose(
        [
            v2.Resize((4, 4)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    transformed = transform(image)

    assert transformed.shape == (3, 4, 4)
    assert transformed.dtype == torch.float32


def test_compose_updates_detection_boxes_with_torchvision_resize() -> None:
    image = torch.zeros((3, 10, 20), dtype=torch.uint8)
    target = {
        "boxes": [[2.0, 3.0, 10.0, 8.0]],
        "labels": [1],
    }

    transformed_image, transformed_target = Compose([v2.Resize((20, 40))])(
        image,
        target,
    )

    assert transformed_image.shape == (3, 20, 40)
    assert torch.equal(
        transformed_target["boxes"],
        torch.tensor([[4.0, 6.0, 20.0, 16.0]]),
    )
    assert transformed_target["labels"] == [1]


def test_compose_updates_detection_boxes_with_torchvision_flip() -> None:
    image = torch.zeros((3, 10, 20), dtype=torch.uint8)
    target = {
        "boxes": torch.tensor([[2.0, 3.0, 10.0, 8.0]]),
        "labels": torch.tensor([1]),
    }

    _, transformed_target = Compose([v2.RandomHorizontalFlip(p=1.0)])(image, target)

    assert torch.equal(
        transformed_target["boxes"],
        torch.tensor([[10.0, 3.0, 18.0, 8.0]]),
    )
    assert torch.equal(transformed_target["labels"], torch.tensor([1]))


def test_torchvision_augmentation_adapter_wraps_single_transform() -> None:
    image = torch.zeros((3, 10, 20), dtype=torch.uint8)
    target = {"boxes": [[2.0, 3.0, 10.0, 8.0]]}

    transformed_image, transformed_target = TorchVisionAugmentation(v2.Resize((5, 10)))(
        image,
        target,
    )

    assert transformed_image.shape == (3, 5, 10)
    assert torch.equal(
        transformed_target["boxes"],
        torch.tensor([[1.0, 1.5, 5.0, 4.0]]),
    )
