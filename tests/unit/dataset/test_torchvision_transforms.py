from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from vision_studio.dataset import ImageNetClassificationDataset
from vision_studio.transforms import Normalize, ToTensor


def test_preprocessing_wrappers_delegate_to_torchvision() -> None:
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    target = {"label": 1}

    tensor, transformed_target = ToTensor()(image, target)
    normalized, normalized_target = Normalize(
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
    )(
        tensor,
        transformed_target,
    )

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 4, 4)
    assert torch.isclose(tensor[0, 0, 0], torch.tensor(1.0))
    assert torch.equal(normalized, tensor)
    assert normalized_target == target


def test_preprocessing_wrappers_support_standard_single_image_call() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    tensor = ToTensor()(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 4, 4)


def test_imagenet_dataset_accepts_standard_torchvision_pipeline(tmp_path) -> None:
    class_dir = tmp_path / "train" / "class-a"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (6, 6), color=(128, 64, 32)).save(class_dir / "sample.jpg")

    dataset = ImageNetClassificationDataset(
        tmp_path,
        transform=transforms.Compose(
            [
                transforms.Resize((2, 2)),
                transforms.ToTensor(),
            ]
        ),
    )

    image, target = dataset[0]

    assert image.shape == (3, 2, 2)
    assert image.dtype == torch.float32
    assert target == {"label": 0}


def test_imagenet_dataset_preserves_internal_transform_type_errors(tmp_path) -> None:
    class_dir = tmp_path / "train" / "class-a"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (6, 6), color=(128, 64, 32)).save(class_dir / "sample.jpg")

    class BrokenTransform:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, image, target):
            self.calls += 1
            raise TypeError("internal transform bug")

    transform = BrokenTransform()
    dataset = ImageNetClassificationDataset(tmp_path, transform=transform)

    with pytest.raises(TypeError, match="internal transform bug"):
        dataset[0]

    assert transform.calls == 1
