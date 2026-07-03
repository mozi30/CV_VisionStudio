"""Full task pipeline integration tests with tiny in-memory datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.utils.data import Dataset

from vision_studio.data_loader import SimpleDataLoader
from vision_studio.evaluate import ClassificationEvaluationMetrics, LoopEvaluator
from vision_studio.models.base import BaseModel
from vision_studio.trainer.trainer import VisionTrainer
from vision_studio.types import TrainerSettings


def _pipeline_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _trainer_settings() -> TrainerSettings:
    return TrainerSettings(
        epochs=1,
        checkpoint_path=None,
        best_checkpoint_count=0,
    )


def _assert_devices(devices: list[torch.device], expected: torch.device) -> None:
    assert devices
    assert {device.type for device in devices} == {expected.type}


class _ClassificationDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[..., index % 3] = 255
        return image, {"label": index}


class _ClassificationModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3 * 4 * 4, 2)
        self.seen_input_devices: list[torch.device] = []
        self.seen_target_devices: list[torch.device] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_input_devices.append(inputs.device)
        return self.classifier(inputs.flatten(start_dim=1))

    def postprocess(self, logits: torch.Tensor):
        return {"logits": logits}

    def compute_loss(self, logits: torch.Tensor, targets: dict[str, Any]):
        labels = targets["label"].long()
        self.seen_target_devices.append(labels.device)
        return {"loss": torch.nn.functional.cross_entropy(logits, labels)}


class _DetectionDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = np.full((6, 6, 3), 32 + index, dtype=np.uint8)
        return image, {
            "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
        }


class _DetectionModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.score_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_input_devices: list[torch.device] = []
        self.seen_box_devices: list[torch.device] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_input_devices.append(inputs.device)
        batch_size = inputs.shape[0]
        return self.score_bias.expand(batch_size, 1)

    def postprocess(self, logits: torch.Tensor):
        batch_size = logits.shape[0]
        scores = torch.sigmoid(logits[:, 0])
        return [
            {
                "boxes": torch.tensor(
                    [[1.0, 1.0, 4.0, 4.0]],
                    device=logits.device,
                ),
                "scores": scores[index : index + 1],
                "labels": torch.zeros(1, dtype=torch.long, device=logits.device),
            }
            for index in range(batch_size)
        ]

    def compute_loss(self, logits: torch.Tensor, targets: dict[str, Any]):
        boxes = targets["boxes"]
        if isinstance(boxes, torch.Tensor):
            self.seen_box_devices.append(boxes.device)
        return {"loss": logits.mean() * 0.0 + self.score_bias.square()}


class _DetectionPipelineMetrics:
    def reset(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0
        self.matched_boxes = 0

    def update(self, predictions, targets, loss) -> None:
        target_boxes = targets["boxes"]
        if isinstance(target_boxes, torch.Tensor):
            boxes_per_image = [boxes for boxes in target_boxes]
        else:
            boxes_per_image = target_boxes

        self.loss_sum += float(loss.detach().cpu().item()) * len(predictions)
        self.loss_count += len(predictions)
        for prediction, boxes in zip(predictions, boxes_per_image, strict=True):
            if torch.allclose(prediction["boxes"].detach().cpu(), boxes.detach().cpu()):
                self.matched_boxes += 1

    def compute(self):
        return {
            "loss": self.loss_sum / self.loss_count if self.loss_count else 0.0,
            "matched_boxes": self.matched_boxes,
        }


class _SegmentationDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[1:4, 1:4, :] = 255
        mask = torch.zeros((5, 5), dtype=torch.long)
        mask[1:4, 1:4] = 1
        return image, {"mask": mask}


class _SegmentationModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_input_devices: list[torch.device] = []
        self.seen_mask_devices: list[torch.device] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_input_devices.append(inputs.device)
        foreground = inputs.mean(dim=1)
        background_logits = 1.0 - foreground
        foreground_logits = foreground + self.bias
        return torch.stack([background_logits, foreground_logits], dim=1)

    def postprocess(self, logits: torch.Tensor):
        return {"logits": logits, "masks": logits.argmax(dim=1)}

    def compute_loss(self, logits: torch.Tensor, targets: dict[str, Any]):
        masks = targets["mask"].long()
        self.seen_mask_devices.append(masks.device)
        return {"loss": torch.nn.functional.cross_entropy(logits, masks)}


class _SegmentationPipelineMetrics:
    def reset(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0
        self.intersection = 0.0
        self.union = 0.0

    def update(self, predictions, targets, loss) -> None:
        predicted_masks = predictions["masks"].detach().cpu()
        target_masks = targets["mask"].detach().cpu()
        self.loss_sum += float(loss.detach().cpu().item()) * int(target_masks.shape[0])
        self.loss_count += int(target_masks.shape[0])

        pred_fg = predicted_masks == 1
        target_fg = target_masks == 1
        self.intersection += float((pred_fg & target_fg).sum().item())
        self.union += float((pred_fg | target_fg).sum().item())

    def compute(self):
        return {
            "loss": self.loss_sum / self.loss_count if self.loss_count else 0.0,
            "foreground_iou": self.intersection / self.union if self.union else 0.0,
        }


@pytest.mark.parametrize("device", [_pipeline_device()])
def test_full_classification_pipeline_runs_on_available_device(device: torch.device):
    model = _ClassificationModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loader = SimpleDataLoader(_ClassificationDataset(), batch_size=2, shuffle=False)
    evaluator = LoopEvaluator(
        ClassificationEvaluationMetrics(num_classes=2, topk=(1,)),
        device=device,
    )
    trainer = VisionTrainer(
        optimizer=optimizer,
        evaluator=evaluator,
        device=device,
        settings=_trainer_settings(),
    )

    result = trainer.fit(model, loader, loader)

    assert result["global_step"] == 1
    assert len(result["history"]["train"]) == 1
    assert len(result["history"]["evaluation"]) == 1
    assert "accuracy" in result["history"]["evaluation"][0]
    _assert_devices(model.seen_input_devices, device)
    _assert_devices(model.seen_target_devices, device)


@pytest.mark.parametrize("device", [_pipeline_device()])
def test_full_detection_pipeline_runs_on_available_device(device: torch.device):
    model = _DetectionModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loader = SimpleDataLoader(_DetectionDataset(), batch_size=2, shuffle=False)
    evaluator = LoopEvaluator(_DetectionPipelineMetrics(), device=device)
    trainer = VisionTrainer(
        optimizer=optimizer,
        evaluator=evaluator,
        device=device,
        settings=_trainer_settings(),
    )

    result = trainer.fit(model, loader, loader)

    assert result["global_step"] == 1
    assert result["history"]["evaluation"][0]["matched_boxes"] == 2
    _assert_devices(model.seen_input_devices, device)
    _assert_devices(model.seen_box_devices, device)


@pytest.mark.parametrize("device", [_pipeline_device()])
def test_full_segmentation_pipeline_runs_on_available_device(device: torch.device):
    model = _SegmentationModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loader = SimpleDataLoader(_SegmentationDataset(), batch_size=2, shuffle=False)
    evaluator = LoopEvaluator(_SegmentationPipelineMetrics(), device=device)
    trainer = VisionTrainer(
        optimizer=optimizer,
        evaluator=evaluator,
        device=device,
        settings=_trainer_settings(),
    )

    result = trainer.fit(model, loader, loader)

    assert result["global_step"] == 1
    assert result["history"]["evaluation"][0]["foreground_iou"] == 1.0
    _assert_devices(model.seen_input_devices, device)
    _assert_devices(model.seen_mask_devices, device)
