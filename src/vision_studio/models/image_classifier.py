from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseModel


class ImageClassifier(BaseModel):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(
        self,
        inputs: Tensor,
        targets: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        x = self.features(inputs)
        x = x.flatten(1)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "labels": labels,
        }

    def compute_loss(
        self,
        outputs: dict[str, Any],
        targets: dict[str, Any],
    ) -> dict[str, Tensor]:
        logits = outputs["logits"]
        labels = targets["label"]
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss}

    def get_config(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
        }
