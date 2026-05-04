from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseModel, InputSpec, OutputSpec
from ..types import LossOutput, ClassificationPostprocessOutput


class ImageClassifier(BaseModel):
    """Strict image classification model.

    - forward() returns only logits (raw outputs)
    - postprocess() handles softmax, argmax, and probability computation
    - Input/output specs are strictly defined
    """

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

    @property
    def input_spec(self) -> InputSpec:
        """Expected input: [batch_size, channels, height, width]."""
        return InputSpec(
            dtype=torch.float32,
        )

    @property
    def output_spec(self) -> OutputSpec:
        """Output from forward: [batch_size, num_classes] logits."""
        return OutputSpec(
            dtype=torch.float32,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass - returns ONLY logits for speed.

        Args:
            inputs: [batch_size, in_channels, height, width]

        Returns:
            logits: [batch_size, num_classes]
        """
        x = self.features(inputs)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits

    def postprocess(self, logits: Tensor) -> ClassificationPostprocessOutput:
        """
        Task-specific postprocessing of raw logits.

        Args:
            logits: [batch_size, num_classes]

        Returns:
            Dictionary with:
                - logits: [batch_size, num_classes]
                - probs: [batch_size, num_classes] softmax probabilities
                - labels: [batch_size] predicted class indices
        """
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(logits, dim=1)

        return ClassificationPostprocessOutput(
            logits=logits,
            probs=probs,
            labels=labels,
        )

    def compute_loss(
        self,
        logits: Tensor,
        targets: dict,
    ) -> LossOutput:
        """
        Compute classification loss from logits.

        Args:
            logits: [batch_size, num_classes] raw model output
            targets: dictionary with 'label' key containing [batch_size] class indices

        Returns:
            LossOutput with 'loss' key
        """
        labels = targets["label"]
        loss = F.cross_entropy(logits, labels)
        return LossOutput(loss=loss)

    def get_config(self) -> dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
        }
