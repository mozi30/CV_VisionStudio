from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Generic base interface for ML/CV models."""

    def __init__(self) -> None:
        super().__init__()

    # -------------------------
    # Core forward / inference
    # -------------------------
    @abstractmethod
    def forward(
        self,
        inputs: Tensor,
        targets: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Main model computation."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, inputs: Tensor) -> dict[str, Any]:
        """Inference-only helper."""
        was_training = self.training
        self.eval()
        outputs = self.forward(inputs, targets=None)
        if was_training:
            self.train()
        return outputs

    # -------------------------
    # Training / evaluation
    # -------------------------
    @abstractmethod
    def compute_loss(
        self,
        outputs: dict[str, Any],
        targets: dict[str, Any],
    ) -> dict[str, Tensor]:
        """Return one or more losses."""
        raise NotImplementedError

    def training_step(
        self,
        batch: tuple[Tensor, dict[str, Any]],
    ) -> dict[str, Tensor]:
        """Run one training step on a batch."""
        inputs, targets = batch
        outputs = self.forward(inputs, targets=targets)
        losses = self.compute_loss(outputs, targets)
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[Tensor, dict[str, Any]],
    ) -> dict[str, Tensor]:
        """Run one validation step on a batch."""
        inputs, targets = batch
        was_training = self.training
        self.eval()
        outputs = self.forward(inputs, targets=targets)
        losses = self.compute_loss(outputs, targets)
        if was_training:
            self.train()
        return losses

    @torch.no_grad()
    def test_step(
        self,
        batch: tuple[Tensor, dict[str, Any]],
    ) -> dict[str, Tensor]:
        """Run one test step on a batch."""
        return self.validation_step(batch)

    # -------------------------
    # Checkpoint / weights
    # -------------------------
    def save_weights(self, path: str | Path) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), str(path))

    def load_weights(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> None:
        """Load model weights."""
        state_dict = torch.load(str(path), map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save training checkpoint."""
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, str(path))

    def load_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load training checkpoint and optionally optimizer state."""
        checkpoint = torch.load(str(path), map_location=map_location)

        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    # -------------------------
    # Export
    # -------------------------
    def export_torchscript(
        self,
        path: str | Path,
        example_inputs: Tensor,
    ) -> None:
        """Export model as TorchScript."""
        was_training = self.training
        self.eval()
        traced = torch.jit.trace(self, example_inputs)
        traced.save(str(path))
        if was_training:
            self.train()

    def export_onnx(
        self,
        path: str | Path,
        example_inputs: Tensor,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        opset_version: int = 17,
    ) -> None:
        """Export model to ONNX."""
        was_training = self.training
        self.eval()
        torch.onnx.export(
            self,
            example_inputs,
            str(path),
            input_names=input_names or ["inputs"],
            output_names=output_names or ["outputs"],
            opset_version=opset_version,
        )
        if was_training:
            self.train()

    # -------------------------
    # Metadata / config
    # -------------------------
    def get_config(self) -> dict[str, Any]:
        """Return model construction config if available."""
        return {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseModel:
        """Construct model from config."""
        return cls(**config)

    def summary(self) -> dict[str, Any]:
        """Return basic model info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": self.__class__.__name__,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "config": self.get_config(),
        }
