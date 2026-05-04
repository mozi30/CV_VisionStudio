from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..types import InputSpec, OutputSpec, LossOutput, PostprocessOutput


class BaseModel(nn.Module, ABC):
    """Strict base interface for ML/CV models.

    Key design principles:
    - forward() returns ONLY logits/raw outputs (Tensor) for speed
    - postprocess() handles task-specific postprocessing (probabilities, labels, etc.)
    - Input/output specs are explicitly defined
    - Strict separation between forward pass and postprocessing
    """

    # -------------------------
    # Input/Output specifications
    # -------------------------
    @property
    @abstractmethod
    def input_spec(self) -> InputSpec:
        """Define expected input shape, dtype, and device."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_spec(self) -> OutputSpec:
        """Define output shape, dtype from forward()."""
        raise NotImplementedError

    # -------------------------
    # Core forward / inference
    # -------------------------
    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Main model computation - returns ONLY raw logits/outputs.

        Args:
            inputs: Input tensor matching input_spec

        Returns:
            Raw model outputs (logits) as a single Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, logits: Tensor) -> PostprocessOutput:
        """
        Task-specific postprocessing of raw outputs.

        Converts raw logits to task-specific outputs (probs, labels, etc.)
        This is called during inference and evaluation.

        Args:
            logits: Raw model output from forward()

        Returns:
            Dictionary with task-specific processed outputs
        """
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, inputs: Tensor) -> PostprocessOutput:
        """Inference-only helper: forward + postprocess."""
        was_training = self.training
        self.eval()
        logits = self.forward(inputs)
        outputs = self.postprocess(logits)
        if was_training:
            self.train()
        return outputs

    # -------------------------
    # Training / evaluation
    # -------------------------
    @abstractmethod
    def compute_loss(
        self,
        logits: Tensor,
        targets: dict[str, Any],
    ) -> LossOutput:
        """Return loss output with strict type contract."""
        raise NotImplementedError

    def training_step(
        self,
        batch: tuple[Tensor, dict[str, Any]],
    ) -> LossOutput:
        """Run one training step on a batch."""
        inputs, targets = batch
        logits = self.forward(inputs)
        losses = self.compute_loss(logits, targets)
        return losses

    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[Tensor, dict[str, Any]],
    ) -> LossOutput:
        """Run one validation step on a batch."""
        inputs, targets = batch
        was_training = self.training
        self.eval()
        logits = self.forward(inputs)
        losses = self.compute_loss(logits, targets)
        if was_training:
            self.train()
        return losses

    @torch.no_grad()
    def test_step(
        self,
        batch: tuple[Tensor, dict[str, Any]],
    ) -> LossOutput:
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
        traced = torch.jit.trace(self, (example_inputs,))
        torch.jit.save(traced, str(path))
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
            (example_inputs,),
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
