from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module


class Delivery(ABC):
    def __init__(
        self,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)

    @abstractmethod
    def deliver(
        self,
        model: Module,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def save_model_state(
        self,
        model: Module,
        path: str | Path,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        return path

    def save_full_checkpoint(
        self,
        path: str | Path,
        model: Module,
        optimizer_state_dict: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict
        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        return path

    def save_metadata(
        self,
        path: str | Path,
        metadata: dict[str, Any],
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return path

    def export_torchscript(
        self,
        model: Module,
        path: str | Path,
        example_inputs: Tensor,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            traced = torch.jit.trace(model, example_inputs.to(self.device))
            traced.save(str(path))

        return path

    def export_onnx(
        self,
        model: Module,
        path: str | Path,
        example_inputs: Tensor,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            torch.onnx.export(
                model,
                example_inputs.to(self.device),
                str(path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names or ["inputs"],
                output_names=output_names or ["outputs"],
                dynamic_axes=dynamic_axes,
            )

        return path
