from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from .base import Delivery


class SimpleDelivery(Delivery):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        export_torchscript_model: bool = False,
        export_onnx_model: bool = False,
        example_inputs: Tensor | None = None,
    ) -> None:
        super().__init__(device=device)
        self.export_torchscript_model = export_torchscript_model
        self.export_onnx_model = export_onnx_model
        self.example_inputs = example_inputs

    def deliver(
        self,
        model: Module,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Any] = {}

        state_path = self.save_model_state(model, output_dir / "model.pt")
        artifacts["model_state_dict"] = str(state_path)

        metadata = metadata or {}
        metadata_path = self.save_metadata(output_dir / "metadata.json", metadata)
        artifacts["metadata"] = str(metadata_path)

        if self.export_torchscript_model:
            if self.example_inputs is None:
                raise ValueError("example_inputs is required for TorchScript export.")
            ts_path = self.export_torchscript(
                model=model,
                path=output_dir / "model.ts",
                example_inputs=self.example_inputs,
            )
            artifacts["torchscript"] = str(ts_path)

        if self.export_onnx_model:
            if self.example_inputs is None:
                raise ValueError("example_inputs is required for ONNX export.")
            onnx_path = self.export_onnx(
                model=model,
                path=output_dir / "model.onnx",
                example_inputs=self.example_inputs,
                dynamic_axes={
                    "inputs": {0: "batch_size"},
                    "outputs": {0: "batch_size"},
                },
            )
            artifacts["onnx"] = str(onnx_path)

        return {
            "output_dir": str(output_dir),
            "artifacts": artifacts,
        }
