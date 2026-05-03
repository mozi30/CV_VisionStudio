from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import wandb
from torch import Tensor
from torch.nn import Module

from .base import Delivery


class WandbDelivery(Delivery):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        project: str = "my-model-delivery",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        artifact_name: str = "model",
        artifact_type: str = "model",
        export_torchscript_model: bool = False,
        export_onnx_model: bool = False,
        example_inputs: Tensor | None = None,
        use_wandb: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.export_torchscript_model = export_torchscript_model
        self.export_onnx_model = export_onnx_model
        self.example_inputs = example_inputs
        self.use_wandb = use_wandb
        self._wandb_initialized = False

    def deliver(
        self,
        model: Module,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb and not self._wandb_initialized:
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
            )
            self._wandb_initialized = True

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

        if self.use_wandb:
            artifact = wandb.Artifact(
                name=self.artifact_name,
                type=self.artifact_type,
                metadata=metadata,
            )
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    artifact.add_file(str(file_path))

            wandb.log_artifact(artifact)
            wandb.finish()

        return {
            "output_dir": str(output_dir),
            "artifacts": artifacts,
        }
