"""Unified inference implementation for single-model, ensemble, and stream modes."""

# ruff: noqa: D101,D102,D107

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from vision_studio.types import (
    ConfigurationError,
    FailurePolicy,
    RealtimeInferenceConfig,
    TiePolicy,
)

from .base import Batch, Inference


@dataclass(slots=True)
class EnsembleConfig:
    mode: str = "soft"
    tie_policy: TiePolicy = "index-priority"
    failure_policy: FailurePolicy = "fail-fast"
    weights: list[float] | None = None


class SimpleInference(Inference):
    @staticmethod
    def _extract_num_classes(preds: Tensor) -> int:
        if preds.ndim == 1:
            return 1
        return int(preds.shape[-1])

    def run_ensemble(
        self,
        models: list[Module],
        batch: Batch,
        config: EnsembleConfig | None = None,
    ) -> dict[str, Any]:
        self.validate_non_empty_models(models)
        cfg = config or EnsembleConfig()
        self._validate_ensemble_config(cfg, len(models))

        per_model_preds: list[Tensor] = []
        failed_models: list[int] = []

        for i, model in enumerate(models):
            try:
                out = self.predict_batch(model, batch)
                preds = out["preds"]
                if not isinstance(preds, Tensor):
                    raise ConfigurationError("preds must be Tensor")
                per_model_preds.append(preds.detach().cpu())
            except Exception:
                failed_models.append(i)
                if cfg.failure_policy == "fail-fast":
                    raise

        if not per_model_preds:
            raise ConfigurationError(
                "No successful model outputs available for aggregation"
            )

        class_counts = {self._extract_num_classes(p) for p in per_model_preds}
        if len(class_counts) != 1:
            raise ConfigurationError(
                "All models must expose identical output class count"
            )

        if cfg.mode == "soft":
            stacked = torch.stack(per_model_preds, dim=0)
            agg = stacked.float().mean(dim=0)
            labels = torch.argmax(agg, dim=-1) if agg.ndim > 1 else (agg > 0.5).long()
        elif cfg.mode == "weighted":
            assert cfg.weights is not None
            w = torch.tensor(cfg.weights, dtype=torch.float32).view(
                -1, *([1] * per_model_preds[0].ndim)
            )
            stacked = torch.stack(per_model_preds, dim=0).float()
            agg = (stacked * w).sum(dim=0) / w.sum()
            labels = torch.argmax(agg, dim=-1) if agg.ndim > 1 else (agg > 0.5).long()
        else:  # hard
            votes = []
            for p in per_model_preds:
                if p.ndim > 1:
                    votes.append(torch.argmax(p, dim=-1))
                else:
                    votes.append((p > 0.5).long())
            vote_tensor = torch.stack(votes, dim=0)
            labels = []
            for col in vote_tensor.T:
                uniq, counts = torch.unique(col, return_counts=True)
                max_count = counts.max()
                tied = uniq[counts == max_count]
                if tied.numel() == 1:
                    labels.append(tied[0])
                elif cfg.tie_policy == "index-priority":
                    labels.append(col[0])
                else:
                    labels.append(tied[random.randrange(0, tied.numel())])
            labels = torch.stack(labels, dim=0)
            agg = labels

        status = "completed_with_warnings" if failed_models else "completed"
        aggregation_metadata = {
            "mode": cfg.mode,
            "tie_policy": cfg.tie_policy,
            "failure_policy": cfg.failure_policy,
            "failed_models": failed_models,
            "model_count": len(models),
            "successful_model_count": len(per_model_preds),
        }
        return {
            "preds": labels,
            "aggregated": agg,
            "metrics": {},
            "mode": cfg.mode,
            "tie_policy": cfg.tie_policy,
            "failure_policy": cfg.failure_policy,
            "failed_models": failed_models,
            "status": status,
            "aggregation_metadata": aggregation_metadata,
        }

    def _validate_ensemble_config(
        self, config: EnsembleConfig, model_count: int
    ) -> None:
        if config.mode not in {"soft", "hard", "weighted"}:
            raise ConfigurationError(f"Unsupported aggregation mode: {config.mode}")
        if config.tie_policy not in {"index-priority", "random"}:
            raise ConfigurationError(f"Unsupported tie policy: {config.tie_policy}")
        if config.failure_policy not in {"fail-fast", "continue-with-warning"}:
            raise ConfigurationError(
                f"Unsupported failure policy: {config.failure_policy}"
            )
        if config.mode == "weighted":
            if not config.weights:
                raise ConfigurationError("weights are required for weighted mode")
            if len(config.weights) != model_count:
                raise ConfigurationError("weights length must match model count")

    @torch.no_grad()
    def predict(
        self,
        model: Module,
        data_loader: Iterable[Batch],
    ) -> dict[str, Any]:
        model.to(self.device)
        model.eval()

        all_preds: list[Tensor] = []
        all_targets: dict[str, list[Any]] = defaultdict(list)
        metric_totals: dict[str, float] = defaultdict(float)
        metric_counts: dict[str, int] = defaultdict(int)

        for batch in data_loader:
            batch_output = self.predict_batch(model, batch)

            preds = batch_output["preds"]
            metrics = batch_output.get("metrics", {})
            targets = batch_output.get("targets", {})

            if isinstance(preds, Tensor):
                all_preds.append(preds.detach().cpu())
            else:
                raise TypeError("'preds' must be a Tensor.")

            for key, value in targets.items():
                if isinstance(value, Tensor):
                    all_targets[key].append(value.detach().cpu())
                else:
                    all_targets[key].append(value)

            for key, value in metrics.items():
                metric_totals[key] += float(value)
                metric_counts[key] += 1

        merged_targets: dict[str, Any] = {}
        for key, values in all_targets.items():
            if values and all(isinstance(v, Tensor) for v in values):
                merged_targets[key] = torch.cat(values, dim=0)
            else:
                merged_targets[key] = values

        averaged_metrics = {
            key: metric_totals[key] / metric_counts[key]
            for key in metric_totals
            if metric_counts[key] > 0
        }

        return {
            "preds": torch.cat(all_preds, dim=0) if all_preds else torch.empty(0),
            "targets": merged_targets,
            "metrics": averaged_metrics,
        }

    @torch.no_grad()
    def predict_image(
        self,
        model: Module,
        batch: Batch,
    ) -> dict[str, Any]:
        result = self.predict_batch(model, batch)
        return {
            "preds": result["preds"],
            "targets": result.get("targets", {}),
            "metrics": {},
            "mode": "image",
        }

    @torch.no_grad()
    def predict_video_stream(
        self,
        model: Module,
        frames: Iterable[Batch],
    ) -> dict[str, Any]:
        outputs: list[Tensor] = []
        for batch in frames:
            out = self.predict_batch(model, batch)
            preds = out["preds"]
            if not isinstance(preds, Tensor):
                raise ConfigurationError("preds must be Tensor in video mode")
            outputs.append(preds.detach().cpu())
        return {
            "preds": torch.cat(outputs, dim=0) if outputs else torch.empty(0),
            "metrics": {},
            "mode": "video",
        }

    @torch.no_grad()
    def predict_webcam_stream(
        self,
        model: Module,
        frames: Iterable[Batch],
        config: RealtimeInferenceConfig,
    ) -> dict[str, Any]:
        self.validate_webcam_source(config.source)
        self.validate_frame_skip(config.frame_skip)
        outputs: list[Tensor] = []
        for idx, batch in enumerate(frames):
            if config.frame_skip and idx % (config.frame_skip + 1) != 0:
                continue
            out = self.predict_batch(model, batch)
            preds = out["preds"]
            if not isinstance(preds, Tensor):
                raise ConfigurationError("preds must be Tensor in webcam mode")
            outputs.append(preds.detach().cpu())

        artifact: str | None = None
        if config.record_output:
            path = config.output_path or Path("webcam_annotated_output.pt")
            torch.save(
                {"preds": torch.cat(outputs, dim=0) if outputs else torch.empty(0)},
                path,
            )
            artifact = str(path)

        return {
            "preds": torch.cat(outputs, dim=0) if outputs else torch.empty(0),
            "metrics": {},
            "mode": "webcam",
            "artifact": artifact,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        model: Module,
        batch: Batch,
    ) -> dict[str, Any]:
        inputs, targets = self.move_batch_to_device(batch)
        logits = model(inputs)

        # Call postprocess if the model has it
        postprocess = getattr(model, "postprocess", None)
        if callable(postprocess):
            output = postprocess(logits)
            if isinstance(output, dict):
                preds = output.get("labels", logits)
            else:
                preds = logits
        else:
            preds = logits

        return {
            "preds": preds,
            "targets": targets,
            "metrics": {},
        }
