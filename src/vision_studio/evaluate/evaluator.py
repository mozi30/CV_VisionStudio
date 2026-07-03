"""Evaluator interfaces and loop implementations."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from vision_studio.augmentation.base import (
    Augmentation,
    TorchVisionAugmentation,
    apply_transform,
)
from vision_studio.inference.simple import EnsembleConfig
from vision_studio.models.base import BaseModel
from vision_studio.reporting import BaseReporter, LoggingReporter
from vision_studio.types import ConfigurationError, EvaluatorOutput

Batch = tuple[Tensor, dict[str, Any]]


@dataclass(slots=True)
class EnsembleMember:
    """One model participating in ensemble evaluation with optional preprocessing."""

    model: BaseModel
    augmentation: Callable[..., Any] | None = None
    name: str | None = None


class Evaluator(ABC):
    """Base interface for evaluation-loop implementations."""

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        dataset: Iterable[Batch],
    ) -> EvaluatorOutput:
        """Run evaluation for a model on a dataset and return metrics."""
        raise NotImplementedError


class LoopEvaluator(Evaluator):
    """Default evaluator that iterates a dataset and updates metrics."""

    def __init__(
        self,
        metrics: Any,
        device: torch.device | str = "cpu",
        reporter: BaseReporter | None = None,
    ) -> None:
        """Create a loop evaluator for the provided metrics implementation."""
        self.metrics = metrics
        self.device = torch.device(device)
        self.reporter = reporter or LoggingReporter()

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModel,
        dataset: Iterable[Batch],
    ) -> EvaluatorOutput:
        """Evaluate a model over a dataset and return aggregated metrics."""
        self.metrics.reset()
        self.reporter.start()
        was_training = model.training
        model.eval()

        for batch in dataset:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            moved_targets = self._move_targets(targets)

            logits = model(inputs)
            losses = model.compute_loss(logits, moved_targets)
            outputs = model.postprocess(logits)
            self.metrics.update(outputs, moved_targets, losses["loss"])

        if was_training:
            model.train()
        result = self.metrics.compute()
        self.reporter.log({"evaluation/loss": result["loss"]})
        self.reporter.finish()
        return result

    @torch.no_grad()
    def evaluate_ensemble(
        self,
        models: list[BaseModel | EnsembleMember],
        dataset: Iterable[Batch],
        config: EnsembleConfig | None = None,
    ) -> dict[str, Any]:
        """Evaluate an ensemble over a dataset and return flat metrics plus metadata."""
        members = [
            model if isinstance(model, EnsembleMember) else EnsembleMember(model=model)
            for model in models
        ]
        return self.evaluate_ensemble_members(members, dataset, config)

    @torch.no_grad()
    def evaluate_ensemble_members(
        self,
        members: list[EnsembleMember],
        dataset: Iterable[Batch],
        config: EnsembleConfig | None = None,
    ) -> dict[str, Any]:
        """Evaluate ensemble members with optional per-model augmentations."""
        cfg = config or EnsembleConfig()
        self._validate_ensemble_config(cfg, len(members))
        models = [member.model for member in members]

        self.metrics.reset()
        self.reporter.start()
        model_states = self._capture_model_states(models)
        failed_models: set[int] = set()

        try:
            for model in models:
                model.eval()

            for batch in dataset:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                moved_targets = self._move_targets(targets)

                per_model_preds: list[Tensor] = []
                successful_indexes: list[int] = []
                per_model_losses: list[Tensor] = []

                for index, member in enumerate(members):
                    if index in failed_models:
                        continue
                    try:
                        member_inputs, member_targets = self._member_batch(
                            batch, member.augmentation
                        )
                        member_inputs = member_inputs.to(self.device)
                        moved_member_targets = self._move_targets(member_targets)

                        model = member.model
                        logits = model(member_inputs)
                        losses = model.compute_loss(logits, moved_member_targets)
                        outputs = model.postprocess(logits)
                        pred = self._extract_prediction_tensor(outputs)
                        per_model_preds.append(pred.detach().cpu())
                        successful_indexes.append(index)
                        per_model_losses.append(losses["loss"].detach().cpu())
                    except Exception:
                        failed_models.add(index)
                        if cfg.failure_policy == "fail-fast":
                            raise

                if not per_model_preds:
                    raise ConfigurationError(
                        "No successful model outputs available for aggregation"
                    )

                predictions = self._aggregate_predictions(
                    per_model_preds,
                    successful_indexes,
                    cfg,
                )
                mean_loss = torch.stack(
                    [loss.float().reshape(()) for loss in per_model_losses]
                ).mean()
                self.metrics.update(predictions, moved_targets, mean_loss)

            result = dict(self.metrics.compute())
            failed = self._failed_model_list(failed_models)
            status = "completed_with_warnings" if failed else "completed"
            aggregation_metadata = {
                "mode": cfg.mode,
                "tie_policy": cfg.tie_policy,
                "failure_policy": cfg.failure_policy,
                "failed_models": failed,
                "model_count": len(models),
                "successful_model_count": len(models) - len(failed),
            }
            result.update(
                {
                    "status": status,
                    "aggregation_metadata": aggregation_metadata,
                    "failed_models": failed,
                }
            )
            self.reporter.log({"evaluation/loss": result["loss"]})
            return result
        finally:
            self._restore_model_states(models, model_states)
            self.reporter.finish()

    def prepare_ensemble_handoff(
        self, ensemble_output: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize unified inference ensemble output for evaluator consumers."""
        preds = ensemble_output.get("preds")
        metric_keys = {
            key: value
            for key, value in ensemble_output.items()
            if key
            not in {
                "preds",
                "aggregated",
                "metrics",
                "mode",
                "tie_policy",
                "failure_policy",
                "failed_models",
                "status",
                "aggregation_metadata",
            }
        }
        if preds is None and not metric_keys:
            raise ValueError(
                "Ensemble output must include 'preds' or metric values for evaluator handoff."
            )

        return {
            "preds": preds,
            "metrics": ensemble_output.get("metrics", metric_keys),
            "status": ensemble_output.get("status", "completed"),
            "aggregation_metadata": ensemble_output.get("aggregation_metadata", {}),
            "failed_models": ensemble_output.get("failed_models", []),
        }

    def _move_targets(self, targets: dict[str, Any]) -> dict[str, Any]:
        moved_targets: dict[str, Any] = {}
        for key, value in targets.items():
            moved_targets[key] = (
                value.to(self.device) if isinstance(value, Tensor) else value
            )
        return moved_targets

    @staticmethod
    def _capture_model_states(models: list[BaseModel]) -> list[bool]:
        return [model.training for model in models]

    @staticmethod
    def _restore_model_states(models: list[BaseModel], states: list[bool]) -> None:
        for model, was_training in zip(models, states, strict=False):
            model.train(was_training)

    @staticmethod
    def _failed_model_list(failed_models: set[int]) -> list[int]:
        return sorted(failed_models)

    def _member_batch(
        self,
        batch: Batch,
        augmentation: Callable[..., Any] | None,
    ) -> Batch:
        if augmentation is None:
            return batch

        inputs, targets = batch
        if inputs.ndim != 4:
            raise ConfigurationError(
                "Per-member augmentation requires batched image tensors with shape "
                "[batch, channels, height, width]."
            )

        images: list[Tensor] = []
        sample_targets: list[dict[str, Any]] = []
        for index, image in enumerate(inputs):
            sample_target = self._sample_target(targets, index)
            source_image: Tensor | np.ndarray
            if self._preserves_tensor_inputs(augmentation):
                source_image = image.detach().cpu()
            else:
                source_image = self._tensor_image_to_numpy(image)
            aug_image, aug_target = apply_transform(
                augmentation,
                source_image,
                sample_target,
            )
            images.append(self._augmentation_image_to_tensor(aug_image))
            sample_targets.append(aug_target)

        return torch.stack(images, dim=0), self._merge_sample_targets(sample_targets)

    @staticmethod
    def _preserves_tensor_inputs(augmentation: Callable[..., Any]) -> bool:
        if isinstance(augmentation, TorchVisionAugmentation):
            return True
        if not isinstance(augmentation, Augmentation):
            return True

        nested_transforms = getattr(augmentation, "transforms", None)
        if nested_transforms is None:
            return False
        return all(
            not isinstance(transform, Augmentation)
            or isinstance(transform, TorchVisionAugmentation)
            for transform in nested_transforms
        )

    @staticmethod
    def _sample_target(targets: dict[str, Any], index: int) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        for key, value in targets.items():
            if isinstance(value, Tensor):
                sample[key] = value[index] if value.ndim > 0 else value
            elif isinstance(value, np.ndarray):
                sample[key] = value[index] if value.ndim > 0 else value
            elif isinstance(value, (list, tuple)):
                sample[key] = value[index]
            else:
                sample[key] = value
        return sample

    @staticmethod
    def _merge_sample_targets(samples: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if not samples:
            return merged

        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if all(isinstance(value, Tensor) for value in values):
                merged[key] = torch.stack(
                    [value if value.ndim > 0 else value.reshape(()) for value in values]
                )
            elif all(isinstance(value, np.ndarray) for value in values):
                try:
                    merged[key] = np.stack(values, axis=0)
                except ValueError:
                    merged[key] = values
            else:
                merged[key] = values
        return merged

    @staticmethod
    def _tensor_image_to_numpy(image: Tensor) -> np.ndarray:
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ConfigurationError(
                "Per-member augmentation requires image tensors with shape "
                "[channels, height, width]."
            )
        array = image.permute(1, 2, 0).numpy()
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            min_value = float(np.nanmin(array)) if array.size else 0.0
            if 0.0 <= min_value and max_value <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            return array[..., 0]
        return array

    @staticmethod
    def _augmentation_image_to_tensor(image: Tensor | np.ndarray) -> Tensor:
        if isinstance(image, Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3 and tensor.shape[0] not in {1, 3, 4}:
                tensor = tensor.permute(2, 0, 1)
            return tensor.float()

        array = np.asarray(image)
        if array.ndim == 2:
            array = array[..., None]
        tensor = torch.as_tensor(array)
        if tensor.ndim != 3:
            raise ConfigurationError(
                "Per-member augmentation must return an image with 2 or 3 dimensions."
            )
        tensor = tensor.permute(2, 0, 1).contiguous().float()
        if np.issubdtype(array.dtype, np.integer):
            tensor = tensor / 255.0
        return tensor

    def _validate_ensemble_config(
        self,
        config: EnsembleConfig,
        model_count: int,
    ) -> None:
        if model_count == 0:
            raise ConfigurationError("At least one model is required for evaluation")
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
            if sum(config.weights) == 0:
                raise ConfigurationError("weights must not sum to zero")

    @staticmethod
    def _extract_prediction_tensor(outputs: Tensor | dict[str, Any]) -> Tensor:
        if isinstance(outputs, Tensor):
            return outputs
        if "logits" in outputs and isinstance(outputs["logits"], Tensor):
            return outputs["logits"]
        if "probs" in outputs and isinstance(outputs["probs"], Tensor):
            return outputs["probs"]
        raise ConfigurationError(
            "Model postprocess output must include Tensor 'logits' or 'probs'."
        )

    @staticmethod
    def _extract_num_classes(preds: Tensor) -> int:
        if preds.ndim == 1:
            return 1
        return int(preds.shape[-1])

    def _aggregate_predictions(
        self,
        per_model_preds: list[Tensor],
        successful_indexes: list[int],
        config: EnsembleConfig,
    ) -> Tensor:
        class_counts = {self._extract_num_classes(pred) for pred in per_model_preds}
        if len(class_counts) != 1:
            raise ConfigurationError(
                "All models must expose identical output class count"
            )
        class_count = class_counts.pop()

        shapes = {tuple(pred.shape) for pred in per_model_preds}
        if len(shapes) != 1:
            raise ConfigurationError(
                "All models must expose shape-compatible outputs for aggregation"
            )

        if config.mode == "soft":
            return torch.stack(per_model_preds, dim=0).float().mean(dim=0)

        if config.mode == "weighted":
            assert config.weights is not None
            selected_weights = [config.weights[index] for index in successful_indexes]
            weight_sum = sum(selected_weights)
            if weight_sum == 0:
                raise ConfigurationError(
                    "successful model weights must not sum to zero"
                )
            weights = torch.tensor(selected_weights, dtype=torch.float32).view(
                -1,
                *([1] * per_model_preds[0].ndim),
            )
            stacked = torch.stack(per_model_preds, dim=0).float()
            return (stacked * weights).sum(dim=0) / weights.sum()

        votes = []
        for pred in per_model_preds:
            if pred.ndim > 1:
                votes.append(torch.argmax(pred, dim=-1))
            else:
                votes.append((pred > 0.5).long())
        vote_tensor = torch.stack(votes, dim=0)

        labels = []
        for col in vote_tensor.T:
            uniq, counts = torch.unique(col, return_counts=True)
            max_count = counts.max()
            tied = uniq[counts == max_count]
            if tied.numel() == 1:
                labels.append(tied[0])
            elif config.tie_policy == "index-priority":
                labels.append(col[0])
            else:
                labels.append(tied[random.randrange(0, tied.numel())])

        label_tensor = torch.stack(labels, dim=0).long()
        if class_count <= 1:
            return label_tensor.float()
        return F.one_hot(label_tensor, num_classes=class_count).float()
