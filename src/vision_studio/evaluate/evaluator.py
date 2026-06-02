"""Evaluator interfaces and loop implementations."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from vision_studio.inference.simple import EnsembleConfig
from vision_studio.models.base import BaseModel
from vision_studio.reporting import BaseReporter, LoggingReporter
from vision_studio.types import ConfigurationError, EvaluatorOutput

Batch = tuple[Tensor, dict[str, Any]]


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
        models: list[BaseModel],
        dataset: Iterable[Batch],
        config: EnsembleConfig | None = None,
    ) -> dict[str, Any]:
        """Evaluate an ensemble over a dataset and return flat metrics plus metadata."""
        cfg = config or EnsembleConfig()
        self._validate_ensemble_config(cfg, len(models))

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

                for index, model in enumerate(models):
                    if index in failed_models:
                        continue
                    try:
                        logits = model(inputs)
                        losses = model.compute_loss(logits, moved_targets)
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
