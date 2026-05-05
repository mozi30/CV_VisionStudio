from __future__ import annotations

from collections.abc import Iterable
from numbers import Number
from typing import Any
from tqdm import tqdm
import torch
import wandb
from torch import Tensor

from vision_studio.models.base import BaseModel
from vision_studio.types import EvaluatorOutput
from ..types import LossOutput
from vision_studio.evaluate import LossEvaluator

from .base import Trainer

Batch = tuple[Tensor, dict[str, Any]]


class WandbTrainer(Trainer):
    """WandB trainer with built-in evaluator support.

    Automatically logs metrics to Weights & Biases while training.
    Can optionally integrate a task-specific evaluator for comprehensive metrics.
    """

    def __init__(
        self,
        optimizer,
        evaluator=None,
        device: torch.device | str = "cpu",
        epochs: int = 10,
        project: str = "my-project",
        run_name: str | None = None,
        log_every_n_steps: int = 10,
        config: dict[str, Any] | None = None,
        use_wandb: bool = True,
    ) -> None:
        """Initialize WandB Trainer.

        Args:
            optimizer: PyTorch optimizer
            evaluator: Optional evaluator for computing task-specific metrics
            device: Device to use for training (default: "cpu")
            epochs: Number of epochs to train
            project: WandB project name
            run_name: WandB run name
            log_every_n_steps: Log to WandB every N steps
            config: Configuration dict to log to WandB
            use_wandb: Whether to use WandB logging (default: True)
        """
        super().__init__(optimizer=optimizer, device=device)
        self.epochs = epochs
        if evaluator is not None:
            self.evaluator = evaluator
        else:
            self.evaluator = LossEvaluator()
        self.project = project
        self.run_name = run_name
        self.log_every_n_steps = log_every_n_steps
        self.config = config or {}
        self.use_wandb = use_wandb
        self._wandb_initialized = False

    def fit(
        self,
        model: BaseModel,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
    ) -> dict[str, Any]:
        model.to(self.device)

        if self.use_wandb and not self._wandb_initialized:
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
            )
            self._wandb_initialized = True

        history: dict[str, list[EvaluatorOutput]] = {
            "train": [],
            "val": [],
        }

        best_val_loss = float("inf")

        for epoch in range(self.current_epoch, self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.current_epoch = epoch

            train_metrics = self.train_epoch(model, train_loader)
            history["train"].append(train_metrics)

            log_data: dict[str, float | int] = self._format_wandb_metrics(
                "train/",
                train_metrics,
            )
            log_data["epoch"] = epoch

            if val_loader is not None:
                val_metrics = self.validate(model, val_loader)
                history["val"].append(val_metrics)
                log_data.update(self._format_wandb_metrics("val/", val_metrics))

                val_loss = val_metrics.get("loss")
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss

            if self.use_wandb:
                wandb.log(log_data, step=self.global_step)

        if self.use_wandb:
            wandb.finish()

        return {
            "history": history,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }

    def train_epoch(
        self,
        model: BaseModel,
        train_loader: Iterable[Batch],
    ) -> EvaluatorOutput:
        model.train()

        loss_sum = 0.0
        loss_count = 0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            inputs, targets = self.move_batch_to_device(batch)

            self.optimizer.zero_grad()
            logits = model(inputs)
            metrics_dict = model.compute_loss(logits, targets)
            metrics = self._parse_model_output(metrics_dict)
            loss = metrics["loss"]
            if not isinstance(loss, Tensor):
                raise TypeError("Training loss must be a tensor.")
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            batch_size = int(inputs.shape[0]) if hasattr(inputs, "shape") else 1
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            loss_count += batch_size

            if self.use_wandb and batch_idx % self.log_every_n_steps == 0:
                wandb.log(
                    {
                        "train_step/loss": self._to_float(loss),
                        "epoch": self.current_epoch,
                    },
                    step=self.global_step,
                )
        return {
            "loss": loss_sum / loss_count if loss_count else 0.0,
        }


    @torch.no_grad()
    def validate(
        self,
        model: BaseModel,
        val_loader: Iterable[Batch],
    ) -> EvaluatorOutput:
        model.eval()
        num_batches = 0
        if self.evaluator is None:
            raise Exception("Cannot validate model without evaluator.")
        else:
            # Reset evaluator if available
            self.evaluator.reset()

        for batch in val_loader:
            inputs, targets = self.move_batch_to_device(batch)
            logits = model(inputs)
            metrics_dict = model.compute_loss(logits, targets)
            metrics = self._parse_model_output(metrics_dict)
            loss = metrics["loss"]

            num_batches += 1

            # Collect predictions for evaluator
            if self.evaluator is not None:
                if isinstance(self.evaluator, LossEvaluator):
                    self.evaluator.update(logits, targets, loss)
                else:
                    outputs = model.postprocess(logits)
                    preds = outputs["logits"].detach().cpu()
                    labels = targets["label"].detach().cpu()
                    self.evaluator.update(preds, labels, loss)
        return self.evaluator.compute()


    def _to_float(self, value: Any) -> float:
        if isinstance(value, Tensor):
            if value.numel() == 1:
                return float(value.item())
            return float(value.detach().float().mean().item())
        return float(value)

    def _format_wandb_metrics(
        self,
        prefix: str,
        metrics: EvaluatorOutput,
    ) -> dict[str, float]:
        formatted: dict[str, float] = {}
        for key, value in metrics.items():
            metric_key = f"{prefix}{key}"
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (Number, Tensor)):
                        formatted[f"{metric_key}/{sub_key}"] = self._to_float(
                            sub_value
                        )
                continue

            if isinstance(value, (Number, Tensor)):
                formatted[metric_key] = self._to_float(value)

        return formatted

    def _parse_model_output(self, output: LossOutput) -> dict[str, Tensor]:
        """Parse model loss output to extract loss tensor.

        Args:
            output: LossOutput from model.compute_loss()

        Returns:
            Dictionary with loss tensor
        """
        if "loss" not in output:
            raise ValueError("Model output dict must contain a 'loss' key.")
        return {"loss": output["loss"]}
