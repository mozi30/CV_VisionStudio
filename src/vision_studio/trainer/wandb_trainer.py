from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from tqdm import tqdm
import torch
import wandb
from torch import Tensor
from torch.nn import Module

from vision_studio.models.base import BaseModel
from ..types import LossOutput

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
        self.evaluator = evaluator
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
            wandb.watch(model, log="all", log_freq=self.log_every_n_steps)
            self._wandb_initialized = True

        history: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
        }

        best_val_loss = float("inf")

        for epoch in range(self.current_epoch, self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.current_epoch = epoch

            train_metrics = self.train_epoch(model, train_loader)
            history["train"].append(train_metrics)

            log_data: dict[str, float | int] = {
                f"train/{k}": v for k, v in train_metrics.items()
            }
            log_data["epoch"] = epoch

            if val_loader is not None:
                val_metrics = self.validate(model, val_loader)
                history["val"].append(val_metrics)
                log_data.update({f"val/{k}": v for k, v in val_metrics.items()})

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
    ) -> dict[str, Any]:
        model.train()
        totals: dict[str, float] = defaultdict(float)
        num_batches = 0

        # Reset evaluator if available
        if self.evaluator is not None:
            self.evaluator.reset()

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

            for key, value in metrics.items():
                totals[key] += self._to_float(value)
            self.global_step += 1
            num_batches += 1

            # Collect predictions for evaluator
            if self.evaluator is not None:
                with torch.no_grad():
                    outputs = model.postprocess(logits)
                    preds = outputs["logits"].detach().cpu()
                    labels = targets["label"].detach().cpu()
                    self.evaluator.update(preds, labels)

            if self.use_wandb and batch_idx % self.log_every_n_steps == 0:
                wandb.log(
                    {
                        **{f"train_step/{k}": self._to_float(v) for k, v in metrics.items()},
                        "epoch": self.current_epoch,
                    },
                    step=self.global_step,
                )

        # Add evaluator metrics
        averaged_metrics = self._average_metrics(totals, num_batches)
        if self.evaluator is not None:
            evaluator_metrics = self.evaluator.compute()
            averaged_metrics.update(evaluator_metrics)

        return averaged_metrics

    @torch.no_grad()
    def validate(
        self,
        model: BaseModel,
        val_loader: Iterable[Batch],
    ) -> dict[str, Any]:
        model.eval()
        totals: dict[str, float] = defaultdict(float)
        num_batches = 0

        # Reset evaluator if available
        if self.evaluator is not None:
            self.evaluator.reset()

        for batch in val_loader:
            inputs, targets = self.move_batch_to_device(batch)
            logits = model(inputs)

            metrics = self._parse_model_output(model.compute_loss(logits, targets))

            for key, value in metrics.items():
                totals[key] += self._to_float(value)

            num_batches += 1

            # Collect predictions for evaluator
            if self.evaluator is not None:
                outputs = model.postprocess(logits)
                preds = outputs["logits"].detach().cpu()
                labels = targets["label"].detach().cpu()
                self.evaluator.update(preds, labels)

        averaged_metrics = self._average_metrics(totals, num_batches)
        if self.evaluator is not None:
            evaluator_metrics = self.evaluator.compute()
            averaged_metrics.update(evaluator_metrics)

        return averaged_metrics

    def _to_float(self, value: Any) -> float:
        if isinstance(value, Tensor):
            if value.numel() == 1:
                return float(value.item())
            return float(value.detach().float().mean().item())
        return float(value)

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

    def _average_metrics(
        self,
        totals: dict[str, float],
        num_batches: int,
    ) -> dict[str, float]:
        if num_batches == 0:
            return {k: 0.0 for k in totals}
        return {k: v / num_batches for k, v in totals.items()}
