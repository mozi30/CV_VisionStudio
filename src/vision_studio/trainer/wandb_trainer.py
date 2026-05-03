from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import wandb
from torch import Tensor
from torch.nn import Module

from .base import Trainer

Batch = tuple[Tensor, dict[str, Any]]


class WandbTrainer(Trainer):
    def __init__(
        self,
        optimizer,
        device: torch.device | str = "cpu",
        epochs: int = 10,
        project: str = "my-project",
        run_name: str | None = None,
        log_every_n_steps: int = 10,
        config: dict[str, Any] | None = None,
        use_wandb: bool = True,
    ) -> None:
        super().__init__(optimizer=optimizer, device=device)
        self.epochs = epochs
        self.project = project
        self.run_name = run_name
        self.log_every_n_steps = log_every_n_steps
        self.config = config or {}
        self.use_wandb = use_wandb
        self._wandb_initialized = False

    def fit(
        self,
        model: Module,
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

        history: dict[str, list[dict[str, float]]] = {
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
        model: Module,
        train_loader: Iterable[Batch],
    ) -> dict[str, float]:
        model.train()
        totals: dict[str, float] = defaultdict(float)
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = self.move_batch_to_device(batch)

            self.optimizer.zero_grad()
            output = model(inputs, targets)
            output = model.compute_loss(output, targets)
            metrics = self._parse_model_output(output)
            loss = metrics["loss"]

            loss.backward()
            self.optimizer.step()

            for key, value in metrics.items():
                totals[key] += float(value)

            self.global_step += 1
            num_batches += 1

            if self.use_wandb and batch_idx % self.log_every_n_steps == 0:
                wandb.log(
                    {
                        **{f"train_step/{k}": float(v) for k, v in metrics.items()},
                        "epoch": self.current_epoch,
                    },
                    step=self.global_step,
                )

        return self._average_metrics(totals, num_batches)

    @torch.no_grad()
    def validate(
        self,
        model: Module,
        val_loader: Iterable[Batch],
    ) -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = defaultdict(float)
        num_batches = 0

        for batch in val_loader:
            inputs, targets = self.move_batch_to_device(batch)
            output = model(inputs, targets)
            metrics = self._parse_model_output(output)

            for key, value in metrics.items():
                totals[key] += float(value)

            num_batches += 1

        return self._average_metrics(totals, num_batches)

    def _parse_model_output(self, output: Any) -> dict[str, float | Tensor]:
        if isinstance(output, Tensor):
            return {"loss": output}

        if isinstance(output, dict):
            if "loss" not in output:
                raise ValueError("Model output dict must contain a 'loss' key.")
            return output

        raise TypeError(
            "Model output must be either a Tensor or a dict containing 'loss'."
        )

    def _average_metrics(
        self,
        totals: dict[str, float],
        num_batches: int,
    ) -> dict[str, float]:
        if num_batches == 0:
            return {k: 0.0 for k in totals}
        return {k: v / num_batches for k, v in totals.items()}
