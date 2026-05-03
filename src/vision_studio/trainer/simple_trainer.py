from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer

from .base import Trainer


class SimpleTrainer(Trainer):
    def __init__(
        self,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(optimizer=optimizer, device=device)

    def fit(
        self,
        model,
        train_loader,
        val_loader=None,
        num_epochs: int = 1,
    ) -> dict[str, Any]:
        model.to(self.device)

        history: dict[str, list[dict[str, float]]] = {
            "train": [],
            "val": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(model, train_loader)
            history["train"].append(train_metrics)

            if val_loader is not None:
                val_metrics = self.validate(model, val_loader)
                history["val"].append(val_metrics)

        return history

    def train_epoch(
        self,
        model,
        train_loader,
    ) -> dict[str, float]:
        model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs, targets = self.move_batch_to_device(batch)

            self.optimizer.zero_grad()

            losses = model.training_step((inputs, targets))
            loss = losses["loss"]

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            self.global_step += 1

        return {
            "loss": total_loss / max(1, num_batches),
        }

    @torch.no_grad()
    def validate(
        self,
        model,
        val_loader,
    ) -> dict[str, float]:
        model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            inputs, targets = self.move_batch_to_device(batch)
            losses = model.validation_step((inputs, targets))
            loss = losses["loss"]

            total_loss += float(loss.item())
            num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
        }
