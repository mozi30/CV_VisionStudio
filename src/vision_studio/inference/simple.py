from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from .base import Batch, Inference


class SimpleInference(Inference):
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
    def predict_batch(
        self,
        model: Module,
        batch: Batch,
    ) -> dict[str, Any]:
        inputs, targets = self.move_batch_to_device(batch)
        output = model(inputs, targets)

        if isinstance(output, Tensor):
            return {
                "preds": output,
                "targets": targets,
                "metrics": {},
            }

        if isinstance(output, dict):
            if "preds" not in output:
                raise ValueError("Model output dict must contain a 'preds' key.")
            return {
                "preds": output["preds"],
                "targets": targets,
                "metrics": output.get("metrics", {}),
            }

        raise TypeError(
            "Model output must be either a Tensor or a dict containing 'preds'."
        )
