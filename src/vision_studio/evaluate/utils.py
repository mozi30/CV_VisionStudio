from __future__ import annotations

from numbers import Number
from typing import Any, Mapping
from vision_studio.types import EvaluatorOutput
import numpy as np


def _format_value(value: Any) -> str:
	if isinstance(value, (np.floating, np.integer)):
		value = value.item()
	if isinstance(value, float):
		return f"{value:.4f}"
	if isinstance(value, Number):
		return str(value)
	return str(value)


def _print_metric(label: str, value: Any) -> None:
	print(f"{label}: {_format_value(value)}")


def _print_per_class(label: str, data: Any) -> None:
	if not data:
		print(f"{label}: (none)")
		return

	if isinstance(data, Mapping):
		items = sorted(data.items(), key=lambda item: item[0])
	elif isinstance(data, (list, tuple)):
		items = list(enumerate(data))
	else:
		print(f"{label}: {data}")
		return

	parts = [f"{key}={_format_value(value)}" for key, value in items]
	print(f"{label}: {', '.join(parts)}")


def _infer_metrics_type(metrics: Mapping[str, Any]) -> str:
	keys = set(metrics.keys())

	if "oks" in keys or "oks_per_keypoint" in keys:
		return "keypoint"
	if "miou" in keys or "iou_per_class" in keys or "mean_acc" in keys:
		return "segmentation"
	if "ap50" in keys or "ar_small" in keys or "ar_medium" in keys:
		return "detection"
	if "accuracy" in keys or "f1_macro" in keys or "precision_macro" in keys:
		return "classification"

	return "unknown"


def print_evaluation_metrics(metrics: EvaluatorOutput) -> None:
	metrics_dict = dict(metrics) if metrics else {}
	if not metrics_dict:
		print("No metrics to display.")
		return

	metric_type = _infer_metrics_type(metrics_dict)

	if metric_type == "classification":
		print("Classification metrics")
		for key in [
			"loss",
			"accuracy",
			"precision_macro",
			"precision_micro",
			"recall_macro",
			"recall_micro",
			"f1_macro",
			"f1_micro",
			"top_1_accuracy",
			"top_5_accuracy",
		]:
			if key in metrics_dict:
				_print_metric(key, metrics_dict[key])
		return

	if metric_type == "detection":
		print("Detection metrics")
		for key in [
			"loss",
			"ap",
			"ap50",
			"ap75",
			"ar",
			"ar_small",
			"ar_medium",
			"ar_large",
		]:
			if key in metrics_dict:
				_print_metric(key, metrics_dict[key])
		return

	if metric_type == "segmentation":
		print("Segmentation metrics")
		for key in ["loss", "miou", "mean_acc", "dice"]:
			if key in metrics_dict:
				_print_metric(key, metrics_dict[key])
		if "iou_per_class" in metrics_dict:
			_print_per_class("iou_per_class", metrics_dict["iou_per_class"])
		if "acc_per_class" in metrics_dict:
			_print_per_class("acc_per_class", metrics_dict["acc_per_class"])
		return

	if metric_type == "keypoint":
		print("Keypoint metrics")
		for key in ["loss", "oks", "ap", "ap75", "ar"]:
			if key in metrics_dict:
				_print_metric(key, metrics_dict[key])
		if "oks_per_keypoint" in metrics_dict:
			_print_per_class("oks_per_keypoint", metrics_dict["oks_per_keypoint"])
		return

	print("Metrics")
	for key, value in metrics_dict.items():
		if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
			_print_per_class(key, value)
		else:
			_print_metric(key, value)
