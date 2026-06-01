"""Type definitions and shared settings for vision_studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import Tensor

# ============================================================================
# BASE SPECS (Common to all tasks)
# ============================================================================


class InputSpec(TypedDict, total=False):
    """Specification for model input."""

    shape: tuple[int, ...]  # e.g., (batch_size, channels, height, width)
    dtype: torch.dtype
    device: str | torch.device


class OutputSpec(TypedDict, total=False):
    """Specification for model output."""

    shape: tuple[int, ...]  # e.g., (batch_size, num_classes)
    dtype: torch.dtype


# ============================================================================
# BASE OUTPUTS (Common to all tasks)
# ============================================================================


class LossOutput(TypedDict):
    """Standard loss output required by all models."""

    loss: Tensor


class EvaluatorOutput(TypedDict):
    """Base Evaluator output; all evaluations must include loss."""

    loss: float


class PostprocessOutput(TypedDict):
    """Base postprocess output - all tasks must include logits."""

    logits: Tensor


# ============================================================================
# CLASSIFICATION OUTPUTS
# ============================================================================


class ClassificationPostprocessOutput(PostprocessOutput):
    """Classification task output from postprocess()."""

    probs: Tensor  # [batch, num_classes]
    labels: Tensor  # [batch]


class ClassificationEvaluatorOutput(EvaluatorOutput, total=False):
    """Classification metrics from evaluator.compute()."""

    accuracy: float
    precision_macro: float
    precision_micro: float
    recall_macro: float
    recall_micro: float
    f1_macro: float
    f1_micro: float
    top_1_accuracy: float
    top_5_accuracy: float


# ============================================================================
# DETECTION OUTPUTS
# ============================================================================


class DetectionPostprocessOutput(PostprocessOutput):
    """Object detection task output from postprocess()."""

    boxes: Tensor  # [num_detections, 4] in xyxy format
    scores: Tensor  # [num_detections] confidence scores
    labels: Tensor  # [num_detections] class indices


class DetectionEvaluatorOutput(EvaluatorOutput, total=False):
    """Detection metrics from evaluator.compute()."""

    ap: float  # Average precision @ IoU=0.50:0.95
    ap50: float  # Average precision @ IoU=0.50
    ap75: float  # Average precision @ IoU=0.75
    ar: float  # Average recall @ IoU=0.50:0.95
    ar_small: float  # Average recall for small objects
    ar_medium: float  # Average recall for medium objects
    ar_large: float  # Average recall for large objects


# ============================================================================
# SEGMENTATION OUTPUTS
# ============================================================================


class SegmentationPostprocessOutput(PostprocessOutput):
    """Semantic/instance segmentation task output from postprocess()."""

    masks: Tensor  # [batch, height, width] or [num_masks, height, width]
    class_ids: Tensor  # [batch] or [num_masks]


class SegmentationEvaluatorOutput(EvaluatorOutput, total=False):
    """Segmentation metrics from evaluator.compute()."""

    miou: float  # Mean Intersection over Union
    iou_per_class: dict[int, float]  # IoU for each class
    mean_acc: float  # Mean accuracy
    acc_per_class: dict[int, float]  # Accuracy per class
    dice: float  # Dice coefficient


# ============================================================================
# KEYPOINT DETECTION OUTPUTS
# ============================================================================


class KeypointPostprocessOutput(PostprocessOutput):
    """Keypoint detection task output from postprocess()."""

    keypoints: Tensor  # [batch, num_keypoints, 2] or [num_objects, num_keypoints, 2]
    scores: Tensor  # [batch, num_keypoints] or [num_objects, num_keypoints] confidence
    object_ids: Tensor  # [batch] or [num_objects] which object each keypoint belongs to


class KeypointEvaluatorOutput(EvaluatorOutput, total=False):
    """Keypoint detection metrics from evaluator.compute()."""

    oks: float  # Object Keypoint Similarity
    oks_per_keypoint: dict[int, float]  # OKS per keypoint
    ap: float  # AP @ OKS=0.50
    ap75: float  # AP @ OKS=0.75
    ar: float  # Average recall


# ============================================================================
# UTILITY TYPES
# ============================================================================


class TrainingOutput(TypedDict, total=False):
    """Output from training_step() / validation_step()."""

    loss: Tensor
    # Can include additional task-specific losses


class EvaluatorBatch(TypedDict):
    """Single batch input to evaluator.update()."""

    predictions: Tensor  # Task-specific format
    targets: Tensor  # Task-specific format
    loss: Tensor  # Batch loss value


@dataclass(slots=True)
class TrainerSettings:
    """Configuration for Trainer behavior and runtime policies."""

    epochs: int = 10
    checkpoint_path: Path | None = None
    save_latest_checkpoint: bool = True
    best_checkpoint_count: int = 3
    checkpoint_monitor: str | None = None
    checkpoint_mode: str = "min"
    early_stopping_enabled: bool = False
    early_stopping_monitor: str | None = None
    early_stopping_patience: int = 0
    dry_run_batch_validation: bool = False
    reporting_mode: str = "logging"
    log_every_n_steps: int = 10


@dataclass(slots=True)
class ReporterConfig:
    """Configuration values for reporter setup."""

    mode: str = "logging"
    project: str | None = None
    run_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckpointInfo:
    """Metadata describing a saved checkpoint artifact."""

    path: Path
    kind: str
    epoch: int
    global_step: int
    monitor_metric: str | None = None
    monitor_value: float | None = None
    rank: int | None = None


class VisionStudioError(Exception):
    """Base exception for vision_studio workflow errors."""


class ConfigurationError(VisionStudioError):
    """Raised when Trainer or Evaluator configuration is invalid."""


class ReportingError(VisionStudioError):
    """Raised when a reporter cannot be used safely."""


class EvaluationError(VisionStudioError):
    """Raised when evaluation cannot be completed."""


class CheckpointError(VisionStudioError):
    """Raised when checkpoint configuration or saving fails."""


# ============================================================================
# INFERENCE SETTINGS TYPES
# ============================================================================

AggregationMode = str
TiePolicy = str
FailurePolicy = str


@dataclass(slots=True)
class RealtimeInferenceConfig:
    """Configuration for realtime webcam inference runs."""

    source: int | str = 0
    frame_skip: int = 0
    record_output: bool = False
    output_path: Path | None = None
