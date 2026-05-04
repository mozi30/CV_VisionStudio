"""
Type definitions for CV tasks.

Provides TypedDict contracts for different computer vision tasks.
Each task defines its own output formats while maintaining a common base.
"""

from typing import TypedDict
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


class ClassificationEvaluatorOutput(TypedDict, total=False):
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


class DetectionEvaluatorOutput(TypedDict, total=False):
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
    masks: Tensor  # [batch, height, width] for semantic; [num_masks, height, width] for instance
    class_ids: Tensor  # [batch] or [num_masks]


class SegmentationEvaluatorOutput(TypedDict, total=False):
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


class KeypointEvaluatorOutput(TypedDict, total=False):
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
