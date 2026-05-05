from typing import Optional, List, Dict
import torch
from .base import BaseEvaluator


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: Tensor[N, 4]
    boxes2: Tensor[M, 4]
    Format: xyxy
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - intersection

    return intersection / (union + 1e-8)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def compute_average_precision(
    precisions: torch.Tensor,
    recalls: torch.Tensor,
) -> float:
    """
    COCO/VOC-style interpolated AP.
    """
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.maximum(precisions[i], precisions[i + 1])

    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )

    return ap.item()


class DetectionEvaluator(BaseEvaluator):
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [
            0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.95,
        ]
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.predictions = []
        self.targets = []

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        loss: torch.Tensor | float,
    ) -> None:
        self.update_loss(loss, len(targets))
        for pred, target in zip(predictions, targets):
            self.predictions.append({
                "boxes": pred["boxes"].detach().cpu(),
                "scores": pred["scores"].detach().cpu(),
                "labels": pred["labels"].detach().cpu(),
            })
            self.targets.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            })

    def compute(self) -> Dict[str, float]:
        metrics = self.base_metrics()

        ap_per_threshold = []

        for iou_thr in self.iou_thresholds:
            ap_per_class = []

            for class_id in range(self.num_classes):
                ap = self._compute_ap_for_class(class_id, iou_thr)
                if ap is not None:
                    ap_per_class.append(ap)

            mean_ap = (
                sum(ap_per_class) / len(ap_per_class)
                if ap_per_class else 0.0
            )

            metrics[f"mAP@{iou_thr:.2f}"] = mean_ap
            ap_per_threshold.append(mean_ap)

        metrics["mAP@[0.50:0.95]"] = sum(ap_per_threshold) / len(ap_per_threshold)
        metrics["AP50"] = metrics.get("mAP@0.50", 0.0)
        metrics["AP75"] = metrics.get("mAP@0.75", 0.0)

        precision, recall, mean_iou = self._compute_global_precision_recall_iou(
            iou_threshold=0.50
        )

        metrics["precision@0.50"] = precision
        metrics["recall@0.50"] = recall
        metrics["mean_iou@0.50"] = mean_iou

        for class_id in range(self.num_classes):
            ap50 = self._compute_ap_for_class(class_id, 0.50)
            if ap50 is not None:
                metrics[f"class_{class_id}_AP50"] = ap50

        return metrics

    def _compute_ap_for_class(
        self,
        class_id: int,
        iou_threshold: float,
    ) -> Optional[float]:
        detections = []
        ground_truths = {}

        total_gt = 0

        for image_id, target in enumerate(self.targets):
            gt_mask = target["labels"] == class_id
            gt_boxes = target["boxes"][gt_mask]

            ground_truths[image_id] = {
                "boxes": gt_boxes,
                "matched": torch.zeros(len(gt_boxes), dtype=torch.bool),
            }

            total_gt += len(gt_boxes)

        if total_gt == 0:
            return None

        for image_id, pred in enumerate(self.predictions):
            pred_mask = pred["labels"] == class_id

            boxes = pred["boxes"][pred_mask]
            scores = pred["scores"][pred_mask]

            for box, score in zip(boxes, scores):
                detections.append({
                    "image_id": image_id,
                    "box": box,
                    "score": score.item(),
                })

        detections.sort(key=lambda x: x["score"], reverse=True)

        tp = torch.zeros(len(detections))
        fp = torch.zeros(len(detections))

        for idx, detection in enumerate(detections):
            image_id = detection["image_id"]
            pred_box = detection["box"].unsqueeze(0)

            gt_data = ground_truths[image_id]
            gt_boxes = gt_data["boxes"]

            if len(gt_boxes) == 0:
                fp[idx] = 1
                continue

            ious = box_iou(pred_box, gt_boxes).squeeze(0)
            best_iou, best_gt_idx = ious.max(dim=0)

            if best_iou >= iou_threshold and not gt_data["matched"][best_gt_idx]:
                tp[idx] = 1
                gt_data["matched"][best_gt_idx] = True
            else:
                fp[idx] = 1

        if len(tp) == 0:
            return 0.0

        cumulative_tp = torch.cumsum(tp, dim=0)
        cumulative_fp = torch.cumsum(fp, dim=0)

        recalls = cumulative_tp / (total_gt + 1e-8)
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8)

        return compute_average_precision(precisions, recalls)

    def _compute_global_precision_recall_iou(
        self,
        iou_threshold: float,
    ) -> tuple[float, float, float]:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious = []

        for pred, target in zip(self.predictions, self.targets):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            target_boxes = target["boxes"]
            target_labels = target["labels"]

            matched_gt = torch.zeros(len(target_boxes), dtype=torch.bool)

            order = pred_scores.argsort(descending=True)

            for pred_idx in order:
                box = pred_boxes[pred_idx].unsqueeze(0)
                label = pred_labels[pred_idx]

                valid_gt = torch.where(
                    (target_labels == label) & (~matched_gt)
                )[0]

                if len(valid_gt) == 0:
                    total_fp += 1
                    continue

                ious = box_iou(box, target_boxes[valid_gt]).squeeze(0)
                best_iou, best_local_idx = ious.max(dim=0)
                best_gt_idx = valid_gt[best_local_idx]

                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched_gt[best_gt_idx] = True
                    matched_ious.append(best_iou.item())
                else:
                    total_fp += 1

            total_fn += (~matched_gt).sum().item()

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

        return precision, recall, mean_iou
