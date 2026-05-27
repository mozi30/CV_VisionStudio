# Contract: EvaluationMetrics

## Purpose

EvaluationMetrics aggregate and compute metrics from predictions, targets, and loss supplied by the Evaluator.

## Scope Boundary

- EvaluationMetrics do not iterate Dataset objects.
- EvaluationMetrics do not run model inference.
- EvaluationMetrics do not own reporting.
- EvaluationMetrics-focused contract tests should be planned under grouped folders such as `tests/unit/evaluator`.

## Supported First-Implementation Metrics

| EvaluationMetrics | Required Inputs | Example Metrics |
|-------------------|-----------------|-----------------|
| ClassificationEvaluationMetrics | Class logits/probabilities or labels, class-label targets, optional loss | loss, accuracy, precision, recall, f1, top-k accuracy |
| DetectionEvaluationMetrics | Predicted boxes/scores/labels, ground-truth boxes/labels, optional loss | loss, AP, AP50, AP75, AR variants |
| LossEvaluationMetrics | Explicit loss and inferable batch size | loss |
