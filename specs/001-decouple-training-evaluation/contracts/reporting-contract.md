# Contract: Reporting Modes

## Purpose

Reporting modes define how training and evaluation metrics become visible to users without making WandB mandatory.

## Shared Reporter Interface

- Trainer uses reporter for training metrics.
- Evaluator uses reporter for evaluation metrics.
- EvaluationMetrics do not use reporters directly.
- Reporting-focused contract tests should be planned under grouped folders such as `tests/unit/reporting` and integration coverage under `tests/integration/training`.

## Modes

| Mode | Required Behavior |
|------|-------------------|
| WandB | Initialize remote experiment run, send selected metrics, fail with exception if unavailable |
| Local live plot | Display local live plots when possible, log metrics locally, warn and continue if plotting fails |
| Logging-only | Log metrics locally and never transmit metrics remotely |
