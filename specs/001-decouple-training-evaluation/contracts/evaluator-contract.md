# Contract: Evaluator

## Purpose

The Evaluator owns the evaluation loop for after-epoch and standalone evaluation.

## Required behavior

1. Validate model, Dataset, EvaluationMetrics, and reporter configuration before full evaluation whenever possible.
2. Reset EvaluationMetrics state at the start of each run.
3. Run model inference without optimizer updates.
4. Compute or extract loss needed by EvaluationMetrics.
5. Convert model outputs into EvaluationMetrics-required prediction and target inputs.
6. Update EvaluationMetrics for each evaluation batch.
7. Compute and return evaluation result.
8. Report metrics through the shared reporter when configured.

## Boundaries

- Runs after each completed epoch when called by Trainer.
- Runs standalone without starting training.
- Does not silently fall back from WandB mode to local mode.
- Evaluator-focused contract tests should be planned under grouped folders such as `tests/unit/evaluator`.
