# Quickstart: Decouple Training Evaluation

## Goal

Validate that Trainer, Evaluator, EvaluationMetrics, Dataset, and Reporter responsibilities are separated while preserving training-time evaluation, standalone evaluation, reporting options, optional checkpointing, early stopping, and required failure behavior.

## Prerequisites

- Python environment with project dependencies installed.
- Pytest available for test execution.
- A small model that can compute loss.
- Small training and evaluation Dataset objects.
- Classification, detection, and loss-only EvaluationMetrics fixtures.
- Optional WandB credentials only when WandB mode is intentionally tested.

## Test layout

- Unit tests live under grouped folders such as `tests/unit/evaluator`, `tests/unit/trainer`, and `tests/unit/reporting`.
- Integration tests live under grouped folders such as `tests/integration/training`.
- Mocks and lightweight doubles should be created from available base implementations where practical.

## Suggested verification commands

```bash
python -m pytest tests/unit/evaluator tests/unit/trainer tests/unit/reporting tests/integration/training
python -m ruff check src tests
python -m black --check src tests
```
