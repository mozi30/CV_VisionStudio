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

## Usage examples

### Train with an Evaluator

Configure a `Trainer` with an optional default `Evaluator`, a reporter, and training/evaluation datasets. A completed run should return epoch-level training history and evaluator-produced evaluation history.

### Override the Evaluator for one run

Construct a `Trainer` once, then provide a different `Evaluator` to `fit()` for a single experiment. The per-run override should take precedence over the constructor default.

### Standalone evaluation

Load a trained model, create an `Evaluator` with `EvaluationMetrics`, and run evaluation directly on an evaluation `Dataset` without invoking training.

### Reporting modes

- Use `LoggingReporter` for local metric logging only.
- Use `LivePlotReporter` when local visual feedback is desired and a warning-only fallback is acceptable.
- Use `WandbReporter` when remote experiment tracking is desired and startup/logging failures should be fatal.

## Migration notes

- Replace legacy public `trainer.validate()` or `trainer.test()` usage with an `Evaluator` instance.
- Replace legacy metric-only evaluator usage with `EvaluationMetrics` naming.
- Keep `WandbTrainer` imports only as compatibility aliases; prefer the general `VisionTrainer` implementation going forward.

## Dependency review evidence

- `pytest` is required for the grouped unit and integration test workflow and is documented in [`pyproject.toml`](../../pyproject.toml).
- Reporting dependencies remain optional by mode: `wandb` is required only for remote tracking, while local logging and live-plot paths do not require remote credentials.
- Local/logging-only reporting paths do not transmit metrics remotely.

## Security and release evidence

- User-facing reporting errors should avoid leaking credentials or tokens.
- Local and logging-only reporters should not require WandB credentials.
- Release validation should include pytest, Ruff, and Black checks using the project environment.
