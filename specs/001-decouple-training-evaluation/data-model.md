# Data Model: Decouple Training Evaluation

## Trainer

**Purpose**: Owns model optimization, epoch lifecycle, checkpointing, early stopping, interruption handling, and calls to an Evaluator when evaluation is configured.

**Key fields / attributes**:

- `optimizer`: Optimization state manager used during training.
- `device`: Execution target for model inputs and optimizer state.
- `current_epoch`: Latest active or completed epoch index.
- `global_step`: Total completed optimizer steps.
- `settings`: Trainer settings controlling epochs, checkpoint policy, early stopping, reporting, and optional dry-run validation.
- `default_evaluator`: Optional Evaluator configured at construction.
- `reporter`: Shared reporter implementation used for training metrics.

**Relationships**: Uses Model and Dataset objects, optionally calls Evaluator after completed epochs, emits through Reporter, and produces Training Result plus optional Checkpoint artifacts.

**Validation rules**: No public validation/test methods; fit/train evaluator override takes precedence over constructor default; missing Evaluator warns with no evaluation history; unset checkpoint path warns and saves no checkpoints; configured checkpoint path must exist.

## Trainer Settings

**Key fields / attributes**:

- `epochs`
- `checkpoint_path`
- `save_latest_checkpoint`
- `best_checkpoint_count`
- `checkpoint_monitor`
- `checkpoint_mode`
- `early_stopping_enabled`
- `early_stopping_monitor`
- `early_stopping_patience`
- `dry_run_batch_validation`
- `reporting_mode`
- `log_every_n_steps`

## Evaluator

**Purpose**: Owns the complete evaluation loop over model and evaluation Dataset for both training-time after-epoch evaluation and standalone evaluation.

**Key fields / attributes**:

- `device`: Execution target for evaluation batches.
- `metrics`: EvaluationMetrics instance or collection used during the evaluation loop.
- `reporter`: Optional shared reporter used for evaluation metrics.
- `context`: Optional epoch, model, and Dataset metadata for result traceability.

**Validation rules**: Reject broken models, unsupported tasks, wrong metrics, and incompatible Dataset data with exceptions; do not modify optimizer state; reset EvaluationMetrics for every run.

## EvaluationMetrics

**Purpose**: Aggregates and computes evaluation metrics from predictions, targets, and loss supplied by the Evaluator.

**Types**:

- `ClassificationEvaluationMetrics`
- `DetectionEvaluationMetrics`
- `LossEvaluationMetrics`

**Validation rules**: Unsupported task, wrong metric, or mismatched prediction/target inputs raise exceptions; EvaluationMetrics do not own model inference or Dataset iteration.

## Dataset

**Purpose**: Project data abstraction used as training and evaluation input.

**Validation rules**: Training and evaluation use Dataset objects; dry-run checks may consume Dataset batches only when explicitly enabled.

## Reporter

**Purpose**: Shared interface used by Trainer and Evaluator to publish metrics.

**Validation rules**: WandB mode fails if unavailable; local live plot failure warns and continues with logging; logging-only never transmits remotely; credentials are never logged.

## Training Result

**Key fields / attributes**:

- `history`
- `evaluation_history`
- `current_epoch`
- `global_step`
- `checkpoints`
- `warnings`
- `stopped_early`
- `interrupted`

## Checkpoint

**Purpose**: Saved model and optimizer state retained when checkpoint path is configured and exists.

**Validation rules**: No checkpoints are saved when checkpoint path is unset; latest checkpoint represents latest completed epoch; best checkpoints require explicit monitor metric.

## Test Double

**Purpose**: Mock or lightweight double created from base implementations for pytest tests.

**Validation rules**: Unit tests may use doubles for Trainer, Evaluator, EvaluationMetrics, Dataset, Model, and Reporter boundaries in grouped folders such as `tests/unit/evaluator` and `tests/unit/trainer`; integration tests should use realistic combinations under grouped folders such as `tests/integration/training`.

## State Transitions

```text
Trainer configured
  -> pre-run checks without Dataset batch consumption by default
  -> optional dry-run batch validation if enabled
  -> checkpoint path check: unset warns and disables checkpointing, set but missing fails
  -> epoch training started
  -> epoch completed
  -> Evaluator called if configured OR evaluation skipped with warning and empty history
  -> checkpoints saved only if checkpoint path is configured and exists
  -> next epoch OR stopped early OR interrupted OR completed

Evaluator configured
  -> pre-run evaluation checks passed
  -> EvaluationMetrics state reset
  -> Dataset batches processed
  -> EvaluationMetrics result computed
  -> metrics reported OR exception raised
```
