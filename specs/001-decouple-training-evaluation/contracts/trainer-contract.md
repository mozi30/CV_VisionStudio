# Contract: Trainer

## Purpose

The Trainer contract defines externally visible behavior for training workflows after evaluation is decoupled.

## Required behavior

1. Validate configuration before the epoch loop without consuming Dataset batches by default.
2. Use train/fit Evaluator override when provided; otherwise use constructor default Evaluator.
3. Train each epoch using optimizer updates and model loss.
4. If an Evaluator and evaluation Dataset are configured, call the Evaluator after each successfully completed epoch.
5. If no Evaluator is configured, warn, perform no evaluation, log only training loss, and leave evaluation history empty.
6. If checkpoint path is unset, warn and save no checkpoints.
7. If checkpoint path is set but does not exist, fail before training starts.
8. Save latest and best checkpoints only when checkpoint path and required monitor settings allow it.
9. Apply early stopping according to trainer settings.
10. Stop gracefully on interruption.

## API Boundary

- Trainer base exposes train/fit and checkpoint lifecycle behavior.
- Trainer base does not expose public validation or test responsibilities.
- Training-time evaluation is performed only by calling an Evaluator.
- The concrete Trainer has a general name and does not imply WandB-only behavior.
- Trainer-focused contract tests should be planned under grouped folders such as `tests/unit/trainer` and `tests/integration/training`.
