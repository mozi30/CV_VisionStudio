# Research: Ensemble Dataset Evaluation

## Decision: Add a dedicated ensemble evaluation entry point

Add an explicit `LoopEvaluator` ensemble evaluation method rather than changing the existing single-model `evaluate()` behavior.

**Rationale**: Existing training code and tests call `evaluate(model, dataset)` for one model. A dedicated entry point keeps current behavior stable while making ensemble usage obvious in notebooks and examples.

**Alternatives considered**:

- Overload `evaluate()` to accept either one model or a list of models: rejected because it would make call-site behavior less explicit and increase compatibility risk.
- Build ensemble evaluation only as notebook helper code: rejected because the feature needs reusable library behavior and tests.

## Decision: Align aggregation properties with ensemble inference

Use the same aggregation modes and policies as the current inference implementation: soft voting by default, hard voting, weighted averaging, tie policy, failure policy, and model-count metadata.

**Rationale**: The feature request says evaluator behavior should have the same properties as inference. Keeping one semantic contract reduces user surprise and test duplication.

**Alternatives considered**:

- Define evaluator-specific aggregation modes: rejected because it would split behavior between inference and evaluation.
- Support only soft voting for v1: rejected because the existing inference implementation already exposes hard and weighted modes and the spec requires parity.

## Decision: Compute metrics from aggregated predictions and loss from mean successful per-model losses

Non-loss metrics are computed from aggregated ensemble predictions and dataset targets. Loss is reported as the mean of successful per-model losses per batch.

**Rationale**: Metrics should describe the ensemble's final predictions, not separate model predictions. Loss has no single ensemble model owner, so mean successful per-model loss is a clear, testable, practical default aligned with the clarification.

**Alternatives considered**:

- Omit loss for ensemble evaluation: rejected because existing evaluator outputs expect loss and notebooks commonly inspect it.
- Compute loss from aggregated logits/probabilities: rejected because not all aggregation modes produce logits/probabilities suitable for a model loss function.

## Decision: Return a flat result with audit fields

Return metric keys alongside `status`, `aggregation_metadata`, and `failed_models` in one flat result.

**Rationale**: Existing evaluator consumers receive flat metric dictionaries. Adding audit fields at the same level keeps notebook usage simple while exposing ensemble behavior.

**Alternatives considered**:

- Nest metrics under a `metrics` field: rejected because it diverges from current evaluator output shape.
- Return metadata only through a helper: rejected because it makes ensemble audit behavior too easy to miss.

## Decision: Restore model training states after success or failure

Capture each model's original training state before evaluation and restore it after the run, including error paths.

**Rationale**: Single-model evaluation already preserves training mode. Ensembles must preserve that property for all participating models.

**Alternatives considered**:

- Restore only after successful completion: rejected because failures during evaluation should not leave models in unexpected modes.
- Leave models in eval mode: rejected because it would regress existing evaluator expectations.

## Decision: Keep first scope to classification-style outputs

Limit this increment to classification-style tensor outputs compatible with current ensemble inference and classification metrics.

**Rationale**: Current ensemble inference aggregates tensor class outputs. Detection ensemble evaluation needs separate box/score matching semantics and should be designed independently.

**Alternatives considered**:

- Include detection ensemble evaluation now: rejected because aggregation semantics differ materially and would broaden scope.
- Make the method task-agnostic without tests: rejected because it would create an unclear contract.
