# Data Model: Ensemble Dataset Evaluation

## EnsembleEvaluationConfig

Represents user-selected behavior for an ensemble evaluation run.

**Fields**:

- `mode`: Aggregation mode. Valid values are `soft`, `hard`, and `weighted`; default is `soft`.
- `tie_policy`: Hard-vote tie behavior. Valid values match ensemble inference; default is `index-priority`.
- `failure_policy`: Per-model failure behavior. Valid values are `fail-fast` and `continue-with-warning`; default is `fail-fast`.
- `weights`: Optional ordered weights for weighted aggregation. Required when mode is `weighted`.

**Validation Rules**:

- `mode` must be supported.
- `tie_policy` must be supported.
- `failure_policy` must be supported.
- Weighted mode requires non-empty weights whose length matches the number of models.
- Weights must be usable for deterministic aggregation.

## EnsembleModelSet

Represents the ordered collection of models evaluated together.

**Fields**:

- `models`: Ordered list of participating models.
- `model_count`: Total number of models provided.
- `successful_model_count`: Number of models that produced usable outputs for the current run or batch.
- `failed_models`: Ordered list of model indexes that failed.

**Relationships**:

- Owns the model ordering used for weighted aggregation, failed-model reporting, and index-priority ties.
- Produces per-model predictions and per-model losses for each dataset batch.

**Validation Rules**:

- Must contain at least one model.
- Successful model outputs must expose compatible prediction shapes and class counts.
- At least one model must succeed for every evaluated batch.

## AggregatedBatchPrediction

Represents the ensemble output for one dataset batch after aggregation.

**Fields**:

- `predictions`: Final prediction tensor used for metric updates.
- `aggregated`: Aggregated logits, probabilities, or labels according to the selected mode.
- `targets`: Dataset targets for the batch after normal evaluation target handling.
- `loss`: Mean successful per-model loss for the batch.

**Validation Rules**:

- Predictions must be consumable by the configured evaluation metrics.
- The loss value must be based only on successful model outputs.
- Hard-voting aggregation may produce labels; metric update behavior must receive a supported representation.

## EnsembleEvaluationResult

Represents the final flat result returned to the user.

**Fields**:

- Metric keys produced by the configured evaluation metrics, including `loss`.
- `status`: `completed` or `completed_with_warnings`.
- `aggregation_metadata`: Audit metadata for aggregation mode, policies, model counts, and failed models.
- `failed_models`: Flat list of failed model indexes.

**Validation Rules**:

- Successful runs must include metric keys and audit fields.
- Runs with failed models under continue-with-warning must use `completed_with_warnings`.
- Runs with no successful model outputs must fail instead of returning this result.

## State Transitions

1. Models begin in caller-controlled training or evaluation state.
2. Ensemble evaluation captures original states and sets models to evaluation mode.
3. Each dataset batch produces per-model outputs, failures, aggregated predictions, and mean successful loss.
4. Metrics are updated from aggregated predictions, targets, and mean successful loss.
5. Final result is computed and audit fields are attached.
6. All models are restored to their original states after success or failure.
