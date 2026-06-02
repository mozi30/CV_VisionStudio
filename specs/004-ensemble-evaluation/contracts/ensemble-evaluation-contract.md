# Contract: Ensemble Dataset Evaluation

## Public Entry Point

`LoopEvaluator` exposes an ensemble evaluation entry point for dataset-level evaluation of multiple models.

## Inputs

- `models`: Ordered list of models.
- `dataset`: Iterable evaluation dataset yielding batches in the same shape accepted by single-model evaluation.
- `config`: Optional ensemble evaluation configuration. If omitted, defaults match ensemble inference defaults.
- `metrics`: The evaluator's configured metrics object.
- `reporter`: The evaluator's configured reporting destination.

## Behavior

1. Validate the model list is non-empty.
2. Validate aggregation mode, tie policy, failure policy, and weights.
3. Capture each model's initial training/evaluation state.
4. Set all models to evaluation mode for the run.
5. For each dataset batch:
   - Move inputs and tensor targets to the evaluator device.
   - Run each model independently on the same inputs.
   - Compute each successful model's loss against the same moved targets.
   - Postprocess each successful model's outputs into prediction data.
   - Apply the selected ensemble aggregation behavior.
   - Compute the batch loss as the mean successful per-model loss.
   - Update metrics with aggregated predictions, moved targets, and mean successful loss.
6. Restore every model to its original state after success or failure.
7. Return a flat result containing metric keys plus `status`, `aggregation_metadata`, and `failed_models`.

## Aggregation Modes

- `soft`: Average compatible prediction tensors and choose final labels from the averaged result.
- `weighted`: Apply ordered per-model weights to compatible prediction tensors and choose final labels from the weighted result.
- `hard`: Convert each model output to class votes and choose labels by majority vote with the configured tie policy.

## Failure Policies

- `fail-fast`: Stop evaluation when any model fails on a batch and surface the failure.
- `continue-with-warning`: Continue when one or more models fail, as long as at least one model succeeds for every batch. Return `completed_with_warnings` and list failed model indexes.

## Output Shape

The returned result is flat:

```text
{
  "loss": <metric value>,
  "<other metric>": <metric value>,
  "status": "completed" | "completed_with_warnings",
  "aggregation_metadata": {
    "mode": "soft" | "hard" | "weighted",
    "tie_policy": "index-priority" | "random",
    "failure_policy": "fail-fast" | "continue-with-warning",
    "failed_models": [<model indexes>],
    "model_count": <total models>,
    "successful_model_count": <successful models>
  },
  "failed_models": [<model indexes>]
}
```

## Error Contract

- Empty model list fails before dataset consumption.
- Unsupported aggregation mode, tie policy, failure policy, or invalid weights fail before dataset consumption.
- Incompatible successful model outputs fail before metric update.
- All models failing for a batch fails the run.
- Metric incompatibility fails the run with a clear diagnostic.
- Model states are restored even when evaluation fails.

## Compatibility Requirements

- Existing `LoopEvaluator.evaluate(model, dataset)` behavior remains unchanged.
- Default ensemble configuration uses soft voting.
- The contract targets classification-style tensor outputs for this feature.
