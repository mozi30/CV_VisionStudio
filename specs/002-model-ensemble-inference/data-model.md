# Data Model: Unified Inference and Ensemble Evaluation

## InferenceService

**Purpose**: Single concrete inference orchestrator that executes both single-model and ensemble inference/evaluation flows.

**Key fields / attributes**:

- `reporter`: Selected reporter strategy used for output publication.
- `default_aggregation_mode`: Default ensemble aggregation mode (soft voting).
- `default_tie_policy`: Default hard-vote tie policy (index-priority).
- `default_failure_policy`: Default model-failure policy (fail-fast).
- `metric_mode_default`: Inference default behavior (prediction-only, no metric computation).

**Validation rules**:

- Reject empty model list.
- Validate supported aggregation mode.
- Validate supported tie policy and failure policy.

## EnsembleRunConfig

**Purpose**: User-provided settings controlling an ensemble run.

**Key fields / attributes**:

- `models`: Ordered list of participating models.
- `aggregation_mode`: One of soft voting, hard voting, weighted averaging.
- `weights`: Optional weights list used only for weighted averaging.
- `tie_policy`: Tie handling policy for hard voting.
- `failure_policy`: Per-model failure handling mode.

**Validation rules**:

- `models` must contain at least one model.
- All models must expose the same output-class count.
- `weights` length must match model count when weighted averaging is selected.
- `weights` must satisfy configured numeric constraints.

## ModelExecutionResult

**Purpose**: Per-model execution output for a run.

**Key fields / attributes**:

- `model_id`: Stable identifier for the model.
- `status`: success or failed.
- `predictions`: Model output predictions (on success).
- `error_summary`: Safe diagnostic summary (on failure).

## AggregatedPrediction

**Purpose**: Final per-sample ensemble prediction.

**Key fields / attributes**:

- `sample_id`: Input sample reference.
- `final_class`: Final predicted class for the sample.
- `final_probabilities`: Final aggregated class probabilities when available.
- `aggregation_metadata`: Applied mode, tie policy, failure policy, failed model list, total model count, and successful model count.

## RunOutcome

**Purpose**: High-level run result for inference/evaluation.

**Key fields / attributes**:

- `status`: completed, completed_with_warnings, or failed.
- `failed_models`: List of model identifiers that failed.
- `predictions`: Collection of final predictions.
- `metrics`: Evaluation metrics when run type is evaluation.

**State transitions**:

- `initialized` → `validating` → `executing` → `aggregating` → (`completed` | `completed_with_warnings` | `failed`)

## RealtimeInferenceConfig

**Purpose**: Controls behavior for realtime webcam inference runs.

**Key fields / attributes**:

- `source`: Webcam source selector.
- `frame_skip`: Optional frame-skipping rule.
- `record_output`: Toggle for annotated output-video recording.
- `output_path`: Target location for annotated output artifact when recording is enabled.

**Validation rules**:

- `source` must be openable before run start.
- `source` must not be a negative index or empty string.
- `frame_skip` must be non-negative and interpretable by run loop.
- `output_path` must be writable when recording is enabled.
