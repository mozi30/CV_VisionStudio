# Feature Specification: Ensemble Dataset Evaluation

**Feature Branch**: `[004-ensamble-evaluator]`

**Created**: 2026-06-02

**Status**: Draft

**Input**: User description: "Implement the missing ensamble evaluation function in the Loop Evaluator. The evaluator should have the same properties as the inference implementation"

## Clarifications

### Session 2026-06-02

- Q: How should ensemble evaluation handle loss reporting? → A: Report metrics from aggregated predictions and report loss as the mean of successful per-model losses per batch.
- Q: What shape should ensemble evaluation results use? → A: Return a flat result containing metric keys plus `status`, `aggregation_metadata`, and `failed_models`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Evaluate an Ensemble on a Dataset (Priority: P1)

As a model developer, I want to evaluate multiple compatible models together over an evaluation dataset, so that I receive one consolidated result that reflects the ensemble prediction quality without manually running per-batch aggregation.

**Why this priority**: This is the primary missing capability. Ensemble inference already produces aggregated predictions for a batch, but users need the same ensemble behavior across a full dataset with evaluation metrics.

**Independent Test**: Can be fully tested by providing two compatible classification models, an evaluation dataset, and evaluation metrics, then verifying that the run returns one consolidated evaluation result based on aggregated ensemble predictions.

**Acceptance Scenarios**:

1. **Given** multiple compatible models and an evaluation dataset, **When** the user evaluates the ensemble, **Then** the system processes every dataset batch and returns one consolidated evaluation result.
2. **Given** no aggregation mode is selected, **When** the user evaluates the ensemble, **Then** the system uses soft voting by default and reports that mode in the result metadata.
3. **Given** a valid weighted configuration, **When** the user evaluates the ensemble, **Then** the system applies the configured weights consistently across all evaluated batches.

---

### User Story 2 - Audit Ensemble Evaluation Behavior (Priority: P2)

As a model developer, I want ensemble evaluation results to expose the selected aggregation behavior and model failure status, so that I can audit how the final metrics were produced.

**Why this priority**: Ensemble results are harder to interpret than single-model results unless the run explains how predictions were combined and whether any model failed.

**Independent Test**: Can be fully tested by running ensemble evaluation with each supported aggregation mode and checking that the output includes aggregation mode, tie policy, failure policy, model count, successful model count, failed model indexes, and run status.

**Acceptance Scenarios**:

1. **Given** an ensemble evaluation run, **When** the result is returned, **Then** it includes aggregation metadata sufficient to identify the chosen aggregation behavior.
2. **Given** an allowed continue-with-warning model failure policy and at least one model succeeds, **When** some models fail during evaluation, **Then** the result is marked as completed with warnings and lists the failed models.
3. **Given** an ensemble evaluation run computes standard metrics, **When** the result is returned, **Then** metric values and ensemble audit fields are available in one flat result.

---

### User Story 3 - Preserve Single-Model Evaluation Behavior (Priority: P3)

As an existing user of single-model evaluation, I want current evaluation behavior to remain unchanged, so that adopting ensemble evaluation does not break existing validation workflows.

**Why this priority**: Ensemble support should extend evaluation, not alter established single-model behavior.

**Independent Test**: Can be fully tested by running an existing single-model evaluation scenario and verifying that its output, metric updates, reporting behavior, and model training-mode restoration remain equivalent.

**Acceptance Scenarios**:

1. **Given** an existing single-model evaluation workflow, **When** the workflow is run after this feature is introduced, **Then** it produces the same type of result as before.
2. **Given** a model is in training mode before evaluation, **When** single-model or ensemble evaluation finishes successfully, **Then** the models return to their original training or evaluation state.

### Edge Cases

- Empty model lists are rejected before dataset processing begins.
- Weighted aggregation is rejected when weights are missing, empty, non-matching in count, or unusable for deterministic aggregation.
- Models whose outputs expose incompatible class counts or incompatible prediction shapes are rejected with a clear diagnostic.
- If all models fail under continue-with-warning behavior, evaluation fails rather than returning a misleading successful result.
- Empty datasets return a clear safe result or error according to the existing evaluation behavior, without marking incomplete metrics as successful.
- Hard-vote ties use the configured tie policy and report that policy in the result metadata.
- Model failures do not hide the original failure category from diagnostics while still returning safe user-facing messages.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow users to evaluate a list of models over an evaluation dataset and receive one consolidated evaluation result.
- **FR-002**: System MUST execute each ensemble model independently on the same evaluation samples before aggregating predictions for each sample.
- **FR-003**: System MUST support the same aggregation modes available to ensemble inference: soft voting, hard voting, and weighted averaging.
- **FR-004**: System MUST use soft voting as the default aggregation mode when no explicit ensemble evaluation mode is provided.
- **FR-005**: System MUST compute evaluation metrics from the aggregated ensemble predictions and the dataset targets, not from separate per-model metrics.
- **FR-006**: System MUST include aggregation mode, tie policy, failure policy, model count, successful model count, failed models, and run status in ensemble evaluation outputs.
- **FR-007**: System MUST validate that all successful model outputs in an ensemble expose compatible output class counts before aggregation.
- **FR-008**: System MUST support deterministic weighted averaging when valid per-model weights are provided.
- **FR-009**: System MUST support the same model failure policies as ensemble inference, including fail-fast and continue-with-warning.
- **FR-010**: System MUST require at least one successful model output for every evaluated batch before producing aggregated predictions.
- **FR-011**: System MUST preserve existing single-model evaluation behavior and output expectations.
- **FR-012**: System MUST restore each evaluated model to its pre-evaluation training or evaluation state after the run completes or fails safely.
- **FR-013**: System MUST report evaluation progress and final evaluation metrics through the configured reporting destination consistently with existing evaluation runs.
- **FR-014**: System MUST report ensemble loss as the mean of successful per-model losses for each batch while computing non-loss metrics from aggregated ensemble predictions.
- **FR-015**: System MUST return ensemble evaluation results as a flat result containing metric keys plus `status`, `aggregation_metadata`, and `failed_models`.

### Error Handling Requirements *(mandatory)*

- **ERR-001**: If an ensemble has no models, evaluation MUST fail before consuming the dataset with a clear message that at least one model is required.
- **ERR-002**: If aggregation configuration is unsupported or incomplete, evaluation MUST fail with a clear message identifying the invalid setting.
- **ERR-003**: If model outputs are incompatible for aggregation, evaluation MUST fail with a clear message that identifies shape or class-count incompatibility.
- **ERR-004**: If fail-fast behavior is selected and any model fails on a batch, evaluation MUST stop and surface the failure safely.
- **ERR-005**: If continue-with-warning behavior is selected and no model succeeds for a batch, evaluation MUST fail rather than returning partial metrics as successful.
- **ERR-006**: If metric computation cannot consume the aggregated prediction format, evaluation MUST fail with a clear diagnostic that identifies metric incompatibility.

### AI and Release Requirements *(mandatory if AI assists or feature is releasable)*

- **AI-001**: AI-assisted specification, planning, and implementation artifacts MUST be human-reviewed for correctness, metric validity, and consistency with existing ensemble inference behavior before release.
- **REL-001**: Release readiness MUST include unit tests for supported aggregation modes, invalid configurations, incompatible model outputs, model failure policies, metadata output, and unchanged single-model evaluation behavior.
- **REL-002**: Release readiness MUST include an integration test demonstrating ensemble evaluation over a dataset with consolidated metrics.
- **REL-003**: Release readiness MUST include documentation or example updates showing how to run ensemble evaluation and interpret aggregation metadata.

### Non-Functional Requirements *(mandatory)*

- **NFR-001**: Ensemble evaluation results MUST be reproducible for the same dataset, model list, aggregation configuration, and deterministic tie policy.
- **NFR-002**: Ensemble evaluation MUST not add manual post-processing steps for users beyond selecting models, dataset, metrics, and aggregation configuration.
- **NFR-003**: Ensemble evaluation MUST keep metric ownership clear by using the existing evaluation metric concepts for final result computation.
- **NFR-004**: Ensemble evaluation SHOULD complete within the expected cost of evaluating each participating model once per dataset sample plus aggregation overhead.

### Key Entities *(include if feature involves data)*

- **Ensemble Evaluation Configuration**: User-selected aggregation mode, tie policy, failure policy, and optional per-model weights for a dataset evaluation run.
- **Model Set**: Ordered list of models participating in the ensemble evaluation, including their success or failure status during the run.
- **Aggregated Batch Prediction**: Final per-sample prediction output for one dataset batch after applying the selected aggregation behavior.
- **Ensemble Evaluation Result**: Consolidated metrics and metadata returned after evaluating the ensemble across the dataset.
- **Ensemble Audit Fields**: Flat result fields that describe run status, aggregation metadata, and failed model indexes alongside metric values.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can evaluate an ensemble of at least two compatible models over a dataset and receive one consolidated metric result without manual per-batch aggregation in 100% of supported scenarios.
- **SC-002**: 100% of ensemble evaluation results include aggregation metadata identifying mode, policies, model counts, failed models, and final status.
- **SC-003**: 100% of invalid ensemble configurations covered by requirements fail before reporting a successful evaluation result.
- **SC-004**: Existing single-model evaluation tests continue to pass without requiring user workflow changes.
- **SC-005**: For deterministic aggregation configurations, repeated ensemble evaluation runs over the same inputs produce identical predictions and metrics.
- **SC-006**: Ensemble evaluation loss matches the mean successful per-model loss aggregation rule in 100% of tested supported runs.
- **SC-007**: 100% of supported ensemble evaluation results expose metric values and ensemble audit fields in one flat result.

## Supplemental Governance Checklist *(mandatory)*

| Gate | Applicability | Planned Evidence |
|------|---------------|------------------|
| Security reviewed | No - feature does not add new trust boundaries or external data exposure beyond existing evaluation inputs | Confirm no new credential, network, or artifact-loading behavior is introduced |
| Dependencies justified | No - feature should use existing evaluation and inference capabilities | Dependency review confirms no new runtime dependency is required |
| Errors specified and tested | Yes - invalid ensemble and metric states are central to the feature | Unit and integration tests for configuration, model failures, compatibility, and metric incompatibility |
| AI output reviewed | Yes - AI is assisting specification creation | Human review of spec, plan, implementation, and tests before acceptance |
| Release gates passed | Yes - feature is releasable behavior | Pytest coverage, documentation/example update, and owner acceptance |

## Assumptions

- Ensemble evaluation initially targets classification-style outputs, matching the current ensemble inference scope.
- Existing evaluation metric objects can consume aggregated prediction outputs after appropriate normalization.
- Dataset batches follow the existing evaluation dataset shape and target conventions.
- The ordered model list is meaningful for weighted aggregation, failure reporting, and index-priority hard-vote ties.
- Random tie behavior, when selected, is allowed to be non-deterministic unless the caller controls randomness through existing project conventions.
