# Research: Unified Inference and Ensemble Evaluation

## Decision: Keep one inference base and one concrete inference implementation

**Rationale**: The feature explicitly requires alignment with trainer/evaluator architecture and rejects fragmented inference implementations by task or backend.

**Alternatives considered**:

- Separate inference classes for classification and detection: rejected due to duplication and drift risk.
- Keep legacy multiple concrete implementations: rejected because it conflicts with unified orchestration requirement.

## Decision: Implement ensemble aggregation as selectable strategy in one workflow

**Rationale**: Soft voting, hard voting, and weighted averaging are behavior variants of one inference/evaluation flow and should not require separate pipeline implementations.

**Alternatives considered**:

- Dedicated pipeline per aggregation mode: rejected because it duplicates validation and reporting behavior.
- External post-processing only: rejected because feature requires native ensemble output and metadata.

## Decision: Strict class-count compatibility across models

**Rationale**: Class mapping identity cannot be guaranteed, but equal output-class count is a minimum correctness guard and must fail early on mismatch.

**Alternatives considered**:

- Auto-skip incompatible models: rejected because it can hide data/label incompatibility.
- Attempt auto-remapping by heuristics: rejected because mapping is undefined and unsafe.

## Decision: Default tie policy is index-priority; random is optional

**Rationale**: Index-priority is deterministic and reproducible by default; random policy remains available for explicit experimentation.

**Alternatives considered**:

- Random default: rejected due to reduced reproducibility.
- Error on ties by default: rejected because it interrupts common voting behavior unnecessarily.

## Decision: Default failure policy is fail-fast; continuation is explicit

**Rationale**: Fail-fast best protects result integrity. Continuation mode is permitted for resilience workflows but must mark status as `completed_with_warnings` and include failed model metadata.

**Alternatives considered**:

- Continue-by-default: rejected because silent quality degradation risk is higher.
- Always fail-fast with no continuation option: rejected because user requested continuation support.

## Decision: Shared reporter strategy controls output channel only

**Rationale**: Reporting mode should not alter inference/evaluation orchestration, only where and how results are emitted.

**Alternatives considered**:

- Reporter-specific inference classes: rejected because it reintroduces duplicated control flow.
- Hard-code one reporter backend: rejected because local and optional remote modes are both required.

## Decision: Inference defaults to prediction-only outputs

**Rationale**: Inference is intended for real-data behavior inspection and should not require evaluation-metric computation unless explicitly running evaluation.

**Alternatives considered**:

- Always compute metrics during inference: rejected because metrics need labeled ground truth and add unnecessary overhead.
- Separate prediction-only class: rejected because unified inference architecture is required.

## Decision: Realtime webcam inference supports optional frame skipping and annotated recording

**Rationale**: Live testing on real inputs requires continuous stream handling, adjustable processing rate, and optional saved visual evidence of predictions.

**Alternatives considered**:

- Webcam without frame skipping: rejected because it limits practical performance control.
- Webcam without recording output: rejected because users need reproducible artifacts from live runs.

## Dependency and Release-Gate Evidence Notes

### Dependency Evidence

- No new mandatory runtime dependencies are introduced for T011–T012 scope.
- Existing stack remains sufficient: PyTorch, torchvision, numpy, pytest, with optional wandb/matplotlib reporting integrations already in project context.
- Validation for this scope relies on existing test and lint/format tooling; no additional package adoption is required.

### Release-Gate Evidence

- Required quality gates for this feature stream are explicitly tracked in [`tasks.md`](tasks.md):
  - [`T053`](tasks.md): run pytest suite and record results in [`quickstart.md`](quickstart.md)
  - [`T054`](tasks.md): run lint/format checks and record results in [`quickstart.md`](quickstart.md)
  - [`T055`](tasks.md): record release notes, rollback guidance, and owner acceptance evidence in [`quickstart.md`](quickstart.md)
- Security and error-handling gate expectations are documented in [`plan.md`](plan.md) Constitution checks and reinforced by trust-boundary/logging assumptions in [`quickstart.md`](quickstart.md).
- AI-generated artifacts require human review before acceptance per constitution alignment; evidence is captured through completed tasks/checklist records.
