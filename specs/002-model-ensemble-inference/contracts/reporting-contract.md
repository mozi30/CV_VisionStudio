# Contract: Reporting Strategy for Inference and Evaluation

## Purpose

Define reporter-pattern behavior shared by unified inference and evaluation flows.

## Shared Reporter Interface Behavior

1. Reporter selection is configuration-driven.
2. Reporter mode changes only output destination/format, not inference/evaluation control flow.
3. Reporter failures follow configured failure policy and must not silently mask run status.
4. In `completed_with_warnings` runs, failed-model metadata must be reportable.
5. Reporter selection must not alter inference orchestration decisions, aggregation outcomes, or failure-policy semantics.
6. In webcam recording runs, reporter output may include artifact references but must not mutate artifact generation behavior.

## Supported Modes

- Logging/local output mode
- Optional remote experiment-tracking mode

## Failure Behavior

- Reporter configuration errors fail before run start when detectable.
- Safe diagnostics must avoid leaking secrets or sensitive payloads.
