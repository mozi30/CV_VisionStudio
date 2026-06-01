# Contract: Ensemble Aggregation

## Purpose

Define required behavior for model-output aggregation in ensemble runs.

## Supported Aggregation Modes

1. Soft voting (probability averaging)
2. Hard voting (majority class voting)
3. Weighted averaging (weighted probability averaging)

## Required Behavior

1. Use soft voting as default when mode is omitted.
2. Require equal output-class count across all participating models.
3. In hard-vote ties, apply configured tie policy:
   - index-priority (default)
   - random
4. In weighted averaging, validate weight count and numeric validity before aggregation.
5. Include aggregation mode and applied policies in output metadata.
6. Include failure-policy and model execution summary metadata (`model_count`, `successful_model_count`, `failed_models`).
7. Reject unsupported aggregation modes and unsupported tie policies with explicit configuration errors.

## Failure Behavior

- Class-count mismatch must fail before aggregation.
- Invalid weights or policy values must fail with clear diagnostics.
- Weighted mode without weights, or with mismatched weight length, must fail before aggregation.
