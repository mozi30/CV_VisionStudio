# Contract: Inference Service

## Purpose

Define externally visible behavior for unified inference orchestration.

## Required Behavior

1. Expose one base inference abstraction and one concrete inference implementation.
2. Support single-model and multi-model ensemble runs through the same orchestration flow.
3. Support prediction-only default behavior for inference runs (no metric computation unless evaluation mode is explicitly selected).
3. Validate run configuration before execution (model list, policies, aggregation settings).
4. Delegate reporting through configured reporter strategy without changing inference control flow.
5. Return deterministic results for deterministic policy settings.
6. Support image input, video-path stream input, and realtime webcam input modes.
7. Support optional frame-skipping behavior for continuous webcam inference.
8. Support optional annotated output-video recording for webcam inference.
9. Validate webcam source before realtime execution starts (reject negative index or empty source path).
10. Emit webcam-mode artifact path metadata when recording is enabled.

## Failure Behavior

- Empty model list must fail before execution.
- Invalid policy or aggregation mode must fail with clear diagnostic.
- Fail-fast policy must stop on first model failure.
- Continue-with-warning policy must fail if zero models succeed.
- Webcam source open failure must return clear safe diagnostics.
- Invalid webcam source configuration must fail with explicit configuration error before frame processing.

## Compatibility

- Preserve existing single-model workflow compatibility.
- Do not require task-specific inference class selection by users.
