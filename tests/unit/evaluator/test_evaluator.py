"""Unit tests for evaluator exceptions and contracts."""

from vision_studio.types import EvaluationError


def test_evaluation_error_is_exception() -> None:
    """EvaluationError should remain an exception subtype."""
    assert issubclass(EvaluationError, Exception)
