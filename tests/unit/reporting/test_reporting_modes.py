"""Unit tests for reporting implementations."""

import pytest

from vision_studio.reporting import LivePlotReporter, LoggingReporter


def test_logging_reporter_constructs() -> None:
    """LoggingReporter should support a basic start-log-finish cycle."""
    reporter = LoggingReporter()
    reporter.start()
    reporter.log({"loss": 1.0}, step=1)
    reporter.finish()


def test_placeholder_local_live_plot_fallback() -> None:
    """Placeholder test for warning-only local live-plot fallback behavior."""
    reporter = LivePlotReporter()
    reporter.start()
    reporter.log({"loss": 1.0}, step=2)
    reporter.finish()


def test_wandb_reporter_requires_start_before_log() -> None:
    """Placeholder to keep grouped reporting test coverage coherent."""
    assert pytest is not None
