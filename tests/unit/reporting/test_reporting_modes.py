"""Unit tests for reporting implementations."""

from vision_studio.reporting import LoggingReporter


def test_logging_reporter_constructs() -> None:
    """LoggingReporter should support a basic start-log-finish cycle."""
    reporter = LoggingReporter()
    reporter.start()
    reporter.log({"loss": 1.0}, step=1)
    reporter.finish()
