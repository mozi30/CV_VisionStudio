"""Unit tests for checkpoint-related trainer settings."""

from pathlib import Path

from vision_studio.types import TrainerSettings


def test_checkpoint_path_can_be_unset() -> None:
    """Checkpoint path may be left unset for warning-only behavior."""
    settings = TrainerSettings(checkpoint_path=None)
    assert settings.checkpoint_path is None


def test_checkpoint_path_accepts_existing_directory(tmp_path: Path) -> None:
    """An existing directory should be accepted as checkpoint path."""
    settings = TrainerSettings(checkpoint_path=tmp_path)
    assert settings.checkpoint_path == tmp_path


def test_best_checkpoint_count_defaults_to_three() -> None:
    """Best checkpoint retention should default to three."""
    settings = TrainerSettings()
    assert settings.best_checkpoint_count == 3


def test_checkpoint_monitor_can_be_configured() -> None:
    """Checkpoint monitor should be configurable."""
    settings = TrainerSettings(checkpoint_monitor="loss")
    assert settings.checkpoint_monitor == "loss"


def test_checkpoint_mode_defaults_to_min() -> None:
    """Checkpoint mode should default to minimizing the monitor metric."""
    settings = TrainerSettings()
    assert settings.checkpoint_mode == "min"
