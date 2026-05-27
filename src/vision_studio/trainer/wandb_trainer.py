"""Backward-compatible Trainer alias for legacy WandB-oriented imports."""

from __future__ import annotations

from .trainer import VisionTrainer


class WandbTrainer(VisionTrainer):
    """Backward-compatible alias for the generalized Trainer implementation."""
