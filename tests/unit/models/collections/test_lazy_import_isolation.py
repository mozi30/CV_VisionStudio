from __future__ import annotations

import sys


def test_importing_models_does_not_import_optional_collection_backends():
    sys.modules.pop("timm", None)
    sys.modules.pop("mmpretrain", None)

    import vision_studio.models  # noqa: F401

    assert "timm" not in sys.modules
    assert "mmpretrain" not in sys.modules
