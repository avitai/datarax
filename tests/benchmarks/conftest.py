"""Shared fixtures for TDD performance target tests (P0-P5)."""

import numpy as np
import pytest


@pytest.fixture
def cv1_large_image_data():
    """10K 224x224x3 uint8 images for CV-1 comparative benchmarks.

    Used by P2 (GPU augmentation vs DALI) and P3 (memory efficiency vs SPDL).
    """
    return {
        "image": np.random.default_rng(42).integers(0, 255, (10_000, 224, 224, 3), dtype=np.uint8)
    }
