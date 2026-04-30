"""Shared fixtures for TDD performance target tests (P0-P5)."""

import io
from pathlib import Path

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


def pytest_configure(config: pytest.Config) -> None:
    """Materialise a small synthetic ImageNet64-shaped ArrayRecord dataset.

    Runs before collection so the class-level ``skipif`` in
    ``test_imagenet_scaling.py`` sees the populated directory. Real
    ImageNet is too large to ship; this generates ~45MB of random
    JPEG-encoded 64x64 RGB tiles so the throughput test exercises the
    full ArrayRecord I/O code path. The directory is gitignored so the
    fixture re-creates it on fresh checkouts (CI included).
    """
    del config  # unused; pytest hook signature requires it
    out_dir = Path("tests/data/imagenet64_arrayrecord")
    if out_dir.exists() and any(out_dir.glob("*.array_record")):
        return

    try:
        from array_record.python.array_record_module import ArrayRecordWriter
    except ImportError:
        # array_record is an optional dependency; let the test's skipif
        # handle it via the existing DATASET_DIR check.
        return

    try:
        from PIL import Image
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    num_shards = 4
    records_per_shard = 4_000
    img_size = 64

    for shard_idx in range(num_shards):
        shard_path = out_dir / f"shard-{shard_idx:05d}.array_record"
        writer = ArrayRecordWriter(str(shard_path), "group_size:1")
        try:
            for _ in range(records_per_shard):
                arr = rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                buf = io.BytesIO()
                Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=70)
                writer.write(buf.getvalue())
        finally:
            writer.close()
