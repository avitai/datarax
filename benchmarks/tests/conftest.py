"""Shared fixtures for benchmark tests.

Provides fake in-memory dataset loaders that replace the real-data
provider's download seams, so tests exercising real-data code paths run
fast, offline, and deterministically.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pytest

from benchmarks.fixtures import real_data


FAKE_CIFAR_COUNT = 100
"""Number of images served by the fake cifar10 loader."""

FAKE_WIKITEXT_LINES = [
    "the quick brown fox jumps over the lazy dog",
    "",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
] * 5
"""Small fake corpus served by the fake wikitext loader."""

FAKE_DAC_LINES = [
    # label, 13 dense (some empty), 26 categorical (some empty)
    "0\t1\t2\t\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13" + "\t68fd1e64" * 25 + "\t",
    "1\t\t0\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13" + "\tabc123ff" * 26,
    "0\t5\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13" + "\tdeadbeef" * 26,
]
"""Fake Criteo DAC sample lines served by the fake criteo loader."""

FAKE_COCO_COUNT = 20
"""Number of image/caption pairs served by the fake coco loader."""


@pytest.fixture
def fake_cifar(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """Replace the TFDS cifar10 loader with a deterministic fake."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (FAKE_CIFAR_COUNT, 32, 32, 3), dtype=np.uint8)

    def _fake_load(data_dir: str | None, allow_download: bool) -> np.ndarray:
        return images

    monkeypatch.setattr(real_data, "_load_cifar10_train", _fake_load)
    return images


@pytest.fixture
def fake_wikitext(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the wikitext loader with a small fake corpus."""

    def _fake_load(allow_download: bool) -> Iterable[str]:
        return FAKE_WIKITEXT_LINES

    monkeypatch.setattr(real_data, "_load_wikitext_lines", _fake_load)
    return FAKE_WIKITEXT_LINES


@pytest.fixture
def fake_criteo(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the criteo loader with fake DAC-format lines."""

    def _fake_load(allow_download: bool) -> Iterable[str]:
        return FAKE_DAC_LINES

    monkeypatch.setattr(real_data, "_load_criteo_lines", _fake_load)
    return FAKE_DAC_LINES


@pytest.fixture
def fake_coco(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, str]]:
    """Replace the coco loader with fake variable-size image/caption pairs."""
    rng = np.random.default_rng(0)
    pairs = [
        (
            rng.integers(0, 256, (40 + i, 50 + i, 3), dtype=np.uint8),
            f"a photo of object number {i} on a table",
        )
        for i in range(FAKE_COCO_COUNT)
    ]

    def _fake_load(count: int, allow_download: bool) -> Iterator[tuple[np.ndarray, str]]:
        return iter(pairs[:count])

    monkeypatch.setattr(real_data, "_load_coco_rows", _fake_load)
    return pairs
