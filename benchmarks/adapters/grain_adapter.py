"""Grain adapter for the benchmark framework.

Wraps Google Grain's MapDataset API with the BenchmarkAdapter lifecycle.
Grain is a Tier 1 JAX ecosystem alternative -- already a base dependency.

Uses the MapDataset API (recommended for Grain v0.2+):
  source -> MapDataset.source() -> .map(transform) -> .batch(n) -> .to_iter_dataset()

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from functools import partial

from benchmarks.adapters import register
from benchmarks.adapters._utils import (
    apply_to_dict,
    cast_to_float32,
    gaussian_noise,
    normalize_uint8,
    random_brightness,
    random_scale,
)
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig


class _NumpyDataSource:
    """In-memory Grain-compatible random access data source.

    Implements the RandomAccessDataSource protocol: __len__ + __getitem__.
    Wraps a dict of numpy arrays as per-element dicts.
    """

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self._data = data
        self._key = next(iter(data))
        self._len = len(data[self._key])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {k: v[idx] for k, v in self._data.items()}


# ---------------------------------------------------------------------------
# Transform mapping: ScenarioConfig.transforms -> Grain .map() callables
# Uses shared array-level functions from _utils (DRY).
# ---------------------------------------------------------------------------

_GRAIN_TRANSFORMS: dict[str, Any] = {
    "Normalize": partial(apply_to_dict, normalize_uint8),
    "CastToFloat32": partial(apply_to_dict, cast_to_float32),
    "GaussianNoise": partial(apply_to_dict, gaussian_noise),
    "RandomBrightness": partial(apply_to_dict, random_brightness),
    "RandomScale": partial(apply_to_dict, random_scale),
}


@register
class GrainAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for Google Grain.

    Uses the MapDataset API (Grain v0.2+) for idiomatic pipeline construction.
    """

    def __init__(self) -> None:
        self._iterator: Any = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "Google Grain"

    @property
    def version(self) -> str:
        import grain

        return getattr(grain, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import grain  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Normalize + CastToFloat32
            "NLP-1",  # No transforms
            "TAB-1",  # Normalize on float32 (pass-through)
            "DIST-1",  # Normalize on float32
            "PR-1",  # Normalize on float32
            "AUG-1",  # Stochastic chain
            "AUG-2",  # Deterministic vs stochastic
            "AUG-3",  # Stochastic depth scaling
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        import grain

        source = _NumpyDataSource(data)

        ds = grain.MapDataset.source(source)

        for transform_name in config.transforms:
            fn = _GRAIN_TRANSFORMS.get(transform_name)
            if fn is not None:
                ds = ds.map(fn)

        ds = ds.batch(config.batch_size)

        self._iterator = ds.to_iter_dataset()
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._iterator

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        if isinstance(batch, dict):
            return [np.asarray(v) for v in batch.values()]
        return [np.asarray(batch)]

    def teardown(self) -> None:
        self._iterator = None
        self._config = None
