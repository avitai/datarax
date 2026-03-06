"""Grain adapter for the benchmark framework.

Wraps Google Grain's MapDataset API with the PipelineAdapter lifecycle.
Grain is a Tier 1 JAX ecosystem alternative -- already a base dependency.

Uses the MapDataset API (recommended for Grain v0.2+):
  source -> MapDataset.source() -> .map(transform) -> .batch(n) -> .to_iter_dataset()

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import apply_to_dict, STANDARD_TRANSFORMS
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


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

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {k: v[index] for k, v in self._data.items()}


# ---------------------------------------------------------------------------
# Transform mapping: ScenarioConfig.transforms -> Grain .map() callables
# Wraps STANDARD_TRANSFORMS with apply_to_dict for per-element dict processing.
# ---------------------------------------------------------------------------

_GRAIN_TRANSFORMS: dict[str, Any] = {
    name: partial(apply_to_dict, fn) for name, fn in STANDARD_TRANSFORMS.items()
}


@register
class GrainAdapter(PipelineAdapter):
    """PipelineAdapter for Google Grain.

    Uses the MapDataset API (Grain v0.2+) for idiomatic pipeline construction.
    """

    def __init__(self) -> None:
        """Initialize the Grain adapter."""
        super().__init__()
        self._iterator: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "Google Grain"

    @property
    def version(self) -> str:
        """Return the Grain version string."""
        import grain

        return getattr(grain, "__version__", "unknown")

    def is_available(self) -> bool:
        """Return True if Grain is installed."""
        try:
            import grain  # noqa: F401
            import grain.sources  # noqa: F401

            return hasattr(grain, "MapDataset")
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """All transforms Grain can execute via _GRAIN_TRANSFORMS."""
        return set(_GRAIN_TRANSFORMS)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the Grain pipeline for the given scenario configuration."""
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
        """Release resources and reset adapter state."""
        self._iterator = None
        super().teardown()
