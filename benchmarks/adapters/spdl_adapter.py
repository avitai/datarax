"""Meta SPDL adapter for the benchmark framework.

Wraps SPDL's thread-based DataLoader with the BenchmarkAdapter lifecycle.
Tier 2 -- key I/O performance baseline.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import (
    cast_to_float32,
    gaussian_noise,
    normalize_uint8,
    random_brightness,
    random_scale,
)
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig

_SPDL_TRANSFORMS: dict[str, Any] = {
    "Normalize": normalize_uint8,
    "CastToFloat32": cast_to_float32,
    "GaussianNoise": gaussian_noise,
    "RandomBrightness": random_brightness,
    "RandomScale": random_scale,
}


@register
class SpdlAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for Meta SPDL.

    SPDL uses threads (not processes) for I/O -- lower overhead on CPU-bound
    workloads.  Uses ``spdl.dataloader.DataLoader`` for index-based iteration
    over in-memory numpy arrays.
    """

    def __init__(self) -> None:
        self._loader: Any = None
        self._data: dict[str, np.ndarray] | None = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "SPDL"

    @property
    def version(self) -> str:
        import spdl

        return getattr(spdl, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import spdl.dataloader  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Normalize + CastToFloat32
            "NLP-1",  # No transforms
            "TAB-1",  # Normalize on float32
            "DIST-1",  # Normalize on float32
            "AUG-1",  # Stochastic chain
            "AUG-2",  # Deterministic vs stochastic
            "AUG-3",  # Stochastic depth scaling
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        import spdl.dataloader

        self._data = data
        self._config = config
        self._transform_fns = [
            _SPDL_TRANSFORMS[t] for t in config.transforms if t in _SPDL_TRANSFORMS
        ]
        self._loader = spdl.dataloader.DataLoader(
            list(range(len(data[next(iter(data))]))),
            batch_size=config.batch_size,
            num_threads=max(config.num_workers, 1),
        )

    def _iterate_batches(self) -> Iterator[dict[str, np.ndarray]]:
        for batch_indices in self._loader:
            indices = np.asarray(batch_indices)
            yield {k: v[indices] for k, v in self._data.items()}

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arrays = [np.asarray(v) for v in batch.values()]
        for fn in self._transform_fns:
            arrays = [fn(a) for a in arrays]
        return arrays

    def teardown(self) -> None:
        self._loader = None
        self._data = None
        self._config = None
