"""Meta SPDL adapter for the benchmark framework.

Wraps SPDL's thread-based DataLoader with the PipelineAdapter lifecycle.
Tier 2 -- key I/O performance baseline.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import STANDARD_TRANSFORMS
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


_SPDL_TRANSFORMS = STANDARD_TRANSFORMS


@register
class SpdlAdapter(PipelineAdapter):
    """PipelineAdapter for Meta SPDL.

    SPDL uses threads (not processes) for I/O -- lower overhead on CPU-bound
    workloads.  Uses ``spdl.dataloader.DataLoader`` for index-based iteration
    over in-memory numpy arrays.
    """

    def __init__(self) -> None:
        """Initialize the SPDL adapter."""
        super().__init__()
        self._loader: Any = None
        self._data: dict[str, np.ndarray] | None = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "SPDL"

    @property
    def version(self) -> str:
        """Return the SPDL version string."""
        import spdl

        return getattr(spdl, "__version__", "unknown")

    def is_available(self) -> bool:
        """Return True if SPDL is installed."""
        try:
            import spdl.dataloader  # noqa: F401

            return True
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """All transforms SPDL can execute."""
        return set(_SPDL_TRANSFORMS)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the SPDL pipeline for the given scenario configuration."""
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
        assert self._data is not None
        for batch_indices in self._loader:
            indices = np.asarray(batch_indices)
            yield {k: v[indices] for k, v in self._data.items()}

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arrays = [np.asarray(v) for v in batch.values()]
        for fn in self._transform_fns:
            arrays = [fn(a) for a in arrays]
        return arrays

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._loader = None
        self._data = None
        super().teardown()
