"""Deep Lake adapter for the benchmark framework.

Wraps Deep Lake's tensor database with the PipelineAdapter lifecycle.
Tier 3 specialized -- only supports CV-1.

Design ref: Section 14.2 of the benchmark report.

Deep Lake v4 API
~~~~~~~~~~~~~~~~
Version 4.x replaced the v3 ``deeplake.dataset()`` API entirely:
- ``deeplake.create(path)`` creates a new dataset at a filesystem path
- ``ds.add_column(name, type)`` defines typed columns
- ``ds.append([{col: value, ...}])`` appends rows
- ``ds[start:end]["col"]`` reads a column slice as numpy
- ``deeplake.delete(path)`` removes a dataset

There is no in-memory mode in v4 -- a temporary directory is used instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import BASIC_TRANSFORMS, setup_temp_dir
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


_DEEPLAKE_TRANSFORMS = BASIC_TRANSFORMS


@register
class DeepLakeAdapter(PipelineAdapter):
    """PipelineAdapter for Deep Lake v4.

    Uses a temporary directory for storage (v4 has no in-memory mode).
    """

    def __init__(self) -> None:
        """Initialize the Deep Lake adapter."""
        super().__init__()
        self._ds: Any = None
        self._ds_path: str | None = None
        self._tmp_dir: Any = None
        self._columns: list[str] = []
        self._transform_fns: list[Any] = []

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "Deep Lake"

    @property
    def version(self) -> str:
        """Return the Deep Lake version string."""
        import deeplake

        return getattr(deeplake, "__version__", "unknown")

    def is_available(self) -> bool:
        """Return True if Deep Lake is installed."""
        try:
            import deeplake  # noqa: F401

            return True
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """Return the registry of available transform functions."""
        return set(_DEEPLAKE_TRANSFORMS)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the Deep Lake pipeline for the given scenario configuration."""
        import deeplake

        data_dir, self._tmp_dir = setup_temp_dir(config, "deeplake")
        ds_path = str(data_dir / "ds")
        self._ds_path = ds_path

        ds = deeplake.create(ds_path)

        columns = []
        for key, arr in data.items():
            dtype_str = str(arr.dtype)
            ndim = arr.ndim - 1  # subtract the dataset dimension
            ds.add_column(key, deeplake.types.Array(dtype_str, dimensions=ndim))
            columns.append(key)

        rows = [{key: data[key][i] for key in columns} for i in range(len(data[columns[0]]))]
        ds.append(rows)
        ds.commit()

        self._ds = ds
        self._columns = columns
        self._transform_fns = [
            _DEEPLAKE_TRANSFORMS[t] for t in config.transforms if t in _DEEPLAKE_TRANSFORMS
        ]
        self._config = config

    def _iterate_batches(self) -> Iterator[dict[str, Any]]:
        assert self._config is not None
        batch_size = self._config.batch_size
        n = len(self._ds)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield {col: self._ds[start:end][col] for col in self._columns}

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arrays = [np.asarray(v) for v in batch.values()]
        for fn in self._transform_fns:
            arrays = [fn(a) for a in arrays]
        return arrays

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        import deeplake

        ds_path = self._ds_path
        self._ds = None
        self._columns = []
        self._transform_fns = []
        self._ds_path = None

        if ds_path is not None:
            try:
                deeplake.delete(ds_path)
            except (ImportError, OSError):
                pass

        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

        super().teardown()
