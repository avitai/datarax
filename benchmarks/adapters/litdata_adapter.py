"""LitData adapter for the benchmark framework.

Wraps Lightning's LitData StreamingDataset with the BenchmarkAdapter lifecycle.
Tier 3 -- supports 2 scenarios. Requires format conversion in setup().

Design ref: Section 14.2 of the benchmark report.

Pickle compatibility
~~~~~~~~~~~~~~~~~~~~
``litdata.optimize()`` uses multiprocessing to process samples in parallel.
Python's ``multiprocessing`` requires that the worker function is picklable,
which excludes closures and lambdas.  We use a module-level callable class
(``_SampleExtractor``) instead of a local closure to satisfy this constraint.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import cleanup_temp_dir, setup_temp_dir
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig


class _SampleExtractor:
    """Picklable callable that extracts a single sample from a data dict.

    Used as the ``fn`` argument to ``litdata.optimize()``, which requires
    picklability for multiprocessing workers.
    """

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self._data = data

    def __call__(self, idx: int) -> dict[str, np.ndarray]:
        return {k: v[idx] for k, v in self._data.items()}


@register
class LitDataAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for Lightning LitData.

    Requires format conversion via optimize() in setup().
    """

    def __init__(self) -> None:
        self._loader: Any = None
        self._config: ScenarioConfig | None = None
        self._tmp_dir: Any = None

    @property
    def name(self) -> str:
        return "LitData"

    @property
    def version(self) -> str:
        import litdata

        return getattr(litdata, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import litdata  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {"CV-1"}  # Raw streaming throughput (LitData format)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        from litdata import StreamingDataLoader, StreamingDataset, optimize

        output_dir, self._tmp_dir = setup_temp_dir(config, "litdata")

        primary_key = next(iter(data))
        n = len(data[primary_key])

        sample_fn = _SampleExtractor(data)

        # Suppress tqdm progress bars in the multiprocessing worker.
        # verbose=False handles litdata's own logging; TQDM_DISABLE
        # handles the tqdm bars spawned in the worker subprocess.
        prev_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            optimize(
                fn=sample_fn,
                inputs=list(range(n)),
                output_dir=str(output_dir),
                num_workers=1,
                chunk_bytes="64MB",
                verbose=False,
            )
        finally:
            if prev_tqdm is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm

        dataset = StreamingDataset(input_dir=str(output_dir))
        self._loader = StreamingDataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
        )
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._loader

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        if isinstance(batch, dict):
            return [np.asarray(v) for v in batch.values()]
        return [np.asarray(batch)]

    def teardown(self) -> None:
        self._loader = None
        self._config = None
        cleanup_temp_dir(self._tmp_dir)
        self._tmp_dir = None
