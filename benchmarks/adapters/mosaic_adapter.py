"""MosaicML StreamingDataset adapter for the benchmark framework.

Wraps MosaicML's StreamingDataset (MDS format) with the BenchmarkAdapter
lifecycle. Tier 2 cloud/streaming -- supports 5 scenarios.

Design ref: Section 14.2 of the benchmark report.

Background thread noise
~~~~~~~~~~~~~~~~~~~~~~~
MosaicML's StreamingDataset spawns a ``_prepare_thread`` that proactively
downloads shards.  In local-only mode (``remote=None``), this thread still
runs and raises ``ValueError`` from the download code path.  The error is
caught by ``concurrent.futures`` and logged at ERROR level, producing noisy
tracebacks that don't affect results.  We suppress this logger at import
time since the data is fully local.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

# Suppress noisy background-thread tracebacks from MosaicML's shard downloader.
logging.getLogger("concurrent.futures").setLevel(logging.CRITICAL)

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import cleanup_temp_dir, setup_temp_dir
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig


@register
class MosaicAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for MosaicML StreamingDataset.

    Requires MDS format conversion in setup(). Uses a temp directory for
    the converted data, cleaned up in teardown().
    """

    def __init__(self) -> None:
        self._dataset: Any = None
        self._config: ScenarioConfig | None = None
        self._tmp_dir: Any = None

    @property
    def name(self) -> str:
        return "MosaicML Streaming"

    @property
    def version(self) -> str:
        import streaming

        return getattr(streaming, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import streaming  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Raw streaming throughput (MDS format)
            "NLP-1",  # No transforms
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        from streaming import MDSWriter, StreamingDataset

        mds_dir, self._tmp_dir = setup_temp_dir(config, "mds")

        columns = {}
        for key, arr in data.items():
            dtype_str = str(arr.dtype)
            shape_str = ",".join(str(s) for s in arr.shape[1:])
            columns[key] = f"ndarray:{dtype_str}:{shape_str}"

        primary_key = next(iter(data))
        n = len(data[primary_key])
        with MDSWriter(out=str(mds_dir), columns=columns) as writer:
            for i in range(n):
                writer.write({k: v[i] for k, v in data.items()})

        self._dataset = StreamingDataset(
            local=str(mds_dir),
            remote=None,
            batch_size=config.batch_size,
            shuffle=False,
        )
        self._config = config

    def _iterate_batches(self) -> Iterator[list[Any]]:
        """Yield batches as lists of sample dicts (MosaicML is sample-based)."""
        batch_samples: list[Any] = []
        for sample in self._dataset:
            batch_samples.append(sample)
            if len(batch_samples) >= self._config.batch_size:
                yield batch_samples
                batch_samples = []

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        keys = list(batch[0].keys())
        return [np.stack([np.asarray(s[k]) for s in batch]) for k in keys]

    def teardown(self) -> None:
        self._dataset = None
        self._config = None
        cleanup_temp_dir(self._tmp_dir)
        self._tmp_dir = None
