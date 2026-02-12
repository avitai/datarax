"""WebDataset adapter for the benchmark framework.

Wraps WebDataset's TAR-based streaming with the BenchmarkAdapter lifecycle.
Tier 2 cloud/streaming -- supports 3 scenarios.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import cleanup_temp_dir, setup_temp_dir, write_numpy_tar
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig


@register
class WebDatasetAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for WebDataset.

    Requires TAR shard conversion in setup(). Uses numpy serialization
    (.npy extension) for fair comparison with in-memory loaders.
    """

    def __init__(self) -> None:
        self._dataset: Any = None
        self._config: ScenarioConfig | None = None
        self._tmp_dir: Any = None

    @property
    def name(self) -> str:
        return "WebDataset"

    @property
    def version(self) -> str:
        import webdataset

        return getattr(webdataset, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import webdataset  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Raw streaming throughput (TAR format)
            "NLP-1",  # No transforms
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        import webdataset as wds

        shard_dir, self._tmp_dir = setup_temp_dir(config, "shards")

        shard_path = shard_dir / "shard-000000.tar"
        write_numpy_tar(data, shard_path)

        keys = list(data.keys())

        def _decode_npy(sample: dict) -> dict:
            result = {}
            for k in keys:
                npy_key = f"{k}.npy"
                if npy_key in sample:
                    result[k] = np.load(io.BytesIO(sample[npy_key]))
            result["__key__"] = sample.get("__key__", "")
            return result

        self._dataset = (
            wds.WebDataset(str(shard_path), shardshuffle=False)
            .map(_decode_npy)
            .batched(config.batch_size)
        )
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._dataset

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        if isinstance(batch, list):
            keys = [k for k in batch[0] if k != "__key__"]
            return [np.stack([s[k] for s in batch]) for k in keys]
        if isinstance(batch, dict):
            keys = [k for k in batch if k != "__key__"]
            return [np.asarray(batch[k]) for k in keys]
        return [np.asarray(batch)]

    def teardown(self) -> None:
        self._dataset = None
        self._config = None
        cleanup_temp_dir(self._tmp_dir)
        self._tmp_dir = None
