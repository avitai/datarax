"""NVIDIA DALI adapter for the benchmark framework.

Wraps NVIDIA DALI's GPU-accelerated pipeline with the BenchmarkAdapter
lifecycle. Tier 2 -- requires CUDA.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig


class _ExternalSourceIterator:
    """Feeds in-memory numpy data to DALI's external_source operator."""

    def __init__(self, data: np.ndarray, batch_size: int) -> None:
        self._data = data
        self._batch_size = batch_size
        self._idx = 0

    def __iter__(self) -> _ExternalSourceIterator:
        self._idx = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._idx >= len(self._data):
            raise StopIteration
        end = min(self._idx + self._batch_size, len(self._data))
        batch = self._data[self._idx : end]
        self._idx = end
        return batch


@register
class DaliAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for NVIDIA DALI."""

    def __init__(self) -> None:
        self._pipe: Any = None
        self._iterator: Any = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "NVIDIA DALI"

    @property
    def version(self) -> str:
        import nvidia.dali as dali

        return getattr(dali, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import nvidia.dali  # noqa: F401

            # DALI requires CUDA
            import nvidia.dali.fn  # noqa: F401

            return True
        except (ImportError, RuntimeError):
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Normalize + CastToFloat32 on uint8 images
            "NLP-1",  # No transforms (pure iteration)
            "TAB-1",  # Normalize on float32 (pass-through)
            "DIST-1",  # Normalize on float32
            "AUG-1",  # Stochastic chain
            "AUG-2",  # Deterministic vs stochastic
            "AUG-3",  # Stochastic depth scaling
        }

    @staticmethod
    def _numpy_to_dali_dtype(np_dtype: np.dtype) -> Any:
        """Map numpy dtype to DALI DALIDataType."""
        from nvidia.dali import types

        _DTYPE_MAP = {
            np.dtype("uint8"): types.UINT8,
            np.dtype("int32"): types.INT32,
            np.dtype("int64"): types.INT64,
            np.dtype("float32"): types.FLOAT,
            np.dtype("float64"): types.FLOAT64,
        }
        return _DTYPE_MAP.get(np_dtype, types.FLOAT)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        primary_key = next(iter(data))
        source_data = data[primary_key]
        dali_dtype = self._numpy_to_dali_dtype(source_data.dtype)
        source_iter = _ExternalSourceIterator(source_data, config.batch_size)

        @pipeline_def(
            batch_size=config.batch_size, num_threads=max(config.num_workers, 1), device_id=0
        )
        def pipe():
            images = fn.external_source(source=source_iter, dtype=dali_dtype)
            for t_name in config.transforms:
                if t_name == "Normalize":
                    images = fn.normalize(images)
                elif t_name == "CastToFloat32":
                    images = fn.cast(images, dtype=types.FLOAT)
                elif t_name == "GaussianNoise":
                    noise = fn.random.normal(images, stddev=0.05)
                    images = images + noise
                elif t_name == "RandomBrightness":
                    images = fn.brightness(images, brightness=fn.random.uniform(range=(-0.2, 0.2)))
                elif t_name == "RandomScale":
                    images = images * fn.random.uniform(range=(0.8, 1.2))
            return images

        self._pipe = pipe()
        self._pipe.build()

        self._iterator = DALIGenericIterator(
            self._pipe,
            ["data"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        self._config = config

    def warmup(self, num_batches: int = 3) -> None:
        """DALI requires an iterator reset after warmup."""
        super().warmup(num_batches)
        self._iterator.reset()

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._iterator

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        # DALI returns list of dicts; extract the primary tensor
        return [batch[0]["data"].cpu().numpy()]

    def teardown(self) -> None:
        self._iterator = None
        self._pipe = None
        self._config = None
