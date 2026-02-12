"""FFCV adapter for the benchmark framework.

Wraps FFCV's .beton format loader with the BenchmarkAdapter lifecycle.
Tier 2 vision-only -- supports only CV-1.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import cast_to_float32, normalize_uint8
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig

_FFCV_TRANSFORMS: dict[str, Any] = {
    "Normalize": normalize_uint8,
    "CastToFloat32": cast_to_float32,
}


@register
class FfcvAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for FFCV.

    FFCV requires .beton format. setup() writes synthetic data to a temp
    .beton file, teardown() cleans it up.
    """

    def __init__(self) -> None:
        self._loader: Any = None
        self._config: ScenarioConfig | None = None
        self._beton_path: Path | None = None
        self._tmp_dir: Any = None
        self._transform_fns: list[Any] = []

    @property
    def name(self) -> str:
        return "FFCV"

    @property
    def version(self) -> str:
        import ffcv

        return getattr(ffcv, "__version__", "unknown")

    def is_available(self) -> bool:
        try:
            import ffcv  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {"CV-1"}

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        from ffcv.fields import NDArrayField
        from ffcv.fields.basics import IntField
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor
        from ffcv.writer import DatasetWriter

        self._tmp_dir = tempfile.TemporaryDirectory()
        self._beton_path = Path(self._tmp_dir.name) / "benchmark.beton"

        primary_key = next(iter(data))
        images = data[primary_key]

        writer = DatasetWriter(
            str(self._beton_path),
            {
                "image": NDArrayField(dtype=images.dtype, shape=images.shape[1:]),
                "label": IntField(),
            },
        )

        class _IndexedDataset:
            def __init__(self, arr: np.ndarray) -> None:
                self._arr = arr

            def __len__(self) -> int:
                return len(self._arr)

            def __getitem__(self, idx: int) -> tuple:
                return (self._arr[idx], 0)

        writer.from_indexed_dataset(_IndexedDataset(images))

        self._loader = Loader(
            str(self._beton_path),
            batch_size=config.batch_size,
            order=OrderOption.SEQUENTIAL,
            pipelines={"image": [ToTensor()], "label": [ToTensor()]},
        )
        self._transform_fns = [
            _FFCV_TRANSFORMS[t] for t in config.transforms if t in _FFCV_TRANSFORMS
        ]
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._loader

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        # batch is a tuple of tensors (image, label)
        arr = batch[0].numpy()
        for fn in self._transform_fns:
            arr = fn(arr)
        return [arr]

    def teardown(self) -> None:
        self._loader = None
        self._config = None
        self._beton_path = None
        self._transform_fns = []
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
