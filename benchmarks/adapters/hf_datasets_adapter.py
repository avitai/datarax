"""HuggingFace Datasets adapter for the benchmark framework.

Wraps HF Datasets' Arrow-based Dataset with the BenchmarkAdapter lifecycle.
Tier 3 specialized -- NLP and tabular.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import cast_to_float32, normalize_uint8
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig

_HF_TRANSFORMS: dict[str, Any] = {
    "Normalize": normalize_uint8,
    "CastToFloat32": cast_to_float32,
}


@register
class HfDatasetsAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for HuggingFace Datasets."""

    def __init__(self) -> None:
        self._dataset: Any = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "HuggingFace Datasets"

    @property
    def version(self) -> str:
        import datasets

        return datasets.__version__

    def is_available(self) -> bool:
        try:
            import datasets  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "NLP-1",  # No transforms (HF's strength: NLP)
            "TAB-1",  # Normalize on float32
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        from datasets import Dataset

        self._transform_fns = [_HF_TRANSFORMS[t] for t in config.transforms if t in _HF_TRANSFORMS]

        hf_data = {k: [v[i] for i in range(len(v))] for k, v in data.items()}
        self._dataset = Dataset.from_dict(hf_data)
        self._dataset.set_format("numpy")
        self._config = config

    def _iterate_batches(self) -> Iterator[dict]:
        batch_size = self._config.batch_size
        for i in range(0, len(self._dataset), batch_size):
            yield self._dataset[i : i + batch_size]

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arrays = [np.asarray(v) for v in batch.values()]
        for fn in self._transform_fns:
            arrays = [fn(a) for a in arrays]
        return arrays

    def teardown(self) -> None:
        self._dataset = None
        self._config = None
