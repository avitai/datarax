"""HuggingFace Datasets adapter for the benchmark framework.

Wraps HF Datasets' Arrow-based Dataset with the PipelineAdapter lifecycle.
Tier 3 specialized -- NLP and tabular.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import BASIC_TRANSFORMS
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


_HF_TRANSFORMS = BASIC_TRANSFORMS


@register
class HfDatasetsAdapter(PipelineAdapter):
    """PipelineAdapter for HuggingFace Datasets."""

    def __init__(self) -> None:
        """Initialize the HuggingFace Datasets adapter."""
        super().__init__()
        self._dataset: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "HuggingFace Datasets"

    @property
    def version(self) -> str:
        """Return the HuggingFace Datasets version string."""
        import datasets

        return datasets.__version__

    def is_available(self) -> bool:
        """Return True if HuggingFace Datasets is installed."""
        try:
            import datasets  # noqa: F401

            return True
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """Return the registry of available transform functions."""
        return set(_HF_TRANSFORMS)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the HuggingFace Datasets pipeline for the given scenario."""
        from datasets import Dataset

        self._transform_fns = [_HF_TRANSFORMS[t] for t in config.transforms if t in _HF_TRANSFORMS]

        hf_data = {k: [v[i] for i in range(len(v))] for k, v in data.items()}
        self._dataset = Dataset.from_dict(hf_data)
        self._dataset.set_format("numpy")
        self._config = config

    def _iterate_batches(self) -> Iterator[dict]:
        assert self._config is not None
        batch_size = self._config.batch_size
        for i in range(0, len(self._dataset), batch_size):
            yield self._dataset[i : i + batch_size]

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arrays = [np.asarray(v) for v in batch.values()]
        for fn in self._transform_fns:
            arrays = [fn(a) for a in arrays]
        return arrays

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._dataset = None
        super().teardown()
