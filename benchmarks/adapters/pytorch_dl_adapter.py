"""PyTorch DataLoader adapter for the benchmark framework.

Wraps torch.utils.data.DataLoader with the BenchmarkAdapter lifecycle.
Tier 2 universal baseline.

Design ref: Section 7.3 of the benchmark report.
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

_PYTORCH_TRANSFORMS: dict[str, Any] = {
    "Normalize": normalize_uint8,
    "CastToFloat32": cast_to_float32,
    "GaussianNoise": gaussian_noise,
    "RandomBrightness": random_brightness,
    "RandomScale": random_scale,
}


class _DictDataset:
    """Wraps a dict of numpy arrays as a PyTorch-style map dataset."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self._data = data
        self._key = next(iter(data))
        self._len = len(data[self._key])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        return {k: torch.from_numpy(np.asarray(v[idx])) for k, v in self._data.items()}


@register
class PyTorchDataLoaderAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for PyTorch DataLoader."""

    def __init__(self) -> None:
        self._loader: Any = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "PyTorch DataLoader"

    @property
    def version(self) -> str:
        import torch

        return torch.__version__

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Normalize + CastToFloat32
            "NLP-1",  # No transforms
            "TAB-1",  # Normalize on float32
            "DIST-1",  # Normalize on float32
            "PR-1",  # Normalize on float32
            "AUG-1",  # Stochastic chain
            "AUG-2",  # Deterministic vs stochastic
            "AUG-3",  # Stochastic depth scaling
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        import torch
        from torch.utils.data import DataLoader

        # Resolve transforms from config
        self._transform_fns = [
            _PYTORCH_TRANSFORMS[t] for t in config.transforms if t in _PYTORCH_TRANSFORMS
        ]

        dataset = _DictDataset(data)

        # Optimal settings per Section 8.6 Fairness Principles
        use_workers = config.num_workers > 0
        self._loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            persistent_workers=use_workers,
        )
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._loader

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(batch, dict):
            arrays = [v.numpy() for v in batch.values()]
        else:
            arrays = [batch.numpy()]

        for fn in self._transform_fns:
            arrays = [fn(a) for a in arrays]
        return arrays

    def teardown(self) -> None:
        self._loader = None
        self._config = None
