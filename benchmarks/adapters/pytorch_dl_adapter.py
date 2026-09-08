"""PyTorch DataLoader adapter for the benchmark framework.

Wraps torch.utils.data.DataLoader with the PipelineAdapter lifecycle.
Tier 2 universal baseline.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import resolve_transforms, STANDARD_TRANSFORMS
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


_PYTORCH_TRANSFORMS = STANDARD_TRANSFORMS


def _dict_dataset(data: dict[str, np.ndarray]) -> Any:
    """Wrap a dict of numpy arrays as a PyTorch map-style ``Dataset``.

    Defined inside the function so the module imports without torch installed;
    the adapter reports itself unavailable in that case.
    """
    import torch
    from torch.utils.data import Dataset

    class _DictDataset(Dataset[dict[str, Any]]):
        def __init__(self, arrays: dict[str, np.ndarray]) -> None:
            self._data = arrays
            self._len = len(arrays[next(iter(arrays))])

        def __len__(self) -> int:
            return self._len

        def __getitem__(self, idx: int) -> dict[str, Any]:
            # torch re-exports its constructors from torch._C without listing them in
            # __all__, so pyright reads every one of them as private.
            return {
                k: torch.from_numpy(np.asarray(v[idx]))  # pyright: ignore[reportPrivateImportUsage]
                for k, v in self._data.items()
            }

    return _DictDataset(data)


@register
class PyTorchDataLoaderAdapter(PipelineAdapter):
    """PipelineAdapter for PyTorch DataLoader."""

    def __init__(self) -> None:
        """Initialize the PyTorch DataLoader adapter."""
        super().__init__()
        self._loader: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "PyTorch DataLoader"

    @property
    def version(self) -> str:
        """Return the PyTorch version string."""
        import torch

        return torch.__version__

    def is_available(self) -> bool:
        """Return True if PyTorch is installed."""
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """All transforms PyTorch DataLoader can execute."""
        return set(_PYTORCH_TRANSFORMS)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the PyTorch DataLoader for the given scenario configuration."""
        import torch
        from torch.utils.data import DataLoader

        # Resolve transforms from config
        self._transform_fns = resolve_transforms(config.transforms, _PYTORCH_TRANSFORMS, self.name)

        dataset = _dict_dataset(data)

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
        """Release resources and reset adapter state."""
        self._loader = None
        super().teardown()
