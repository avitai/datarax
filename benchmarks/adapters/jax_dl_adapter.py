"""jax-dataloader adapter for the benchmark framework.

Wraps jax-dataloader's minimal DataLoader with the PipelineAdapter lifecycle.
Tier 1 JAX ecosystem -- lightweight, only supports CV-1.

Design ref: Section 14.2 of the benchmark report.

beartype/plum compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~
jax-dataloader uses plum for multiple dispatch and beartype for runtime type
checking.  When HuggingFace ``datasets`` is installed, plum tries to resolve
a forward reference (``"Dataset"``) in the HF overload via beartype's
``is_bearable()``, which raises ``BeartypeDecorHintForwardRefException``.

This is a known upstream bug (plum resolves *all* overloads eagerly, even
those for unrelated dataset types).  We work around it by constructing the
``DataLoaderJAX`` object directly via ``object.__new__()`` and manually
replicating its ``__init__`` logic, skipping both ``@typecheck`` and the
plum-dispatched ``to_jax_dataset()`` call.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import BASIC_TRANSFORMS
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


_JAX_DL_TRANSFORMS = BASIC_TRANSFORMS


def _create_jax_loader(dataset: Any, batch_size: int, shuffle: bool, drop_last: bool) -> Any:
    """Create a DataLoaderJAX bypassing beartype/plum type dispatch.

    Manually replicates ``DataLoaderJAX.__init__`` without calling
    ``to_jax_dataset()`` (which triggers the beartype crash) or the
    ``@typecheck``-decorated constructor.
    """
    from jax_dataloader.loaders.jax import DataLoaderJAX, Generator, get_config

    dataset.asnumpy()  # Convert to numpy (what to_jax_dataset does for ArrayDataset)

    loader = object.__new__(DataLoaderJAX)
    loader.dataset = dataset
    loader.indices = np.arange(len(dataset))
    loader.batch_size = batch_size
    loader.shuffle = shuffle
    loader.drop_last = drop_last
    generator = Generator().manual_seed(get_config().global_seed)
    loader.key = generator.jax_generator()
    return loader


@register
class JaxDataloaderAdapter(PipelineAdapter):
    """PipelineAdapter for jax-dataloader."""

    def __init__(self) -> None:
        """Initialize the jax-dataloader adapter."""
        super().__init__()
        self._loader: Any = None
        self._transform_fns: list[Any] = []

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "jax-dataloader"

    @property
    def version(self) -> str:
        """Return the jax-dataloader version string."""
        import jax_dataloader as jdl

        return getattr(jdl, "__version__", "unknown")

    def is_available(self) -> bool:
        """Return True if jax-dataloader is installed."""
        try:
            import jax_dataloader  # noqa: F401

            return True
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """Return the registry of available transform functions."""
        return set(_JAX_DL_TRANSFORMS)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the jax-dataloader pipeline for the given scenario."""
        import jax_dataloader as jdl

        primary_key = next(iter(data))
        arr = data[primary_key]
        dataset = jdl.ArrayDataset(arr)

        self._loader = _create_jax_loader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        self._transform_fns = [
            _JAX_DL_TRANSFORMS[t] for t in config.transforms if t in _JAX_DL_TRANSFORMS
        ]
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._loader

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arr = np.asarray(batch[0] if isinstance(batch, tuple) else batch)
        for fn in self._transform_fns:
            arr = fn(arr)
        return [arr]

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._loader = None
        self._transform_fns = []
        super().teardown()
