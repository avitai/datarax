"""Mixed data source implementation for Datarax.

This module provides a data source that mixes elements from multiple child
sources according to configurable weights. Useful for combining heterogeneous
data streams (e.g., different image datasets, synthetic + real data).
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.config.registry import register_component
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixDataSourcesConfig(StructuralConfig):
    """Configuration for MixDataSourcesNode.

    Attributes:
        num_sources: Number of child sources (validated against actual sources)
        weights: Sampling weights per source (normalized automatically)
    """

    num_sources: int | None = None
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize mixed-source configuration fields."""
        # Validate required fields
        if self.num_sources is None:
            raise ValueError("num_sources is required")
        if self.weights is None:
            raise ValueError("weights is required")
        if self.num_sources < 1:
            raise ValueError("num_sources must be >= 1")
        if len(self.weights) != self.num_sources:
            raise ValueError(
                f"len(weights) ({len(self.weights)}) must match num_sources ({self.num_sources})"
            )
        if any(w < 0 for w in self.weights):
            raise ValueError("All weights must be positive (>= 0)")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")

        # Normalize weights to sum to 1.0
        normalized = tuple(w / total for w in self.weights)
        object.__setattr__(self, "weights", normalized)

        # Force stochastic=True (mixing requires RNG)
        object.__setattr__(self, "stochastic", True)
        object.__setattr__(self, "stream_name", "mix")

        # Call parent validation (validates stochastic config, then freezes)
        super().__post_init__()


@register_component("source", "MixDataSources")
class MixDataSourcesNode(DataSourceModule):
    """Mix multiple data sources with configurable weights.

    Sampling strategy: For each element, randomly select a source according
    to weights, then yield the next element from that source. If the chosen
    source is exhausted, fall back to a non-exhausted source.

    Total elements = sum of all source lengths.
    """

    def __init__(
        self,
        config: MixDataSourcesConfig,
        sources: list[DataSourceModule],
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ) -> None:
        """Initialize MixDataSourcesModule.

        Args:
            config: Configuration with num_sources, weights, and mixing seed.
            sources: List of data source modules to mix from.
            rngs: Flax NNX random number generators for sampling.
            name: Optional module name for identification.
        """
        if name is None:
            name = "MixDataSourcesNode"

        super().__init__(config, rngs=rngs, name=name)

        # Validate sources count matches config
        if len(sources) != config.num_sources:
            raise ValueError(
                f"len(sources) ({len(sources)}) must match "
                f"config.num_sources ({config.num_sources})"
            )

        # Use nnx.List so child modules are part of the NNX module graph
        self._sources = nnx.List(sources)
        self._weights = nnx.Variable(jnp.array(config.weights))
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)
        self._total_len = sum(len(s) for s in sources)

        # Iterators are created fresh on each __iter__ call (not NNX state)
        self._iterators: list[Iterator] = []
        # Track which sources are exhausted during iteration
        self._exhausted: list[bool] = []

    def __len__(self) -> int:
        """Return total elements across all child sources."""
        return self._total_len

    def __iter__(self) -> "MixDataSourcesNode":
        """Reset iterators and start a new epoch."""
        self.index.set_value(0)
        self.epoch.set_value(self.epoch.get_value() + 1)
        sources: list[DataSourceModule] = list(self._sources)
        self._iterators = [iter(s) for s in sources]
        self._exhausted = [False] * len(sources)
        return self

    def __next__(self) -> Any:
        """Sample a source by weight and yield the next element from it."""
        if self.index.get_value() >= self._total_len:
            raise StopIteration

        # rngs is guaranteed non-None by StructuralModule (stochastic=True)
        assert self.rngs is not None  # noqa: S101 (invariant, not control flow)
        # Sample source index according to weights
        rng_key = self.rngs.mix()
        weights = self._weights.get_value()
        source_idx = int(jax.random.choice(rng_key, len(self._sources), p=weights))

        # Try to get an element, falling back if the chosen source is exhausted
        element = self._try_next(source_idx)
        self.index.set_value(self.index.get_value() + 1)
        return element

    def _try_next(self, preferred: int) -> Any:
        """Get next element from preferred source, falling back to others."""
        # Try preferred source first
        if not self._exhausted[preferred]:
            try:
                return next(self._iterators[preferred])
            except StopIteration:
                self._exhausted[preferred] = True

        # Preferred is exhausted — try others in round-robin order
        n = len(self._sources)
        for offset in range(1, n):
            idx = (preferred + offset) % n
            if not self._exhausted[idx]:
                try:
                    return next(self._iterators[idx])
                except StopIteration:
                    self._exhausted[idx] = True

        # All sources exhausted (shouldn't happen if _total_len is correct)
        raise StopIteration

    def reset(self) -> None:
        """Reset all internal state and child sources to initial conditions."""
        self.index.set_value(0)
        self.epoch.set_value(0)
        self._iterators = []
        self._exhausted = []
        for s in self._sources:
            reset_fn = getattr(s, "reset", None)
            if reset_fn is not None:
                reset_fn()
