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
import grain

from datarax.config.registry import register_component
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.sources._grain_streaming import data_source_to_iter_dataset, mix_streaming_sources


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

        object.__setattr__(self, "stochastic", False)
        object.__setattr__(self, "stream_name", None)

        # Call parent validation (validates stochastic config, then freezes)
        super().__post_init__()


@register_component("source", "MixDataSources")
class MixDataSourcesNode(DataSourceModule):
    """Mix multiple data sources with configurable weights.

    Sampling strategy is delegated to Grain's weighted ``IterDataset.mix``.

    Total elements = sum of all source lengths.
    """

    def __init__(
        self,
        config: MixDataSourcesConfig,
        sources: list[DataSourceModule],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize MixDataSourcesModule.

        Args:
            config: Configuration with num_sources and weights.
            sources: List of data source modules to mix from.
            rngs: Optional Flax NNX random number generators.
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

        weights = config.weights
        if weights is None:
            raise ValueError("weights is required")

        self._sources = nnx.List(sources)
        self._weights = tuple(weights)
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)
        self._total_len = sum(len(s) for s in sources)
        self._iterator: Iterator[Any] | None = None

    def __len__(self) -> int:
        """Return total elements across all child sources."""
        return self._total_len

    def __iter__(self) -> "MixDataSourcesNode":
        """Reset iterators and start a new epoch."""
        self.index.set_value(0)
        self.epoch.set_value(self.epoch.get_value() + 1)
        self._iterator = iter(self.to_grain_iter_dataset())
        return self

    def __next__(self) -> Any:
        """Sample a source by weight and yield the next element from it."""
        if self.index.get_value() >= self._total_len:
            raise StopIteration

        if self._iterator is None:
            self._iterator = iter(self.to_grain_iter_dataset())
        element = next(self._iterator)
        self.index.set_value(self.index.get_value() + 1)
        return element

    def to_grain_iter_dataset(self) -> grain.IterDataset:
        """Return the Grain mixed streaming dataset backing this source."""
        return mix_streaming_sources(
            [data_source_to_iter_dataset(source) for source in self._sources],
            weights=self._weights,
        )

    def reset(self) -> None:
        """Reset all internal state and child sources to initial conditions."""
        self.index.set_value(0)
        self.epoch.set_value(0)
        self._iterator = None
        for s in self._sources:
            reset_fn = getattr(s, "reset", None)
            if reset_fn is not None:
                reset_fn()
