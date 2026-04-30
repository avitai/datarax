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
import jax
import jax.numpy as jnp

from datarax.config.registry import register_component
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.sources._grain_streaming import data_source_to_iter_dataset, mix_streaming_sources


logger = logging.getLogger(__name__)


def _validate_compatible_element_specs(sources: list[DataSourceModule]) -> None:
    """Verify that every source produces records with the same element_spec.

    Required so that the per-position ``lax.switch`` dispatch in
    ``get_batch_at`` has branches with matching output shapes — JAX
    rejects ``lax.switch`` calls whose branches return differently-
    shaped pytrees.

    Args:
        sources: Sources to check.

    Raises:
        ValueError: If any two sources produce records with different
            ``element_spec`` structure (different keys, shapes, or dtypes).
    """
    if not sources:
        return

    # Only validate sources that can produce a numeric element_spec. Sources
    # holding non-JAX data (string labels, Python objects) bypass this check
    # — they still flow through the iterator/__next__ path; ``get_batch_at``
    # is the only path that requires matching specs (because of lax.switch).
    try:
        reference = sources[0].element_spec()
    except Exception:  # noqa: BLE001 — opportunistic compatibility check
        return

    for index, source in enumerate(sources[1:], start=1):
        try:
            other = source.element_spec()
        except Exception:  # noqa: BLE001 — opportunistic compatibility check  # nosec B112
            continue
        if jax.tree.structure(reference) != jax.tree.structure(other):
            raise ValueError(
                f"MixDataSourcesNode requires every source to produce records "
                f"with the same element_spec; source 0 and source {index} have "
                f"different structure. Mixing under lax.switch needs matching "
                f"output shapes across all branches."
            )
        ref_leaves = jax.tree.leaves(reference)
        other_leaves = jax.tree.leaves(other)
        for leaf_a, leaf_b in zip(ref_leaves, other_leaves):
            if not (hasattr(leaf_a, "shape") and hasattr(leaf_b, "shape")):
                continue
            if leaf_a.shape != leaf_b.shape or leaf_a.dtype != leaf_b.dtype:
                raise ValueError(
                    f"MixDataSourcesNode requires every source to produce records "
                    f"with the same element_spec; source 0 leaf has shape="
                    f"{leaf_a.shape} dtype={leaf_a.dtype} but source {index} leaf "
                    f"has shape={leaf_b.shape} dtype={leaf_b.dtype}."
                )


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

        # Validate that every source produces records with the same element_spec.
        # This is the constraint that lets get_batch_at use lax.switch — every
        # branch must produce identically-shaped records.
        _validate_compatible_element_specs(sources)

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

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Stateless weighted-interleave batch access for ``Pipeline``-driven iteration.

        Each output position deterministically chooses a source via
        weighted categorical sampling derived from ``key`` and the
        absolute position, picks a local index uniformly within that
        source, and dispatches to that source's own ``get_batch_at``.

        Algorithm (per output position ``p``):

        1. ``pos_key = jax.random.fold_in(key, start + p)`` — deterministic.
        2. Split ``pos_key`` into ``(src_key, idx_key, fetch_key)``.
        3. ``chosen_src = jax.random.categorical(src_key, log_weights)``.
        4. ``local_idx = jax.random.randint(idx_key, 0, len(sources[chosen_src]))``.
        5. ``record = lax.switch(chosen_src, [s.get_batch_at(li, 1, fk) for s in sources])``.

        The same ``(start, size, key)`` always returns the same output
        — no internal counters are mutated. ``vmap`` over positions
        builds the full batch in one trace.

        Args:
            start: Starting logical position (int or traced ``jax.Array``).
            size: Number of records to return.
            key: PRNG key for deterministic source / index selection.
                Required — mixing without a key has no defined semantics.

        Returns:
            Dict mapping each data key to a JAX array with leading dim
            ``size``, drawn from the underlying sources in proportion
            to ``self._weights``.

        Raises:
            ValueError: If ``key is None``.
        """
        if key is None:
            raise ValueError(
                "MixDataSourcesNode.get_batch_at requires a PRNG key for "
                "deterministic mixing. Pass `key=jax.random.key(seed)` or "
                "drive iteration via Pipeline (which threads its own rngs)."
            )

        log_weights = jnp.log(jnp.asarray(self._weights, dtype=jnp.float32))
        source_lengths = jnp.asarray([len(s) for s in self._sources], dtype=jnp.int32)
        sources = list(self._sources)

        start_arr = jnp.asarray(start, dtype=jnp.int32)
        positions = start_arr + jnp.arange(size, dtype=jnp.int32)

        def _fetch_one(position: jax.Array) -> dict[str, jax.Array]:
            pos_key = jax.random.fold_in(key, position)
            src_key, idx_key, fetch_key = jax.random.split(pos_key, 3)

            chosen_src = jax.random.categorical(src_key, log_weights)
            chosen_length = source_lengths[chosen_src]
            local_idx = jax.random.randint(idx_key, (), 0, chosen_length)

            # Each branch fetches a single record from one source.
            # All branches share the same output shape (validated at
            # construction by _validate_compatible_element_specs).
            branches = [lambda li, fk, src=src: src.get_batch_at(li, 1, fk) for src in sources]
            record = jax.lax.switch(chosen_src, branches, local_idx, fetch_key)
            # Each source returned a batch of size 1; squeeze the leading axis.
            return jax.tree.map(lambda x: x[0], record)

        return jax.vmap(_fetch_one)(positions)
