"""Mixed data source implementation for Datarax.

This module provides a data source that mixes elements from multiple child
sources according to configurable weights. Useful for combining heterogeneous
data streams (e.g., different image datasets, synthetic + real data).
"""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import grain
import jax
import jax.numpy as jnp
from flax import nnx

from datarax.config.registry import register_component
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.spec import spec_mismatches, SpecMismatchError
from datarax.sources._grain_streaming import data_source_to_iter_dataset, mix_streaming_sources


logger = logging.getLogger(__name__)


def _validate_compatible_element_specs(sources: Sequence[DataSourceModule]) -> None:  # noqa: DOC502
    """Verify that every source produces records with the same element_spec.

    Required so that the per-record ``lax.switch`` dispatch in
    ``get_records`` has branches with matching output shapes — JAX
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
    # — they still flow through the iterator/__next__ path; ``get_records``
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
        _assert_specs_match(reference, other, index)


def _assert_specs_match(reference: Any, other: Any, index: int) -> None:
    """Raise if ``other``'s element_spec differs from ``reference`` in structure or leaves.

    Args:
        reference: Element spec of source 0 (the comparison baseline).
        other: Element spec of the source at position ``index``.
        index: Position of ``other`` among the sources, for error messages.

    Raises:
        SpecMismatchError: Naming every field where the two specs differ in
            structure, shape or dtype.
    """
    problems = spec_mismatches(reference, other)
    if problems:
        raise SpecMismatchError(
            f"MixDataSourcesNode requires every source to produce records with the same "
            f"element_spec (mixing under lax.switch needs matching output shapes across "
            f"all branches); source {index} differs from source 0:",
            problems,
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
        sources: Sequence[DataSourceModule],
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

        Raises:
            ValueError: If ``sources`` does not match ``config.num_sources``, or ``config.weights``
                is ``None``.
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
        # This is the constraint that lets get_records use lax.switch — every
        # branch must produce identically-shaped records.
        _validate_compatible_element_specs(sources)

        self._sources = nnx.List(list(sources))
        self._weights = tuple(weights)
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)
        self._iterator: Iterator[Any] | None = None

    def __len__(self) -> int:
        """Return total elements across all child sources, as long as they are now."""
        return sum(len(source) for source in self._sources)

    def _offsets(self) -> tuple[int, ...]:
        """Where each source's records start in the concatenation of the sources, now.

        A mixed record's index is its source's offset plus its index within that source;
        offsets follow the sources' current lengths, as the sampling does.
        """
        lengths = [len(source) for source in self._sources]
        return tuple(sum(lengths[:position]) for position in range(len(lengths)))

    def __repr__(self) -> str:
        """Config-identifying representation for checkpoint validation.

        Enumerates the child-source reprs and mixing weights so a restore can
        detect a change in the mixture composition or proportions.
        """
        child_reprs = ", ".join(repr(source) for source in self._sources)
        return (
            f"MixDataSourcesNode(sources=[{child_reprs}], "
            f"weights={list(self._weights)!r}, "
            f"length={len(self)})"
        )

    def __iter__(self) -> "MixDataSourcesNode":
        """Reset iterators and start a new epoch."""
        self.index.set_value(0)
        self.epoch.set_value(self.epoch.get_value() + 1)
        self._iterator = iter(self.to_grain_iter_dataset())
        return self

    def __next__(self) -> Any:
        """Sample a source by weight and yield the next element from it."""
        if self.index.get_value() >= len(self):
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

    def _selections(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Choose, for each output position, a source and a record within it.

        Args:
            start: Starting logical position (int or traced ``jax.Array``).
            size: Number of records.
            key: PRNG key for deterministic source / index selection.

        Returns:
            ``(chosen_sources, local_indices)``, each with leading dim ``size``.

        Raises:
            ValueError: If ``key is None``.
        """
        if key is None:
            raise ValueError(
                "MixDataSourcesNode.record_indices_at requires a PRNG key for "
                "deterministic mixing. Pass `key=jax.random.key(seed)` or "
                "drive iteration via Pipeline (which threads its own rngs)."
            )

        log_weights = jnp.log(jnp.asarray(self._weights, dtype=jnp.float32))
        source_lengths = jnp.asarray([len(s) for s in self._sources], dtype=jnp.int32)
        positions = jnp.asarray(start, dtype=jnp.int32) + jnp.arange(size, dtype=jnp.int32)

        def _select(position: jax.Array) -> tuple[jax.Array, jax.Array]:
            src_key, idx_key = jax.random.split(jax.random.fold_in(key, position))
            chosen_src = jax.random.categorical(src_key, log_weights)
            local_idx = jax.random.randint(idx_key, (), 0, source_lengths[chosen_src])
            return chosen_src, local_idx

        return jax.vmap(_select)(positions)

    def record_indices_at(  # noqa: DOC502
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Return the index of each record ``get_batch_at(start, size, key)`` returns.

        A mixed record's index is its source's offset in the concatenation of the sources
        plus its index within that source, so every record of every source has one index.

        Args:
            start: Starting logical position (int or traced ``jax.Array``).
            size: Number of records.
            key: PRNG key for deterministic source / index selection.

        Returns:
            Int32 ``jax.Array`` of shape ``(size,)``.

        Raises:
            ValueError: If ``key is None``.
        """
        chosen_sources, local_indices = self._selections(start, size, key)
        return jnp.asarray(self._offsets(), dtype=jnp.int32)[chosen_sources] + local_indices

    def get_records(self, indices: jax.Array) -> dict[str, jax.Array]:
        """Gather the mixed records at ``indices``, each from the source that owns it.

        A mixed index is a source's offset plus a record index within that source (see
        :meth:`record_indices_at`), so each record is fetched with its source's own
        ``get_records``. Stateless; ``vmap`` over records builds the batch in one trace.

        Args:
            indices: Int32 mixed record indices; concrete or traced.

        Returns:
            Dict mapping each data key to a JAX array with leading dim ``len(indices)``.
        """
        offsets = jnp.asarray(self._offsets(), dtype=jnp.int32)
        indices = jnp.asarray(indices, dtype=jnp.int32)
        owners = jnp.searchsorted(offsets, indices, side="right") - 1
        # Each branch fetches one record from one source. All branches share the same output
        # shape (validated at construction by _validate_compatible_element_specs).
        branches = [
            lambda local, src=src: jax.tree.map(lambda x: x[0], src.get_records(local[None]))
            for src in self._sources
        ]

        def _fetch_one(owner: jax.Array, index: jax.Array) -> dict[str, jax.Array]:
            return jax.lax.switch(owner, branches, index - offsets[owner])

        return jax.vmap(_fetch_one)(owners, indices)
