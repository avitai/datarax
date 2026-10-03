"""A source mixing indexed sources in fixed proportions: Grain's mix, in one 64-bit index space.

Grain's ``MapDataset.mix`` interleaves its parents deterministically: the weights become integer
proportions (the smallest scaled to 100, the rest scaled alike and truncated), and position ``k``
of the mix belongs to the parent whose share of the first ``k + 1`` positions first exceeds its
share of the first ``k``, at the parent position that share counts. Its length is the most
positions that serve each parent record at most once, the minimum over parents of
``len_c * S / p_c`` for proportions ``p`` summing to ``S``. :class:`MixDataSourcesNode` serves
exactly that selection and length. Grain evaluates the selection in Python per index; the mix
evaluates the same integer arithmetic in uint32 words, so it runs inside traced programs and on
the host alike, never as a table of Grain's period (which grows with the weight ratio).

Which record a child serves at its ``j``-th position is the child's own order:
``record_indices_at(j, ...)`` under the pipeline's epoch key folded with the child's position,
or the child's sequential order when the pipeline does not shuffle. A mixed record's index is its
child's offset, the sum of the lengths of the children before it, plus the record's index within
that child, so every record of every child has one 64-bit index.
"""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike

from datarax.config.registry import register_component
from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    record_words,
    RecordIdentity,
    refuse_padding,
)
from datarax.core.index_words import (
    add,
    divmod_word,
    from_words,
    greater,
    low_words,
    MAX_RECORDS,
    multiply_high,
    multiply_word,
    subtract,
    to_words,
    wrapped_positions,
)
from datarax.core.spec import spec_mismatches, SpecMismatchError
from datarax.sources.memory_source import MemorySource
from datarax.typing import DataDict


_MAX_PERIOD = (1 << 32) - 1
"""The largest sum of proportions the word arithmetic of the selection takes."""


def grain_proportions(weights: Sequence[float]) -> tuple[int, ...]:
    """Grain's integer proportions for ``weights``: the smallest scaled to 100, the rest alike.

    The rule of Grain's ``MapDataset.mix`` (``_float_to_int_proportions`` in
    ``grain/_src/python/dataset/transformations/mix.py``, grain 0.2.18): every weight is scaled by
    ``100 / min(weights)`` and truncated.

    Args:
        weights: Positive weights.

    Returns:
        One positive integer proportion per weight.
    """
    scale = 100 / min(weights)
    return tuple(int(weight * scale) for weight in weights)


def _period_counts[A: (jax.Array, np.ndarray)](within: A, proportions: Sequence[int]) -> list[A]:
    """How many of the first ``within`` positions of a period each child serves (Grain's loop).

    Grain's ``_dataset_and_key_of_next_element`` walks the children in order: of the first
    ``n`` positions held by children ``c`` onwards (whose proportions sum to ``R``), child ``c``
    holds ``n - floor(n * (R - p_c) / R)`` and the rest pass on to child ``c + 1``. Within a
    period ``n <= R < 2**32``, so the product is two words and the quotient one.

    Args:
        within: uint32 positions within the period, each at most the period.
        proportions: The children's proportions.

    Returns:
        One uint32 count per child, shaped like ``within``.
    """
    remaining = sum(proportions)
    current = within
    counts = []
    for proportion in proportions:
        rest = remaining - proportion
        if not rest:
            counts.append(current)
            break
        factor = np.uint32(rest)
        (_, following), _ = divmod_word(
            (multiply_high(current, factor), current * factor), remaining
        )
        counts.append(current - following)
        current, remaining = following, rest
    return counts


def _counts_before[A: (jax.Array, np.ndarray)](
    position: tuple[A, A], proportions: Sequence[int]
) -> list[tuple[A, A]]:
    """How many mix positions before ``position`` each child serves, in words.

    Grain's selection repeats every period ``S = sum(proportions)``, in which child ``c`` serves
    ``p_c`` positions, so before position ``q * S + r`` it serves ``q * p_c`` plus its count of
    the first ``r`` positions of a period.

    Args:
        position: The mix positions' words ``(hi, lo)``.
        proportions: The children's proportions.

    Returns:
        Each child's count, as words.
    """
    cycles, within = divmod_word(position, sum(proportions))
    return [
        add(multiply_word(cycles, np.uint32(proportion)), (within - within, count))
        for proportion, count in zip(proportions, _period_counts(within, proportions), strict=True)
    ]


def _selection[A: (jax.Array, np.ndarray)](
    position: tuple[A, A], proportions: Sequence[int]
) -> tuple[A, tuple[A, A]]:
    """The child Grain's mix selects at each position, and that child's position there.

    The child serving position ``k`` is the one whose count rises from ``k`` to ``k + 1``, and
    its position is its count before ``k`` (Grain's key into the parent).

    Args:
        position: The mix positions' words ``(hi, lo)``.
        proportions: The children's proportions.

    Returns:
        ``(child, child_position)``: the child as uint32, its position as words.
    """
    cycles, within = divmod_word(position, sum(proportions))
    before = _period_counts(within, proportions)
    after = _period_counts(within + np.uint32(1), proportions)
    zero = within - within
    child, proportion, count = zero, zero, zero
    for index, (earlier, later, share) in enumerate(zip(before, after, proportions, strict=True)):
        here = later - earlier  # 1 for the selected child, 0 for every other
        child = child + here * np.uint32(index)
        proportion = proportion + here * np.uint32(share)
        count = count + here * earlier
    return child, add(multiply_word(cycles, proportion), (zero, count))


@dataclass(frozen=True)
class MixDataSourcesConfig(StructuralConfig):
    """Configuration for :class:`MixDataSourcesNode`.

    Attributes:
        weights: One positive weight per child source, normalised to sum to 1.
    """

    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the weights.

        Raises:
            ValueError: If no weights are given or a weight is not positive: Grain mixes positive
                proportions only, and a zero weight would never serve its source.
        """
        if self.weights is None:
            raise ValueError("weights is required: one positive weight per source")
        weights = tuple(float(weight) for weight in self.weights)
        if not weights or not all(weight > 0 for weight in weights):
            raise ValueError(f"a mix takes one positive weight per source; got {self.weights!r}")
        total = sum(weights)
        object.__setattr__(self, "weights", tuple(weight / total for weight in weights))
        object.__setattr__(self, "stochastic", False)
        object.__setattr__(self, "stream_name", None)
        super().__post_init__()


@dataclass(frozen=True)
class _Layout:
    """Where the children stand now: their lengths, offsets, the index space and the epoch."""

    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    space: int
    epoch_length: int


def _child_spec(position: int, source: DataSourceModule) -> Any:
    """A child's record spec, refusing a child that cannot describe its records.

    Args:
        position: The child's position among the mix's sources.
        source: The child.

    Returns:
        The child's ``element_spec()``.

    Raises:
        ValueError: If the child's ``element_spec()`` is not implemented.
    """
    try:
        return source.element_spec()
    except NotImplementedError as error:
        raise ValueError(
            f"child {position} ({type(source).__name__}) cannot describe its records: its "
            f"element_spec() raised {error!r}; a mix reads each child's record spec"
        ) from error


def _refuse_unmixable(position: int, source: DataSourceModule) -> None:
    """Refuse a child whose record indices a mix cannot name.

    Args:
        position: The child's position among the mix's sources.
        source: The child.

    Raises:
        TypeError: If the child is not ``INDEXED``: a stream has no stable positions to mix.
        ValueError: If the child is one worker's shard of a ``MemorySource``, whose indices name
            positions of the whole data.
    """
    kind = source.record_identity
    if kind is not RecordIdentity.INDEXED:
        raise TypeError(
            f"child {position} ({type(source).__name__}) names its records as {kind.name}; a mix "
            "names each record by its stable position in its source, so every child is INDEXED"
        )
    if isinstance(source, MemorySource) and source.config.num_workers > 1:
        raise ValueError(
            f"child {position} (MemorySource) is one shard of num_workers="
            f"{source.config.num_workers} workers, whose record indices name positions of the "
            "whole data; mix unsharded sources"
        )


def _refuse_different_records(specs: Sequence[Any]) -> None:
    """Refuse children whose records differ: the traced gather switches over equal records.

    Args:
        specs: Each child's record spec.

    Raises:
        SpecMismatchError: Naming every field where a child's records differ from child 0's.
    """
    for position, spec in enumerate(specs[1:], start=1):
        problems = spec_mismatches(specs[0], spec)
        if problems:
            raise SpecMismatchError(
                f"MixDataSourcesNode requires every source to produce records with the same "
                f"element_spec (its traced gather switches between children whose records "
                f"match); source {position} differs from source 0:",
                problems,
            )


@register_component("source", "MixDataSources")
class MixDataSourcesNode(DataSourceModule):
    """Mix indexed sources in fixed proportions, as Grain's ``MapDataset.mix`` does.

    An epoch is Grain's length, the most positions that serve each child record at most once,
    so every record of an epoch is distinct and no two rows share a key. A child that serves
    fewer of its records per epoch than it holds serves a different part of them each epoch when
    the pipeline shuffles (each child is then ordered by the epoch's key), and the same part when
    it does not. To weight a larger source up, give it a larger weight.

    The mix holds its children and its static proportions, and nothing else: no counter, no key
    and no Variable of its own.
    """

    @property
    def record_identity(self) -> RecordIdentity:
        """A mixed record's index is its source's offset plus its index within that source."""
        return RecordIdentity.INDEXED

    def __init__(
        self,
        config: MixDataSourcesConfig,
        sources: Sequence[DataSourceModule],
        *,
        name: str | None = None,
    ) -> None:
        """Mix ``sources`` with ``config.weights``.

        Args:
            config: The mixing weights, one per source.
            sources: The indexed sources to mix, in the order of the weights.
            name: Optional module name.

        Raises:
            ValueError: If the weights and sources differ in number, Grain's proportions for
                the weights pass one word, or a source is refused (see the checks it calls).
        """
        super().__init__(config, name=name or "MixDataSourcesNode")
        weights = config.weights
        if weights is None or len(sources) != len(weights):
            raise ValueError(f"{len(weights or ())} weights for {len(sources)} sources")
        for position, source in enumerate(sources):
            _refuse_unmixable(position, source)
        proportions = grain_proportions(weights)
        if sum(proportions) > _MAX_PERIOD:
            raise ValueError(
                f"Grain's proportions for these weights, {proportions}, sum past one word "
                f"(2**32 - 1); weights this far apart are not mixed"
            )
        self._sources = nnx.List(list(sources))
        self._weights = tuple(weights)
        self._proportions = proportions
        self._layout()
        _refuse_different_records(
            [_child_spec(position, source) for position, source in enumerate(sources)]
        )

    @property
    def sources(self) -> tuple[DataSourceModule, ...]:
        """The mixed sources, in the order of their weights."""
        return tuple(self._sources)

    @property
    def weights(self) -> tuple[float, ...]:
        """The mixing weights, normalised to sum to 1, in the order of the sources."""
        return self._weights

    def _layout(self) -> _Layout:
        """The children's current lengths and offsets, the index space and the epoch length.

        Returns:
            The layout.

        Raises:
            ValueError: If a child holds no records, the index space reaches the padding index,
                or the epoch is longer than ``len()`` can report.
        """
        lengths = tuple(len(source) for source in self._sources)
        for position, length in enumerate(lengths):
            if length < 1:
                raise ValueError(f"child {position} has no records; a mix serves every child")
        offsets = tuple(sum(lengths[:position]) for position in range(len(lengths)))
        space = sum(lengths)
        if space >= MAX_RECORDS:
            raise ValueError(
                f"the children hold {space} records, whose indices reach the padding index "
                f"(2**64 - 1); a mix indexes at most {MAX_RECORDS - 1}"
            )
        period = sum(self._proportions)
        epoch_length = min(
            length * period // proportion
            for length, proportion in zip(lengths, self._proportions, strict=True)
        )
        if epoch_length > sys.maxsize:
            raise ValueError(
                f"an epoch of this mix has {epoch_length} positions, past the sys.maxsize "
                "records len() reports"
            )
        return _Layout(lengths, offsets, space, epoch_length)

    def __len__(self) -> int:
        """Positions in an epoch: Grain's length over the children as long as they are now."""
        return self._layout().epoch_length

    def element_spec(self) -> Any:
        """One mixed record's spec: its children's, which the constructor requires equal."""
        return _child_spec(0, self._sources[0])

    def __repr__(self) -> str:
        """Config-identifying representation for checkpoint validation.

        Enumerates the child-source reprs, the mixing weights and the epoch length so a restore
        can detect a change in the mixture's composition, proportions or size.
        """
        child_reprs = ", ".join(repr(source) for source in self._sources)
        return (
            f"MixDataSourcesNode(sources=[{child_reprs}], "
            f"weights={list(self._weights)!r}, "
            f"length={len(self)})"
        )

    def _window_starts(
        self, start: int | jax.Array, first: jax.Array, epoch_length: int
    ) -> tuple[list[int] | list[jax.Array], jax.Array]:
        """Where each child's positions start at mix position ``start``.

        Args:
            start: The first mix position, a Python int or a traced int32.
            first: ``start`` wrapped at the epoch's length, as uint32 ``(1, 2)`` words.
            epoch_length: The epoch's length, where positions wrap.

        Returns:
            Each child's first position as its ``record_indices_at`` takes it (a Python int,
            or an int32 below the traced start), and all of them as uint32 ``(n, 2)`` words.
        """
        if isinstance(start, int | np.integer):
            exact = to_words(int(start) % epoch_length)
            counts = _counts_before((exact[0:1], exact[1:2]), self._proportions)
            starts = [int(from_words(np.stack(count, axis=-1))[0]) for count in counts]
            return starts, jnp.asarray(to_words(starts))
        counts = _counts_before((first[:, 0], first[:, 1]), self._proportions)
        words = jnp.stack([jnp.concatenate(count) for count in counts])
        # A traced start is an int32 position, and a child's count before it is at most it.
        return [low.astype(jnp.int32)[0] for _, low in counts], words

    def record_indices_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Name the records at mix positions ``start .. start + size`` of an epoch.

        Grain's selection gives each position's child and the child's position there. A child's
        positions in a run of the mix are consecutive, so each child names its whole window of
        ``size`` positions with its own ``record_indices_at``, keyed ``fold_in(key, c)`` when a
        key is given and sequential otherwise, and each row takes its child's name for its
        position. The program is the same for every start. Positions wrap at the epoch's length;
        a run that wraps names its rows past the end from inside their children's windows, which
        the pipeline replaces with the next epoch's names.

        Args:
            start: Starting position; a Python int of any size or a traced int32 ``jax.Array``.
            size: Number of records (Python int).
            key: The pipeline's epoch key, or ``None`` for the children's sequential orders.

        Returns:
            uint32 array of shape ``(size, 2)``, each index as its words ``(hi, lo)``.
        """
        layout = self._layout()
        positions = wrapped_positions(start, size, layout.epoch_length)
        child, child_position = _selection((positions[:, 0], positions[:, 1]), self._proportions)
        starts, first = self._window_starts(start, positions[:1], layout.epoch_length)
        names = jnp.stack(
            [
                jnp.asarray(
                    source.record_indices_at(
                        child_start, size, None if key is None else jax.random.fold_in(key, c)
                    ),
                    jnp.uint32,
                )
                for c, (source, child_start) in enumerate(zip(self._sources, starts, strict=True))
            ]
        )
        owner = child.astype(jnp.int32)
        high, low = subtract(child_position, (first[owner, 0], first[owner, 1]))
        row = jnp.where((high == 0) & (low < size), low, 0).astype(jnp.int32)
        named = names[owner, row]
        offsets = jnp.asarray(to_words(layout.offsets))
        high, low = add((named[:, 0], named[:, 1]), (offsets[owner, 0], offsets[owner, 1]))
        return jnp.stack([high, low], axis=-1)

    def _owners(self, words: np.ndarray, layout: _Layout) -> tuple[np.ndarray, np.ndarray]:
        """Each mixed index's child and its index within that child, on the host.

        Args:
            words: uint32 ``(n, 2)`` mixed record indices.
            layout: The children's layout.

        Returns:
            The children as ``(n,)`` integers and the local indices as uint32 ``(n, 2)`` words.

        Raises:
            IndexError: If an index is the padding index or outside the mix's index space.
        """
        refuse_padding(words)
        index = (words[:, 0], words[:, 1])
        space = to_words(layout.space)
        outside = ~greater((space[0:1], space[1:2]), index)
        if outside.any():
            raise IndexError(
                f"record index {int(from_words(words[outside][:1])[0])} is outside "
                f"[0, {layout.space})"
            )
        offsets = to_words(layout.offsets)
        owners = np.zeros(len(words), np.int64)
        for offset in offsets[1:]:
            owners += ~greater((offset[0:1], offset[1:2]), index)
        high, low = subtract(index, (offsets[owners, 0], offsets[owners, 1]))
        return owners, np.stack([high, low], axis=-1)

    def provenance(  # noqa: DOC502 - record_words and _owners raise
        self, indices: ArrayLike
    ) -> tuple[Mapping[str, Any], ...]:
        """The provenance of the mixed records ``indices`` names, each from the source owning it.

        A mixed index is a source's offset plus the record's index within that source, so each
        record's provenance is its source's ``provenance`` of that index.

        Args:
            indices: uint32 ``(n, 2)`` mixed record indices.

        Returns:
            One mapping per index, in the order named.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index is the padding index or outside the mix's index space.
        """
        owners, local = self._owners(record_words(indices), self._layout())
        found: list[Mapping[str, Any]] = [{}] * len(owners)
        for owner in np.unique(owners):
            at = np.flatnonzero(owners == owner)
            for position, record in zip(
                at, self._sources[int(owner)].provenance(local[at]), strict=True
            ):
                found[position] = record
        return tuple(found)

    def get_records(self, indices: jax.Array) -> DataDict:
        """Gather the mixed records at ``indices``, each from the source that owns it.

        A mixed index is a source's offset plus a record index within that source (see
        :meth:`record_indices_at`), so each record is fetched with its source's own
        ``get_records``. Stateless; ``vmap`` over records builds the batch in one trace. The
        gather addresses records with one int32 word, so a mix whose children hold more than
        ``2**31 - 1`` records together is refused here.

        Args:
            indices: uint32 ``(n, 2)`` mixed record indices; concrete or traced.

        Returns:
            Dict mapping each data key to a JAX array with leading dim ``len(indices)``.
        """
        layout = self._layout()
        offsets = jnp.asarray(layout.offsets, dtype=jnp.int32)
        indices = low_words(indices, layout.space).astype(jnp.int32)
        owners = jnp.searchsorted(offsets, indices, side="right") - 1
        # Each branch fetches one record from one source. All branches share the same output
        # shape (the constructor refuses children whose records differ).
        branches = [
            lambda local, src=src: jax.tree.map(
                lambda x: x[0], src.get_records(to_words(local[None]))
            )
            for src in self._sources
        ]

        def _fetch_one(owner: jax.Array, index: jax.Array) -> dict[str, jax.Array]:
            return jax.lax.switch(owner, branches, index - offsets[owner])

        return jax.vmap(_fetch_one)(owners, indices)
