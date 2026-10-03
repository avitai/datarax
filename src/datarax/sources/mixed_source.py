"""A source mixing indexed sources in fixed proportions: Grain's mix, in one 64-bit index space.

Grain's ``MapDataset.mix`` interleaves its parents deterministically: the weights become integer
proportions (the smallest scaled to 100, the rest scaled alike and truncated), and position ``k``
of the mix belongs to the parent whose share of the first ``k + 1`` positions first exceeds its
share of the first ``k``, at the parent position that share counts. Its length is the most
positions that serve each parent record at most once, the minimum over parents of
``len_c * S / p_c`` for proportions ``p`` summing to ``S``. :class:`MixDataSourcesNode` serves
that selection, and that length computed exactly in integers, which can exceed Grain's float64
result by one. Grain evaluates the selection in Python per index; the mix
evaluates the same integer arithmetic in uint32 words, so it runs inside traced programs and on
the host alike, never as a table of Grain's period (which grows with the weight ratio).

Which record a child serves at its ``j``-th position is the child's own order:
``record_indices_at(j, ...)`` under the pipeline's epoch key folded with the child's position,
or the child's sequential order when the pipeline does not shuffle. A mixed record's index is its
child's offset, the sum of the index spaces of the children before it, plus the record's index
within that child, so every record of every child has one 64-bit index. A mixed record carries the
union of its children's fields, a field some child lacks being a ``Maybe``.
"""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike

from datarax.config.registry import register_component
from datarax.core import batch_ops
from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    record_words,
    RecordIdentity,
    refuse_padding,
)
from datarax.core.element_batch import Batch
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
from datarax.core.maybe import Maybe
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
        weights: One positive weight per child source, as given. Grain's proportions are
            computed from these, as ``grain.MapDataset.mix`` computes them from the weights it
            is given.
    """

    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the weights.

        Raises:
            ValueError: If no weights are given or a weight is not positive: Grain mixes positive
                proportions only, and a zero weight would never serve its source.
        """
        if self.weights is None:
            raise ValueError("weights is required: one positive weight per source")
        weights = tuple(float(weight) for weight in self.weights)
        if not weights or not all(weight > 0 for weight in weights):
            raise ValueError(f"a mix takes one positive weight per source; got {self.weights!r}")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "stochastic", False)
        object.__setattr__(self, "stream_name", None)
        super().__post_init__()

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        """The weights divided by their sum: each source's share of an epoch's positions."""
        weights = self.weights or ()
        total = sum(weights)
        return tuple(weight / total for weight in weights)


@dataclass(frozen=True)
class _Layout:
    """Where the children stand now: their offsets, the index space and the epoch's length.

    A child's offset counts the index spaces of the children before it; the epoch's length
    counts the positions each child can serve once (its ``len``). The two differ for a mix child
    that does not cover its own children.
    """

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
        TypeError: If the child is not ``INDEXED`` (a stream has no stable positions to mix) or
            has no host read.
        ValueError: If the child is one worker's shard of a ``MemorySource``, whose indices name
            positions of the whole data.
    """
    kind = source.record_identity
    if kind is not RecordIdentity.INDEXED:
        raise TypeError(
            f"child {position} ({type(source).__name__}) names its records as {kind.name}; a mix "
            "names each record by its stable position in its source, so every child is INDEXED"
        )
    if not callable(getattr(source, "get_batch", None)):
        raise TypeError(
            f"child {position} ({type(source).__name__}) has no host read, "
            "get_batch(indices, *, epochs, contiguous), which a mix reads its records with"
        )
    if isinstance(source, MemorySource) and source.config.num_workers > 1:
        raise ValueError(
            f"child {position} (MemorySource) is one shard of num_workers="
            f"{source.config.num_workers} workers, whose record indices name positions of the "
            "whole data; mix unsharded sources"
        )


type _Path = tuple[Any, ...]
"""A field's key path in a record, as ``jax.tree_util`` gives it."""


def _present_spec() -> jax.ShapeDtypeStruct:
    """The spec of one record's ``present`` flag, built when asked (no JAX value at import)."""
    return jax.ShapeDtypeStruct((), jnp.bool_)


def _is_field(node: Any) -> bool:
    """A ``Maybe`` is one field, not two leaves."""
    return isinstance(node, Maybe)


def _fields_of(tree: Any) -> dict[_Path, Any]:
    """Each field of a record's spec or a batch's data by its path, a ``Maybe`` as one field."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_field)
    return dict(flat)


def _value_part(field: Any) -> Any:
    """A field's values: a ``Maybe``'s value, or the field itself."""
    return field.value if isinstance(field, Maybe) else field


def _nested(fields: Mapping[_Path, Any]) -> dict[str, Any]:
    """Rebuild a record's nested dictionaries from its fields' paths.

    Args:
        fields: Each field by its path.

    Returns:
        The record as nested dictionaries.

    Raises:
        TypeError: If a path steps through anything but a dictionary key.
    """
    record: dict[str, Any] = {}
    for path, field in fields.items():
        if not all(isinstance(step, jax.tree_util.DictKey) for step in path):
            raise TypeError(
                f"a mix joins records whose fields are nested dictionaries; field "
                f"{jax.tree_util.keystr(path)} is not"
            )
        node = record
        for step in path[:-1]:
            node = node.setdefault(step.key, {})
        node[path[-1].key] = field
    return record


def _refuse_nesting_conflicts(held: Mapping[_Path, list[tuple[int, Any]]]) -> None:
    """Refuse a field that holds values in one child and nested fields in another.

    Args:
        held: Each field path and the children holding it.

    Raises:
        SpecMismatchError: Naming the field and the nested one.
    """
    for path, holders in held.items():
        for other, nested in held.items():
            if other != path and other[: len(path)] == path:
                raise SpecMismatchError(
                    f"a mix's children disagree on field {jax.tree_util.keystr(path)}:",
                    [
                        f"child {holders[0][0]} holds values at {jax.tree_util.keystr(path)}, "
                        f"child {nested[0][0]} fields under it ({jax.tree_util.keystr(other)})"
                    ],
                )


def _union_field(path: _Path, holders: list[tuple[int, Any]], children: int) -> Any:
    """One field of the union: as its holders declare it, or a ``Maybe`` if any child lacks it.

    Args:
        path: The field's path.
        holders: The children holding the field, with their spec of it.
        children: How many children the mix has.

    Returns:
        The field's spec, or a ``Maybe`` of it.

    Raises:
        SpecMismatchError: If two holders declare different shapes or device dtypes.
    """
    first_child, first = holders[0][0], _value_part(holders[0][1])
    for child, field in holders[1:]:
        value = _value_part(field)
        problems = spec_mismatches(first, value)
        if problems:
            raise SpecMismatchError(
                f"a mix's children disagree on field {jax.tree_util.keystr(path)}: child "
                f"{first_child} holds {first.shape} {first.dtype}, child {child} "
                f"{value.shape} {value.dtype}:",
                problems,
            )
    optional = len(holders) < children or any(isinstance(f, Maybe) for _, f in holders)
    return Maybe(first, _present_spec()) if optional else first


def union_spec(specs: Sequence[Any]) -> Any:  # noqa: DOC502 - the checks it calls raise
    """The union of the children's record specs: a field some child lacks is a ``Maybe``.

    Fields are compared by their declared (device) specs. A field every child has stays as it
    is; a field some child lacks, or that some child holds as a ``Maybe``, is
    ``Maybe(value_spec, present_spec)``.

    Args:
        specs: Each child's record spec, a nested dictionary.

    Returns:
        The union, a nested dictionary of specs.

    Raises:
        SpecMismatchError: Naming a field whose shape or device dtype differs between two
            children, or which holds values in one child and fields in another.
    """
    held: dict[_Path, list[tuple[int, Any]]] = {}
    for child, spec in enumerate(specs):
        for path, field in _fields_of(spec).items():
            held.setdefault(path, []).append((child, field))
    _refuse_nesting_conflicts(held)
    return _nested(
        {path: _union_field(path, holders, len(specs)) for path, holders in held.items()}
    )


type _ReadField = tuple[tuple[int, ...], bool]
"""A union field as the children's reads give it: one record's shape, and whether it can be
missing (some child lacks it or holds it as a ``Maybe``)."""


def _read_union(reads: Sequence[Mapping[_Path, Any]]) -> dict[_Path, _ReadField]:
    """The union's fields from the children's reads (each child read once, empty or not).

    The construction checked the children's specs against each other (:func:`union_spec`), so
    the reads agree on each field's record shape; this reads the union off them instead of
    rebuilding every child's spec per batch.

    Args:
        reads: Each child's rows by field path.

    Returns:
        Each field's record shape and whether it can be missing.
    """
    union: dict[_Path, _ReadField] = {}
    for held in reads:
        for path, field in held.items():
            shape = tuple(np.shape(_value_part(field))[1:])
            optional = isinstance(field, Maybe) or union.get(path, (shape, False))[1]
            union[path] = (shape, optional)
    return {
        path: (shape, optional or any(path not in held for held in reads))
        for path, (shape, optional) in union.items()
    }


def _union_rows(
    held: Mapping[_Path, Any],
    union: Mapping[_Path, _ReadField],
    dtypes: Mapping[_Path, np.dtype],
    size: int,
) -> dict[str, Any]:
    """One child's host rows in the union's fields, each at the field's joined host dtype.

    Args:
        held: The child's rows by field path, as its ``get_batch`` returns them.
        union: The union's fields by path (:func:`_read_union`).
        dtypes: Each field's host dtype, joined over the children.
        size: The rows read.

    Returns:
        The rows as a nested dictionary of the union's fields.
    """
    rows: dict[_Path, Any] = {}
    for path, (shape, optional) in union.items():
        dtype = dtypes[path]
        field = held.get(path)
        if not optional:
            rows[path] = np.asarray(field).astype(dtype, copy=False)
        elif field is None:
            rows[path] = Maybe(np.zeros((size, *shape), dtype), np.zeros(size, np.bool_))
        elif isinstance(field, Maybe):
            rows[path] = Maybe(np.asarray(field.value).astype(dtype, copy=False), field.present)
        else:
            rows[path] = Maybe(np.asarray(field).astype(dtype, copy=False), np.ones(size, np.bool_))
    return _nested(rows)


def _host_dtypes(
    reads: Sequence[Mapping[_Path, Any]], union: Mapping[_Path, _ReadField]
) -> dict[_Path, np.dtype]:
    """Each union field's host dtype: NumPy's promotion of the dtypes the children store.

    Args:
        reads: Each child's read, its fields by path.
        union: The union's fields by path.

    Returns:
        The joined dtype of each field.
    """
    return {
        path: np.result_type(
            *(np.asarray(_value_part(held[path])).dtype for held in reads if path in held)
        )
        for path in union
    }


def _joined(reads: Sequence[tuple[np.ndarray, Mapping[_Path, Any]]]) -> Batch:
    """Join the children's rows in the union's fields, in the order they were named.

    Every field is held at its children's joined host dtype (NumPy's promotion of the dtypes
    they store, empty reads included), so the result does not depend on which children its rows
    come from. A single child's rows already in order are returned without a copy.

    Args:
        reads: Per child, the positions of its rows in the read and its fields by path.

    Returns:
        The rows as a host ``Batch`` named by position.
    """
    union = _read_union([held for _, held in reads])
    dtypes = _host_dtypes([held for _, held in reads], union)
    parts = [
        (rows, batch_ops.from_arrays(_union_rows(held, union, dtypes, len(rows))))
        for rows, held in reads
        if len(rows)
    ] or [(reads[0][0], batch_ops.from_arrays(_union_rows(reads[0][1], union, dtypes, 0)))]
    order = np.concatenate([rows for rows, _ in parts])
    joined = parts[0][1] if len(parts) == 1 else batch_ops.concatenate([b for _, b in parts])
    if not np.array_equal(order, np.arange(len(order))):
        joined = batch_ops.take(joined, np.argsort(order, kind="stable"))
    return joined


class _IndexedHostRead(Protocol):
    """The host read of an indexed source (``EagerSource``, ``StreamingDiskSource``, a mix)."""

    def get_batch(
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the records ``indices`` names, as a ``Batch`` named with them, on the host."""
        ...


@register_component("source", "MixDataSources")
class MixDataSourcesNode(DataSourceModule):
    """Mix indexed sources in fixed proportions, as Grain's ``MapDataset.mix`` does.

    An epoch is Grain's length, the most positions that serve each child record at most once,
    so every record of an epoch is distinct and no two rows share a key. A child that serves
    fewer of its records per epoch than it holds serves a different part of them each epoch when
    the pipeline shuffles (each child is then ordered by the epoch's key), and the same part when
    it does not. To weight a larger source up, give it a larger weight.

    A mixed record carries the union of its children's fields: a field some child lacks is a
    ``Maybe`` in every batch of the mix (:meth:`element_spec`). :meth:`get_batch` reads mixed
    records on the host. The traced :meth:`get_records` gathers only from children whose fields
    are equal.

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
        self.element_spec()

    @property
    def sources(self) -> tuple[DataSourceModule, ...]:
        """The mixed sources, in the order of their weights."""
        return tuple(self._sources)

    @property
    def weights(self) -> tuple[float, ...]:
        """The mixing weights as given, in the order of the sources."""
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
        spaces = tuple(source.index_space() for source in self._sources)
        offsets = tuple(sum(spaces[:position]) for position in range(len(spaces)))
        space = sum(spaces)
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
        return _Layout(offsets, space, epoch_length)

    def __len__(self) -> int:
        """Positions in an epoch: Grain's length over the children as long as they are now."""
        return self._layout().epoch_length

    def index_space(self) -> int:
        """Every record of every child has an index: the sum of the children's index spaces."""
        return self._layout().space

    def _child_specs(self) -> list[Any]:
        """Each child's record spec, as it declares it now."""
        return [_child_spec(position, source) for position, source in enumerate(self._sources)]

    def element_spec(self) -> Any:  # noqa: DOC502 - union_spec and _child_spec raise
        """One mixed record's spec: the union of its children's fields (:func:`union_spec`).

        A field every child has is as the children declare it; a field some child lacks, or holds
        as a ``Maybe``, is a ``Maybe``.

        Returns:
            The union spec, a nested dictionary of ``jax.ShapeDtypeStruct`` and ``Maybe``.

        Raises:
            SpecMismatchError: If a field's shape, device dtype or nesting differs between
                children.
            ValueError: If a child cannot describe its records.
        """
        return union_spec(self._child_specs())

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

    def get_batch(  # noqa: DOC503 - record_words, _owners and the children's reads raise
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the mixed records ``indices`` names, as a ``Batch`` named with them, on the host.

        Each child is read once, with its own ``get_batch``, for the rows it owns (none for some),
        and its rows take the union's fields (:meth:`element_spec`): a field the child lacks is a
        ``Maybe`` of zeros with ``present`` False. Every field is held at its children's joined
        host dtype (NumPy's promotion of the dtypes they store), so the batch's structure and
        dtypes do not depend on which children its rows come from. The rows are returned in the
        order named, with the given words as ``indices``, the given epochs and draws 0. No
        device array is created and no state changes. With ``contiguous=True`` the caller states
        that ``indices`` is a run of consecutive mixed records; each child then reads its part
        of the run as views of its columns.

        Args:
            indices: uint32 ``(n, 2)`` mixed record indices, each as its words ``(hi, lo)``.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive mixed records.

        Returns:
            The records as a host ``Batch``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, or a run declared
                contiguous is not one.
            IndexError: If an index is the padding index or outside the mix's index space.
        """
        words = record_words(indices)
        owners, local = self._owners(words, self._layout())
        if contiguous and len(words) > 1 and (np.diff(from_words(words)) != 1).any():
            raise ValueError(
                "a read declared contiguous names a run of consecutive mixed records; these "
                "indices are not one"
            )
        epoch_of_each = np.ascontiguousarray(
            np.broadcast_to(np.asarray(epochs, np.int32), (len(words),))
        )
        reads = [
            (rows, _fields_of(read.data))
            for rows, read in self._read_children(owners, local, epoch_of_each, contiguous)
        ]
        joined = _joined(reads)
        return joined.replace(
            indices=words, epochs=epoch_of_each, draws=np.zeros(len(words), np.int32)
        )

    def _read_children(
        self, owners: np.ndarray, local: np.ndarray, epochs: np.ndarray, contiguous: bool
    ) -> list[tuple[np.ndarray, Batch]]:
        """Read each child once, for the rows it owns (none for some), with its host read.

        Args:
            owners: Each row's child.
            local: Each row's index within its child, uint32 ``(n, 2)`` words.
            epochs: Each row's epoch.
            contiguous: Whether the rows are a run of consecutive mixed records.

        Returns:
            Per child, the rows it owns and its ``Batch`` of them.
        """
        reads = []
        for child, source in enumerate(self._sources):
            rows = np.flatnonzero(owners == child)
            read = cast(_IndexedHostRead, source).get_batch(
                local[rows], epochs=epochs[rows], contiguous=contiguous
            )
            reads.append((rows, read))
        return reads

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
        gather addresses records with one int32 word, so a mix whose index space passes
        ``2**31 - 1`` is refused here, naming the host read.

        Args:
            indices: uint32 ``(n, 2)`` mixed record indices; concrete or traced.

        Returns:
            Dict mapping each data key to a JAX array with leading dim ``len(indices)``.

        Raises:
            ValueError: If the mix's index space passes ``2**31 - 1``, which one int32 word
                cannot address.
            TypeError: If the children's fields differ, which only the host read joins.
        """
        layout = self._layout()
        if layout.space > np.iinfo(np.int32).max:
            raise ValueError(
                f"MixDataSourcesNode.get_records addresses mixed records with one int32 word, "
                f"and this mix names {layout.space}; read its records on the host with "
                "mix.get_batch(indices)"
            )
        union = self.element_spec()
        for position, spec in enumerate(self._child_specs()):
            if spec_mismatches(union, spec):
                raise TypeError(
                    "MixDataSourcesNode.get_records gathers records whose fields every child "
                    f"holds alike; child {position} lacks fields of the mix's union, so read the "
                    "mix's records on the host with mix.get_batch(indices), which joins them as "
                    "the union of the children's fields"
                )
        offsets = jnp.asarray(layout.offsets, dtype=jnp.int32)
        indices = low_words(indices, layout.space).astype(jnp.int32)
        owners = jnp.searchsorted(offsets, indices, side="right") - 1
        # Each branch fetches one record from one source. All branches share the same output
        # shape (children whose records differ are refused above).
        branches = [
            lambda local, src=src: jax.tree.map(
                lambda x: x[0], src.get_records(to_words(local[None]))
            )
            for src in self._sources
        ]

        def _fetch_one(owner: jax.Array, index: jax.Array) -> dict[str, jax.Array]:
            return jax.lax.switch(owner, branches, index - offsets[owner])

        return jax.vmap(_fetch_one)(owners, indices)
