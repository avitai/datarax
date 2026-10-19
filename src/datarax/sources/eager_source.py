"""In-memory sources: records held on the host and read by one host gather.

An ``EagerSource`` holds its records in two parts. The array part is a pytree of host NumPy
columns, one row per record: the only thing the host read, ``get_records`` and ``element_spec``
read. The non-array part (strings, bytes and other Python objects) is each record's provenance, an
immutable mapping per record kept in a host holder that NNX leaves out of module state and out of
every trace; it is never part of a ``Batch``. ``MemorySource``, ``TFDSEagerSource`` and
``HFEagerSource`` build on it, each turning its own input into the two parts; records become
them through one path, :func:`parts_of_records`.

The host read, ``get_batch(indices, *, epochs=0)``, gathers the named records with NumPy indexing
(a contiguous run as views) and returns a ``Batch`` named with the given indices and epochs. It
reads and changes no state and allocates no device array. The traced read, ``get_records``,
serves the compiled pipeline path.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.core import Tracer
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.core import batch_ops
from datarax.core.block_order import BlockOrder
from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    host_rows,
    NO_PROVENANCE,
    record_words,
    RecordIdentity,
)
from datarax.core.element_batch import Batch, Element
from datarax.core.host_values import HostColumns, HostProvenance, HostRagged
from datarax.core.index_words import low_words, to_words
from datarax.core.ragged import (
    column_lengths,
    is_ragged,
    ragged_column,
    ragged_lengths,
    RaggedColumn,
    take_column,
    take_ragged,
    varies_along_first_axis_only,
)
from datarax.core.spec import array_to_spec_strip_leading, device_spec, JAX_ARRAY_KINDS
from datarax.sources._field_paths import field_name, field_names
from datarax.sources._index_validation import validate_index_batch


def is_array_leaf(value: Any) -> bool:
    """Whether a record's value is numeric, so it belongs to a column rather than provenance.

    Python and NumPy numbers and numeric arrays (host or device) are; strings, bytes and any
    other object are not. A list or tuple is numeric when NumPy reads it as a numeric array.

    Args:
        value: The value.

    Returns:
        Whether the value is numeric.
    """
    if isinstance(value, str | bytes):
        return False
    if isinstance(value, bool | int | float | complex | np.generic | np.ndarray | jax.Array):
        try:
            return np.dtype(getattr(value, "dtype", type(value))).kind in JAX_ARRAY_KINDS
        except TypeError:  # an extended dtype, such as a PRNG key's, is not a column
            return False
    if isinstance(value, list | tuple):
        try:
            return np.asarray(value).dtype.kind in JAX_ARRAY_KINDS
        except ValueError:  # ragged nesting is not one array
            return False
    return False


_ABSENT = object()


def split_record(record: Any) -> tuple[PyTree | None, dict[str, Any]]:
    """Split one record into its array part and its provenance.

    A record is a mapping (nested mappings included) of values, or a single value. Numeric values
    (:func:`is_array_leaf`) become NumPy arrays in the array part, with the record's structure;
    every other value goes to the provenance, keyed by its path (``"metadata/id"``).

    Args:
        record: The record.

    Returns:
        The array part, or ``None`` when the record has no numeric value, and the provenance.
    """
    provenance: dict[str, Any] = {}
    part = _split(record, (), provenance)
    return (None if part is _ABSENT else part), provenance


def _split(node: Any, path: tuple[str, ...], provenance: dict[str, Any]) -> Any:
    if isinstance(node, Mapping):
        part = {}
        for key, child in node.items():
            value = _split(child, (*path, str(key)), provenance)
            if value is not _ABSENT:
                part[key] = value
        return part if part else _ABSENT
    if is_array_leaf(node):
        return np.asarray(node)
    provenance[field_name(path)] = node
    return _ABSENT


RAGGED_SOURCES = "MemorySource, HFEagerSource"
"""The sources that hold ragged fields, which packing reads records from."""


def past_first_axis_refusal(field: str, detail: str) -> ValueError:
    """The refusal of a field whose records differ in more than their first axis.

    Packing joins records along their first axis only, so such a field is neither a column nor a
    ragged field, and only padding fixes it.

    Args:
        field: The field's name.
        detail: How its records differ.

    Returns:
        The error, naming the field and the fix.
    """
    return ValueError(
        f"field {field!r} {detail}: they differ in more than their first axis, which packing "
        "joins records along, and a batch's shapes are static: pad the field to one shape and "
        "keep its mask or lengths in data"
    )


def shape_refusal(
    field: str, first: tuple[int, ...], shape: tuple[int, ...], position: int
) -> ValueError:
    """The refusal of a field whose shape differs between record 0 and record ``position``.

    A field differing along its first axis only could be packed from a source holding ragged
    fields; one differing past it can only be padded (:func:`past_first_axis_refusal`).

    Args:
        field: The field's name.
        first: Its shape in record 0.
        shape: Its shape in record ``position``.
        position: The record.

    Returns:
        The error, naming the field, both shapes and the fix.
    """
    detail = f"is {first} in record 0 and {shape} in record {position}"
    if not varies_along_first_axis_only([first, shape]):
        return past_first_axis_refusal(field, detail)
    return ValueError(
        f"field {field!r} {detail}, but a batch's shapes are static: pad the field to a fixed "
        "length along its first axis and keep its mask or length in data, or pack records with "
        f"segment ids from a source that holds ragged fields ({RAGGED_SOURCES})"
    )


def stack_records(parts: Sequence[PyTree]) -> PyTree:  # noqa: DOC503 - shape_refusal builds one
    """Stack records' array parts into columns, once, on the host.

    Every record must hold the same numeric fields with the same shapes: a batch's shapes are
    static, so a field whose shape varies is refused, naming the field, both shapes and the fix
    (:func:`shape_refusal`).

    Args:
        parts: One array part per record (NumPy leaves), as :func:`split_record` returns them.

    Returns:
        The columns: the records' structure with a leading record axis on every leaf.

    Raises:
        ValueError: If ``parts`` is empty, two records hold different numeric fields, or a field's
            shape differs between records.
    """
    if not parts:
        raise ValueError("stack_records needs at least one record")
    structure = jax.tree.structure(parts[0])
    shapes = [np.shape(leaf) for leaf in jax.tree.leaves(parts[0])]
    for position, part in enumerate(parts):
        if jax.tree.structure(part) != structure:
            raise ValueError(
                f"record {position} holds the numeric fields {field_names(part)} where record 0 "
                f"holds {field_names(parts[0])}; every record holds the same fields"
            )
        for field, first, shape in zip(
            field_names(part),
            shapes,
            (np.shape(leaf) for leaf in jax.tree.leaves(part)),
            strict=True,
        ):
            if shape != first:
                raise shape_refusal(field, first, shape, position)
    return batch_ops.stack([Element(part) for part in parts]).data


def parts_of_records(  # noqa: DOC502 - _split_records and stack_records raise
    records: Sequence[Any],
) -> tuple[PyTree, tuple[dict[str, Any], ...]]:
    """Turn records into columns and provenance: the one records-to-parts path of eager sources.

    Each record's numeric values (:func:`split_record`) are stacked into columns once, on the
    host (:func:`stack_records`); its strings, bytes and other objects become its provenance.

    Args:
        records: The records, each a mapping of values (nested mappings included) or a value.

    Returns:
        The host columns, and one provenance mapping per record, or none when no record carries
        anything but numbers.

    Raises:
        ValueError: If a record holds no numeric value, or the records' numeric fields or shapes
            differ (see :func:`stack_records`).
    """
    if not records:
        return {}, ()
    parts, provenance = _split_records(records)
    return stack_records(parts), provenance


def ragged_parts_of_records(  # noqa: DOC502 - _split_records and stack_records raise
    records: Sequence[Any],
) -> tuple[PyTree, tuple[dict[str, Any], ...], dict[str, RaggedColumn]]:
    """Turn records into columns, provenance and ragged fields, for a source holding ragged fields.

    As :func:`parts_of_records`, except that a field whose records differ along their first axis
    only becomes a ragged field (:func:`split_ragged`) instead of being refused.

    Args:
        records: The records, each a mapping of values (nested mappings included) or a value.

    Returns:
        The host columns, one provenance mapping per record (or none), and the ragged fields.

    Raises:
        ValueError: If a record holds no numeric value, or the records' numeric fields differ, or
            a field's records differ past their first axis (see :func:`stack_records`).
    """
    if not records:
        return {}, (), {}
    parts, provenance = _split_records(records)
    parts, ragged = split_ragged(parts)
    return (stack_records(parts) if jax.tree.leaves(parts[0]) else {}), provenance, ragged


def _split_records(records: Sequence[Any]) -> tuple[list[PyTree], tuple[dict[str, Any], ...]]:
    """Each record's array part and its provenance (:func:`split_record`), refusing an empty part.

    Args:
        records: The records.

    Returns:
        The array parts, and one provenance mapping per record, or none when no record carries
        anything but numbers.

    Raises:
        ValueError: If a record holds no numeric value.
    """
    parts, provenance = [], []
    for position, record in enumerate(records):
        part, extra = split_record(record)
        if part is None:
            raise ValueError(
                f"record {position} holds no numeric value; a batch is built from numeric "
                "fields, and strings and other objects are kept as the record's provenance"
            )
        parts.append(part)
        provenance.append(extra)
    return parts, tuple(provenance) if any(provenance) else ()


def split_ragged(parts: Sequence[PyTree]) -> tuple[list[PyTree], dict[str, RaggedColumn]]:
    """Take the fields whose records differ along their first axis only out of records' parts.

    Such a field becomes a :class:`~datarax.core.ragged.RaggedColumn`, named by its path; the
    rest of each part is returned for stacking. Records with different fields, or a field
    differing past its first axis, are left for :func:`stack_records` to refuse, naming them. A
    record that is a single value is not a field and is not split.

    Args:
        parts: One array part per record, as :func:`split_record` returns them.

    Returns:
        The parts without their ragged fields (an empty mapping where nothing is left), and the
        ragged fields by name.
    """
    found = _ragged_leaves(parts)
    if not found:
        return list(parts), {}
    kept = [_without(part, set(found), ()) for part in parts]
    return [{} if part is _ABSENT else part for part in kept], dict(found.values())


def _ragged_leaves(parts: Sequence[PyTree]) -> dict[tuple[Any, ...], tuple[str, RaggedColumn]]:
    """The fields of records' parts that are ragged, by their key path: each name and column.

    Args:
        parts: One array part per record.

    Returns:
        Each ragged field's key path, mapped to its name and its ``RaggedColumn``; none when the
        records hold different fields (which :func:`stack_records` refuses).
    """
    structure = jax.tree.structure(parts[0])
    if any(jax.tree.structure(part) != structure for part in parts):
        return {}
    flat = [jax.tree_util.tree_flatten_with_path(part)[0] for part in parts]
    found: dict[tuple[Any, ...], tuple[str, RaggedColumn]] = {}
    for column, (path, _) in enumerate(flat[0]):
        leaves = [np.asarray(record[column][1]) for record in flat]
        if path and is_ragged([leaf.shape for leaf in leaves]):
            key = tuple(getattr(step, "key", step) for step in path)
            found[key] = (field_name(path), ragged_column(leaves))
    return found


def _without(node: Any, removed: set[tuple[Any, ...]], path: tuple[Any, ...]) -> Any:
    """``node`` without the leaves at ``removed`` paths, and without mappings left empty."""
    if isinstance(node, Mapping):
        kept = {}
        for key, child in node.items():
            value = _without(child, removed, (*path, key))
            if value is not _ABSENT:
                kept[key] = value
        return kept if kept else _ABSENT
    return _ABSENT if path in removed else node


def column_length(columns: PyTree) -> int:
    """The number of records held by ``columns``: the leading axis every column shares.

    Args:
        columns: A pytree of arrays with a leading record axis.

    Returns:
        The record count; 0 for no columns.

    Raises:
        ValueError: If the columns disagree on their record count.
    """
    flat = jax.tree_util.tree_flatten_with_path(columns)[0]
    lengths = {field_name(path): len(leaf) for path, leaf in flat}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"columns must hold one row per record; their lengths are {lengths}")
    return next(iter(lengths.values()), 0)


def take_rows(columns: PyTree, rows: np.ndarray | slice) -> PyTree:
    """Gather ``rows`` of every column on the host: the one host gather of in-memory sources.

    A slice reads a contiguous run as views; an index array reads any rows, in order, with NumPy
    indexing (a copy).

    Args:
        columns: Host NumPy columns.
        rows: A slice of rows, or row numbers.

    Returns:
        The rows, with the columns' structure.
    """
    return jax.tree.map(lambda column: column[rows], columns)


def read_host_batch(  # noqa: DOC502 - record_words, host_rows and run_words raise
    columns: PyTree,
    length: int,
    indices: ArrayLike,
    *,
    epochs: ArrayLike = 0,
    contiguous: bool = False,
) -> Batch:
    """Read the rows ``indices`` names from host ``columns``, as a ``Batch`` named with them.

    The one host read of indexed sources whose records sit in host arrays (in memory, or a
    memory map): the words are checked (:func:`~datarax.core.data_source.host_rows`), the rows
    gathered with :func:`take_rows`, and the ``Batch`` named with the given words and epochs, with
    draws 0. No device array is created. A run declared ``contiguous`` is read as views of the
    columns; only its ends are checked, and its rows are named as the run.

    Args:
        columns: Host arrays with a leading record axis of ``length`` rows.
        length: The record count.
        indices: uint32 ``(n, 2)`` record indices, each as its words ``(hi, lo)``.
        epochs: The epoch of every record, or of each ``(n,)``.
        contiguous: Whether ``indices`` is a run of consecutive records.

    Returns:
        The records as a host ``Batch``.

    Raises:
        ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, or a run declared contiguous
            is not one.
    """
    words = record_words(indices)
    rows = host_rows(words, length)
    if contiguous and len(rows):
        words = run_words(rows)
        first = int(rows[0])
        data = take_rows(columns, slice(first, first + len(rows)))
    else:
        data = take_rows(columns, rows)
    return named_batch(data, words, epochs)


def run_words(rows: np.ndarray) -> np.ndarray:
    """The words naming ``rows``, which a read declared contiguous: a run of consecutive records.

    Only the run's ends are checked; its rows are named as the run.

    Args:
        rows: The read's row numbers, at least one.

    Returns:
        uint32 ``(n, 2)`` words of ``rows[0] .. rows[0] + n``.

    Raises:
        ValueError: If the ends do not bound a run of ``len(rows)`` records.
    """
    first = int(rows[0])
    if int(rows[-1]) != first + len(rows) - 1:
        raise ValueError(
            f"a read declared contiguous names a run of records; {first} to "
            f"{int(rows[-1])} is not a run of {len(rows)}"
        )
    return to_words(np.arange(first, first + len(rows), dtype=np.uint64))


def named_batch(data: PyTree, words: np.ndarray, epochs: ArrayLike) -> Batch:
    """A host ``Batch`` of ``data`` named with ``words`` and ``epochs``, its draws 0.

    Args:
        data: Host columns with one row per record.
        words: uint32 ``(n, 2)`` record indices.
        epochs: The epoch of every record, or of each ``(n,)``.

    Returns:
        The ``Batch``.
    """
    epoch_of_each = np.broadcast_to(np.asarray(epochs, np.int32), (len(words),))
    return batch_ops.from_arrays(data, indices=words, epochs=np.ascontiguousarray(epoch_of_each))


def _agreed_length(columns: PyTree) -> int | None:
    """The record count the columns agree on, or ``None`` when they disagree (refused at use)."""
    try:
        return column_length(columns)
    except ValueError:
        return None


# The device copies of eager sources' columns that a datarax compiled step, or a caller passing
# them, bound for the trace now running, by the source module's identity.
_BOUND_COLUMNS: contextvars.ContextVar[Mapping[int, PyTree]] = contextvars.ContextVar(
    "datarax_bound_columns", default=MappingProxyType({})
)


@contextlib.contextmanager
def bind_columns(sources: Sequence[EagerSource], columns: Sequence[PyTree]) -> Iterator[None]:
    """Bind each source's columns (device copies or tracers) for the reads made inside the block.

    Args:
        sources: The eager sources, as :func:`eager_sources` lists them.
        columns: Their columns, in the same order.

    Yields:
        None: Control to the block.

    Raises:
        ValueError: If the counts differ.
    """
    if len(sources) != len(columns):
        raise ValueError(
            f"{len(columns)} column sets given for {len(sources)} in-memory sources; pass "
            "what device_columns() returned, as it returned it"
        )
    token = _BOUND_COLUMNS.set(
        MappingProxyType(
            {**_BOUND_COLUMNS.get(), **{id(s): c for s, c in zip(sources, columns, strict=True)}}
        )
    )
    try:
        yield
    finally:
        _BOUND_COLUMNS.reset(token)


def bound_columns(source: EagerSource) -> PyTree | None:
    """The columns bound for ``source`` in the trace now running, or ``None``."""
    return _BOUND_COLUMNS.get().get(id(source))


def eager_sources(module: nnx.Module) -> list[EagerSource]:
    """Every in-memory source in ``module``'s graph, in graph order."""
    return [node for _, node in nnx.iter_graph(module, graph=True) if isinstance(node, EagerSource)]


class EagerSource(DataSourceModule):
    """An indexed source over records held in host memory as NumPy columns and provenance.

    A subclass turns its input into columns and provenance and stores them with :meth:`_store`;
    everything else (length, indexing, iteration, the host read, the traced gather, the order and
    the spec) is the base's. Columns given as device arrays are copied to the host once, where
    they are stored. Nothing in it is random and it keeps no iteration state: the pipeline that
    serves it owns the order and the position.

    A subclass may also store ragged fields (:mod:`datarax.core.ragged`), which it then serves,
    with every column, through the ragged read packing plans and packs from; while it holds one,
    every unpacked read is refused, since a batch's shapes are static.
    """

    _READS_RAGGED: ClassVar[bool] = True
    """Whether the source serves the ragged read; a subclass whose records packing does not read
    yet sets it ``False`` and refuses it as every source does."""

    @property
    def data(self) -> PyTree:
        """The columns, NumPy arrays held on the host (see :class:`HostColumns`)."""
        return self._columns.value

    @data.setter
    def data(self, columns: PyTree) -> None:
        """Hold ``columns`` as host NumPy arrays; columns of unequal length are refused at use.

        Args:
            columns: A pytree of arrays with one row per record.
        """
        host = jax.tree.map(np.asarray, columns)
        self._columns = HostColumns(host)
        self._length = _agreed_length(host)

    @property
    def record_identity(self) -> RecordIdentity:
        """An in-memory record's index is its stable position in the source."""
        return RecordIdentity.INDEXED

    def __init__(self, config: StructuralConfig, *, name: str | None = None) -> None:
        """Create a source holding no records yet.

        Args:
            config: The source's configuration.
            name: Optional module name.
        """
        super().__init__(config, name=name)
        self.data = {}
        self._provenance = HostProvenance(())
        self._ragged = HostRagged(MappingProxyType({}))
        self._ragged_names: tuple[str, ...] = ()

    def _store(
        self,
        columns: PyTree,
        provenance: Sequence[Mapping[str, Any]] = (),
        ragged: Mapping[str, RaggedColumn] | None = None,
    ) -> None:
        """Hold ``columns`` as host NumPy arrays, and ``provenance`` and ``ragged`` beside them.

        Args:
            columns: A pytree of arrays with one row per record; device arrays are copied to the
                host here, once.
            provenance: One mapping per record, or nothing.
            ragged: The ragged fields by name, or none.

        Raises:
            ValueError: If the columns and ragged fields disagree on their record count, or the
                provenance is not one mapping per record.
        """
        host = jax.tree.map(np.asarray, columns)
        ragged = dict(ragged or {})
        counts = {name: len(field) for name, field in ragged.items()}
        if jax.tree.leaves(host):
            counts["columns"] = column_length(host)
        if len(set(counts.values())) > 1:
            raise ValueError(f"every field holds one row per record; their records are {counts}")
        length = next(iter(counts.values()), 0)
        if provenance and len(provenance) != length:
            raise ValueError(
                f"provenance holds {len(provenance)} records but the columns hold {length}"
            )
        self.data = host
        self._length = length
        self._provenance = HostProvenance(tuple(MappingProxyType(dict(p)) for p in provenance))
        self._ragged = HostRagged(MappingProxyType(ragged))
        self._ragged_names = tuple(ragged)

    @property
    def ragged_fields(self) -> tuple[str, ...]:
        """The fields whose records differ in length along their first axis, by name.

        Kept as plain names beside the holder, as :attr:`length` is, so the reads every program
        makes check them without reading the holder.
        """
        return self._ragged_names

    def lengths(self, indices: ArrayLike, field: str) -> np.ndarray:  # noqa: DOC502 - checks raise
        """Each named record's length in ``field``: its array's along the first axis, a scalar 1.

        Args:
            indices: uint32 ``(n, 2)`` record indices.
            field: A field's name (its path, ``"text/tokens"``).

        Returns:
            int64 ``(n,)``, in the order named.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words or the field is not held.
            IndexError: If an index names no record or is outside the source.
        """
        if not self._READS_RAGGED:
            return super().lengths(indices, field)
        rows = host_rows(record_words(indices), self.length)
        held = self._field(field)
        if isinstance(held, RaggedColumn):
            return ragged_lengths(held, rows)
        return column_lengths(held, rows)

    def get_ragged(  # noqa: DOC502 - the checks it calls raise
        self, indices: ArrayLike, field: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """The named records of ``field``, as one values array and int64 offsets, on the host.

        Record ``k`` of the reply is ``values[offsets[k]:offsets[k + 1]]``; a column is read as
        records of its row length (a scalar column as records of one cell), so every field reads
        the same way. One vectorised gather, no loop over records.

        Args:
            indices: uint32 ``(n, 2)`` record indices, any order, repeats allowed.
            field: A field's name (its path, ``"text/tokens"``).

        Returns:
            The records' cells, record after record, and their offsets, ``n + 1`` of them from 0.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words or the field is not held.
            IndexError: If an index names no record or is outside the source.
        """
        if not self._READS_RAGGED:
            return super().get_ragged(indices, field)
        rows = host_rows(record_words(indices), self.length)
        held = self._field(field)
        if isinstance(held, RaggedColumn):
            return take_ragged(held, rows)
        return take_column(held, rows)

    def _field(self, field: str) -> RaggedColumn | np.ndarray:
        """The ragged field or column named ``field``.

        Args:
            field: The field's name.

        Returns:
            The field.

        Raises:
            ValueError: If the source holds no field of that name, naming those it holds.
        """
        ragged = self._ragged.value
        if field in ragged:
            return ragged[field]
        columns = {
            field_name(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(self.data)[0]
        }
        if field not in columns:
            raise ValueError(
                f"{type(self).__name__} holds no field {field!r}; its fields are "
                f"{sorted([*ragged, *columns])}"
            )
        return columns[field]

    def _refuse_unpacked(self) -> None:
        """Refuse an unpacked read of a source holding a ragged field, naming the field and fixes.

        It reads the plain field names only, never the holder: a traced path reads a stand-in of
        the source whose holders are empty.

        Raises:
            ValueError: If the source holds a ragged field.
        """
        if not self._ragged_names:
            return
        raise ValueError(
            f"field {self._ragged_names[0]!r} is ragged (its records differ in length along their "
            "first axis), but a batch's shapes are static: pack records with segment ids, or pad "
            "the field to a fixed length along its first axis and keep its mask or length in data"
        )

    @property
    def length(self) -> int:
        """Records the columns hold now (see :func:`column_length`).

        Kept as a plain count beside the holder, so a stand-in of the source holding its
        description only (the host naming's) still has it.
        """
        return column_length(self.data) if self._length is None else self._length

    def index_space(self) -> int:
        """Every stored row has an index, a worker's shard included: the data's record count."""
        return self.length

    def __len__(self) -> int:
        """Return the number of records."""
        return self.length

    def order_working_bytes(self, order: BlockOrder) -> int:  # noqa: ARG002 - nothing to hold
        """0: the records are host columns, already the training process's resident memory.

        Args:
            order: The block order a run reads in.

        Returns:
            0.
        """
        return 0

    def __getitem__(self, index: int) -> PyTree:  # noqa: DOC503 - _refuse_unpacked raises one
        """Return record ``index`` (negative counts from the end): its array part.

        Args:
            index: The record's position.

        Returns:
            The record's arrays, NumPy rows of the columns.

        Raises:
            IndexError: If ``index`` is outside the source.
            ValueError: If the source holds a ragged field (see :meth:`_refuse_unpacked`).
        """
        self._refuse_unpacked()
        length = self.length
        row = index + length if index < 0 else index
        if not 0 <= row < length:
            raise IndexError(f"Index {index} out of range for {length} records")
        return jax.tree.map(lambda column: column[row], self.data)

    def __iter__(self) -> Iterator[PyTree]:
        """Iterate over the records in order, each its array part; stateless.

        Returns:
            The records.
        """
        self._refuse_unpacked()
        return (self[row] for row in range(self.length))

    def _getitems(self, indices: Sequence[int]) -> list[PyTree]:
        """Read several records for Grain's batched random access, with one host gather.

        Args:
            indices: Record positions.

        Returns:
            Each record's arrays, in order.
        """
        self._refuse_unpacked()
        resolved = validate_index_batch(indices, self.length)
        rows = take_rows(self.data, np.asarray(resolved, dtype=np.intp))
        return [jax.tree.map(lambda column, k=k: column[k], rows) for k in range(len(resolved))]

    def get_batch(  # noqa: DOC502 - read_host_batch raises
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the records ``indices`` names, as a ``Batch`` named with them, on the host.

        One host gather of the array part (:func:`read_host_batch`); the batch's ``indices`` are
        the given words, its ``epochs`` the given epochs and its draws 0. It reads and changes no
        state and creates no device array. With ``contiguous=True`` the caller states that
        ``indices`` is a run of consecutive records, which is read as views of the columns; only
        the run's ends are checked, and its rows are named as the run.

        Args:
            indices: uint32 ``(n, 2)`` record indices, each as its words ``(hi, lo)``
                (:func:`~datarax.core.index_words.to_words` names plain positions).
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive records.

        Returns:
            The records as a host ``Batch``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, a run declared
                contiguous is not one, or the source holds a ragged field.
            IndexError: If an index is the padding index or outside the source.
        """
        self._refuse_unpacked()
        return read_host_batch(
            self.data, self.length, indices, epochs=epochs, contiguous=contiguous
        )

    def provenance(  # noqa: DOC502 - record_words and host_rows raise
        self, indices: ArrayLike
    ) -> tuple[Mapping[str, Any], ...]:
        """The provenance of the records ``indices`` names, read from the host holder by row.

        Args:
            indices: uint32 ``(n, 2)`` record indices.

        Returns:
            Each named record's immutable mapping of strings and objects, in the order named; an
            empty mapping for a record that carries nothing but arrays.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index is the padding index or outside the source.
        """
        rows = host_rows(record_words(indices), self.length)
        held = self._provenance.value
        if not held:
            return (NO_PROVENANCE,) * len(rows)
        return tuple(held[int(row)] for row in rows)

    def device_columns(self) -> PyTree:
        """The columns' device copy, for a caller's transform around :meth:`get_records`.

        The columns are host data, not NNX state, so the transform takes them as an argument:
        get them here, outside it, and pass them as ``get_records(indices, columns=...)``.

        Returns:
            The columns' structure, each leaf a JAX array.
        """
        self._refuse_unpacked()
        return jax.device_put(self.data)

    def get_records(self, indices: jax.Array, *, columns: PyTree | None = None) -> PyTree:
        """Gather the records at ``indices``; JIT-traceable, for the compiled pipeline path.

        The columns are host data, never NNX state, so a traced read takes their device copy as an
        argument: ``columns``, or the copy datarax's own compiled step bound. Called eagerly with
        neither, it gathers from the host columns.

        Args:
            indices: uint32 ``(n, 2)`` record indices in ``[0, index_space())``, as
                :meth:`record_indices_at` names them; concrete or traced.
            columns: The columns' device copy, as :meth:`device_columns` returns it (or the
                source's entry of ``pipeline.device_columns()``), passed into the caller's
                transform.

        Returns:
            The columns' structure, each leaf a JAX array with leading dimension ``n``.

        Raises:
            TypeError: If ``indices`` are traced and no columns are given or bound.
        """
        self._refuse_unpacked()
        rows = low_words(indices, self.length)
        if columns is None:
            columns = bound_columns(self)
        if columns is None:
            if isinstance(rows, Tracer):
                raise TypeError(
                    f"{type(self).__name__}.get_records is traced, but its columns are host data "
                    "and not NNX state, so the trace cannot read them: pass their device copy "
                    "in as an argument: columns = source.device_columns() outside the transform "
                    "and get_records(indices, columns=columns) inside it, or, for a pipeline, "
                    "columns = pipeline.device_columns() and step(columns=columns) or "
                    "scan(..., columns=columns)"
                )
            return jax.tree.map(
                lambda column: jnp.asarray(np.take(column, rows, axis=0)), self.data
            )
        return jax.tree.map(lambda column: jnp.take(column, rows, axis=0), columns)

    def element_spec(self) -> Any:
        """The spec of one record as the device holds it, read from the columns' metadata.

        Every column's leading record axis is stripped and its dtype stated as JAX arrays hold
        it (``device_spec``: while x64 is off a stored ``float64`` is declared ``float32``).

        Returns:
            A pytree of ``jax.ShapeDtypeStruct`` with the columns' structure.

        Raises:
            ValueError: If the source holds no records, or holds a ragged field.
        """
        self._refuse_unpacked()
        if self.length == 0:
            raise ValueError(
                f"{type(self).__name__} has zero elements; element_spec() cannot be inferred "
                "from an empty dataset."
            )
        return device_spec(jax.tree.map(array_to_spec_strip_leading, self.data))


__all__ = [
    "RAGGED_SOURCES",
    "EagerSource",
    "column_length",
    "is_array_leaf",
    "parts_of_records",
    "past_first_axis_refusal",
    "ragged_parts_of_records",
    "read_host_batch",
    "shape_refusal",
    "split_ragged",
    "split_record",
    "stack_records",
    "take_rows",
]
