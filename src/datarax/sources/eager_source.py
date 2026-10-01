"""In-memory sources: records held on the host and read by one host gather.

An ``EagerSource`` holds its records in two parts. The array part is a pytree of host NumPy
columns, one row per record: the only thing the host read, ``get_records`` and ``element_spec``
read. The non-array part (strings, bytes and other Python objects) is each record's provenance, an
immutable mapping per record kept in a host holder that NNX leaves out of module state and out of
every trace; it is never part of a ``Batch``. ``MemorySource``, ``TFDSEagerSource`` and
``HFEagerSource`` build on it, each turning its own input into the two parts.

The host read, ``get_batch(indices, *, epochs=0)``, gathers the named records with NumPy indexing
(a contiguous run as views) and returns a ``Batch`` named with the given indices and epochs. It
reads and changes no state and allocates no device array. The traced read, ``get_records``,
serves the compiled pipeline path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.core import batch_ops
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch, Element, PADDING_INDEX
from datarax.core.index_words import from_words, low_words, to_words
from datarax.core.spec import array_to_spec_strip_leading, device_spec
from datarax.sources._grain_bridge import validate_index_batch
from datarax.sources.source_ops import resolve_wrapped_indices


_NUMERIC_KINDS = frozenset("biufc")


class HostProvenance:
    """The non-array part of a source's records, held on the host and out of NNX state.

    As a plain object it is static to NNX: it is not module state, never a jit argument and never
    traced. It compares equal to every other holder, so two sources differing only in their
    provenance share one graphdef and one compiled program; nothing traced reads it.

    Attributes:
        value: One immutable mapping per record, aligned with the rows, or empty when the
            records carry nothing but arrays.
    """

    __slots__ = ("value",)

    def __init__(self, value: tuple[Mapping[str, Any], ...]) -> None:
        """Hold ``value``.

        Args:
            value: The per-record mappings.
        """
        self.value = value

    def __eq__(self, other: object) -> bool:
        """Every holder is equal to every other: provenance never decides a program."""
        return isinstance(other, HostProvenance)

    def __hash__(self) -> int:
        """One hash for every holder, as ``__eq__`` requires."""
        return hash(HostProvenance)


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
            return np.dtype(getattr(value, "dtype", type(value))).kind in _NUMERIC_KINDS
        except TypeError:  # an extended dtype, such as a PRNG key's, is not a column
            return False
    if isinstance(value, list | tuple):
        try:
            return np.asarray(value).dtype.kind in _NUMERIC_KINDS
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
    provenance["/".join(path)] = node
    return _ABSENT


def _field_paths(part: PyTree) -> list[str]:
    return [
        "/".join(str(getattr(key, "key", key)) for key in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(part)[0]
    ]


def stack_records(parts: Sequence[PyTree]) -> PyTree:
    """Stack records' array parts into columns, once, on the host.

    Every record must hold the same numeric fields with the same shapes: a batch's shapes are
    static, so a field whose shape varies is refused, naming the field, both shapes and the fix.

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
                f"record {position} holds the numeric fields {_field_paths(part)} where record 0 "
                f"holds {_field_paths(parts[0])}; every record holds the same fields"
            )
        for field, first, shape in zip(
            _field_paths(part),
            shapes,
            (np.shape(leaf) for leaf in jax.tree.leaves(part)),
            strict=True,
        ):
            if shape != first:
                raise ValueError(
                    f"field {field!r} is {first} in record 0 and {shape} in record {position}, but "
                    "a batch's shapes are static: pad the field to a fixed length along its own "
                    "axis and keep its mask or length in data, or pack records with segment ids"
                )
    return batch_ops.stack([Element(part) for part in parts]).data


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
    lengths = {"/".join(str(getattr(k, "key", k)) for k in path): len(leaf) for path, leaf in flat}
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


class EagerSource(DataSourceModule):
    """An indexed source over records held in host memory as NumPy columns and provenance.

    A subclass turns its input into columns and provenance and stores them with :meth:`_store`;
    everything else (length, indexing, iteration, the host read, the traced gather, the order and
    the spec) is the base's. Columns given as device arrays are copied to the host once, where
    they are stored. Nothing in it is random and it keeps no iteration state: the pipeline that
    serves it owns the order and the position.
    """

    data: PyTree

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
        self.data = nnx.data({})
        self._provenance = HostProvenance(())

    def _store(self, columns: PyTree, provenance: Sequence[Mapping[str, Any]] = ()) -> None:
        """Hold ``columns`` as host NumPy arrays and ``provenance`` beside them.

        Args:
            columns: A pytree of arrays with one row per record; device arrays are copied to the
                host here, once.
            provenance: One mapping per record, or nothing.

        Raises:
            ValueError: If the columns disagree on their record count, or the provenance is not
                one mapping per record.
        """
        host = jax.tree.map(np.asarray, columns)
        length = column_length(host)
        if provenance and len(provenance) != length:
            raise ValueError(
                f"provenance holds {len(provenance)} records but the columns hold {length}"
            )
        self.data = nnx.data(host)
        self._provenance = HostProvenance(tuple(MappingProxyType(dict(p)) for p in provenance))

    @property
    def length(self) -> int:
        """Records the columns hold now (see :func:`column_length`)."""
        return column_length(self.data)

    def __len__(self) -> int:
        """Return the number of records."""
        return self.length

    def __getitem__(self, index: int) -> PyTree:
        """Return record ``index`` (negative counts from the end): its array part.

        Args:
            index: The record's position.

        Returns:
            The record's arrays, NumPy rows of the columns.

        Raises:
            IndexError: If ``index`` is outside the source.
        """
        length = self.length
        row = index + length if index < 0 else index
        if not 0 <= row < length:
            raise IndexError(f"Index {index} out of range for {length} records")
        return jax.tree.map(lambda column: column[row], self.data)

    def __iter__(self) -> Iterator[PyTree]:
        """Iterate over the records in order, each its array part; stateless."""
        return (self[row] for row in range(self.length))

    def _getitems(self, indices: Sequence[int]) -> list[PyTree]:
        """Read several records for Grain's batched random access, with one host gather.

        Args:
            indices: Record positions.

        Returns:
            Each record's arrays, in order.
        """
        resolved = validate_index_batch(indices, self.length)
        rows = take_rows(self.data, np.asarray(resolved, dtype=np.intp))
        return [jax.tree.map(lambda column, k=k: column[k], rows) for k in range(len(resolved))]

    def get_batch(  # noqa: DOC503 - _host_rows raises the IndexError
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the records ``indices`` names, as a ``Batch`` named with them, on the host.

        One host gather of the array part; the batch's ``indices`` are the given words, its
        ``epochs`` the given epochs and its draws 0. It reads and changes no state and creates no
        device array. With ``contiguous=True`` the caller states that ``indices`` is a run of
        consecutive records, which is read as views of the columns; only the run's ends are
        checked, and its rows are named as the run.

        Args:
            indices: uint32 ``(n, 2)`` record indices, each as its words ``(hi, lo)``
                (:func:`~datarax.core.index_words.to_words` names plain positions).
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive records.

        Returns:
            The records as a host ``Batch``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, or a run declared
                contiguous is not one.
            IndexError: If an index is the padding index or outside the source.
        """
        words = np.asarray(indices)
        if words.dtype != np.uint32 or words.ndim != 2 or words.shape[1] != 2:  # noqa: PLR2004
            raise ValueError(
                "record indices are uint32 (n, 2) words (hi, lo), as index_words.to_words names "
                f"positions; got {words.dtype} {words.shape}"
            )
        rows = self._host_rows(words)
        if contiguous and len(rows):
            first = int(rows[0])
            if int(rows[-1]) != first + len(rows) - 1:
                raise ValueError(
                    f"a read declared contiguous names a run of records; {first} to "
                    f"{int(rows[-1])} is not a run of {len(rows)}"
                )
            data = take_rows(self.data, slice(first, first + len(rows)))
            words = to_words(np.arange(first, first + len(rows), dtype=np.uint64))
        else:
            data = take_rows(self.data, rows)
        epoch_of_each = np.broadcast_to(np.asarray(epochs, np.int32), (len(rows),))
        return batch_ops.from_arrays(data).replace(
            indices=words, epochs=np.ascontiguousarray(epoch_of_each)
        )

    def _host_rows(self, words: np.ndarray) -> np.ndarray:
        """The rows ``words`` name, refusing the padding index and records outside the source.

        Args:
            words: uint32 ``(n, 2)`` record indices.

        Returns:
            uint32 ``(n,)`` row numbers.

        Raises:
            IndexError: If an index is the padding index or outside the source.
        """
        length = self.length
        padding = np.all(words == PADDING_INDEX, axis=1)
        if padding.any():
            raise IndexError(
                f"index {int(np.argmax(padding))} is the padding index (all ones), which names "
                "no record"
            )
        outside = (words[:, 0] != 0) | (words[:, 1] >= length)
        if outside.any():
            raise IndexError(
                f"record index {int(from_words(words[outside][:1])[0])} is outside [0, {length})"
            )
        return low_words(words, length)

    def record_indices_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Return the index of each record at positions ``start .. start + size``.

        Args:
            start: Starting position; concrete int or traced ``jax.Array``.
            size: Number of records (Python int).
            key: The key selecting the order, or ``None`` for the sequential order.

        Returns:
            uint32 ``jax.Array`` of shape ``(size, 2)``, each index as its words ``(hi, lo)``.
        """
        return resolve_wrapped_indices(start, size, self.length, key)

    def get_records(self, indices: jax.Array) -> PyTree:
        """Gather the records at ``indices``; JIT-traceable, for the compiled pipeline path.

        Args:
            indices: uint32 ``(n, 2)`` record indices in ``[0, len(self))``, as
                :meth:`record_indices_at` names them; concrete or traced.

        Returns:
            The columns' structure, each leaf a JAX array with leading dimension ``n``.
        """
        rows = low_words(indices, self.length)
        return jax.tree.map(lambda column: jnp.take(jnp.asarray(column), rows, axis=0), self.data)

    def element_spec(self) -> Any:
        """The spec of one record as the device holds it, read from the columns' metadata.

        Every column's leading record axis is stripped and its dtype stated as JAX arrays hold
        it (``device_spec``: while x64 is off a stored ``float64`` is declared ``float32``).

        Returns:
            A pytree of ``jax.ShapeDtypeStruct`` with the columns' structure.

        Raises:
            ValueError: If the source holds no records.
        """
        if self.length == 0:
            raise ValueError(
                f"{type(self).__name__} has zero elements; element_spec() cannot be inferred "
                "from an empty dataset."
            )
        return device_spec(jax.tree.map(array_to_spec_strip_leading, self.data))


__all__ = [
    "EagerSource",
    "HostProvenance",
    "column_length",
    "is_array_leaf",
    "split_record",
    "stack_records",
    "take_rows",
]
