"""Element specs: building them, and checking data against them.

Datarax modules (sources, operators, batchers, samplers) declare their output
PyTree shape and dtype as a ``PyTree[jax.ShapeDtypeStruct]`` so downstream
consumers can pre-allocate buffers, auto-size learnable layers, and validate
operator chains. This module centralizes the leaf-level helpers for building
those specs and for checking data against them.

Two descriptions of the same data are kept apart on purpose:

- :func:`array_to_spec` and :func:`array_to_spec_strip_leading` describe a value
  exactly as given. They read shape and dtype from array metadata, never copy
  data or move it to a device, and never change a dtype: a host ``float64``
  array is described as ``float64``.
- :func:`device_spec` describes the same data once converted to JAX arrays under
  the active ``jax_enable_x64`` setting, where 64-bit dtypes become 32-bit while
  x64 is off. A source whose batches are produced by that conversion declares
  ``device_spec`` of its stored data.

:func:`validate_batch` checks a batch against a declared element spec (tree
structure, per-element shapes, dtypes, one shared record count) without
coercing anything, and :func:`validate_device_dtypes` refuses a declared dtype
the device cannot hold as declared. Both report every problem with its field
path in one :class:`SpecMismatchError`. The checks read only static metadata,
so they also run on tracers while a function is traced and add nothing to the
compiled graph.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from numpy.typing import ArrayLike

from datarax.core.data_source import DataSourceModule


# NumPy dtype kinds a JAX array can hold: bool, signed, unsigned, float, complex.
_JAX_ARRAY_KINDS = frozenset("biufc")


class SpecMismatchError(ValueError):
    """Data or a spec disagrees with the element spec it is checked against.

    Attributes:
        problems: One message per disagreement, each naming the field it concerns.
    """

    def __init__(self, summary: str, problems: Sequence[str]) -> None:
        """Build the error from a one-line summary and its field-level problems.

        Args:
            summary: What was being checked.
            problems: One message per disagreement.
        """
        self.problems: tuple[str, ...] = tuple(problems)
        details = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{summary}\n{details}")


def add_leading_dim(spec_leaf: jax.ShapeDtypeStruct, size: int) -> jax.ShapeDtypeStruct:
    """Return a ``ShapeDtypeStruct`` with ``size`` prepended to ``spec_leaf.shape``.

    Used by Batchers to lift per-element specs into batch-level specs.
    """
    return jax.ShapeDtypeStruct(shape=(size, *spec_leaf.shape), dtype=spec_leaf.dtype)


def batched_spec(element_spec: Any, batch_size: int) -> Any:
    """Lift a per-element spec PyTree into the spec of a batch of ``batch_size`` records.

    The returned PyTree has the structure of ``element_spec`` with a leading ``batch_size``
    dimension prepended to every ``ShapeDtypeStruct`` leaf, and nothing added: every row of a
    batch is a record, so no leaf marks padding.

    Args:
        element_spec: PyTree of ``jax.ShapeDtypeStruct`` describing per-element
            output (typically a dict from a DataSourceModule's ``element_spec()``).
        batch_size: Number of elements per emitted batch.

    Returns:
        The batched spec PyTree.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    return jax.tree.map(
        lambda leaf: add_leading_dim(leaf, batch_size),
        element_spec,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )


def scalar_index_spec(dtype: DTypeLike = jnp.int32) -> jax.ShapeDtypeStruct:
    """Return the default ``ShapeDtypeStruct`` for a scalar sampler index.

    Most samplers emit a single integer index per call; specialized samplers
    (windowed, vectorized) override their tier's ``index_spec`` to return a
    different shape.
    """
    return jax.ShapeDtypeStruct(shape=(), dtype=dtype)


# Leaf types whose shape and dtype are read from metadata; tracers are ``jax.Array``.
_ARRAY_TYPES = np.ndarray | np.generic | jax.Array
# Leaf types that can carry a leading record axis; a NumPy scalar never does.
_BATCHABLE_TYPES = np.ndarray | jax.Array


def _shape_and_dtype(value: ArrayLike) -> tuple[tuple[int, ...], Any]:
    """Shape and dtype of ``value``, read from metadata when it is an array."""
    if isinstance(value, _ARRAY_TYPES):
        return tuple(value.shape), value.dtype
    host = np.asarray(value)
    return tuple(host.shape), host.dtype


def array_to_spec(value: ArrayLike) -> jax.ShapeDtypeStruct:
    """Return the shape and dtype of ``value`` exactly as given.

    NumPy arrays, NumPy scalars and JAX arrays (tracers included) are described
    from their metadata: nothing is copied, moved to a device or cast, so a host
    ``float64`` array is described as ``float64`` even while x64 is off. Other
    values (Python scalars, nested sequences) are read the way ``numpy.asarray``
    reads them. Use :func:`device_spec` for the same data as JAX arrays.

    Args:
        value: The value to describe.

    Returns:
        A ``ShapeDtypeStruct`` with the value's shape and dtype.
    """
    shape, dtype = _shape_and_dtype(value)
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


def array_to_spec_strip_leading(value: ArrayLike) -> jax.ShapeDtypeStruct:
    """Return the spec of one element of leading-batched data, exactly as given.

    Used by sources whose stored data has a leading dataset-size dimension (a
    dict of arrays each shaped ``(N, *element_shape)``). The leading axis is
    dropped and the dtype is kept as given, as in :func:`array_to_spec`.

    Args:
        value: Array-like whose leading axis is the dataset size.

    Returns:
        The spec of one element, with the leading axis removed.

    Raises:
        ValueError: If ``value`` has no leading axis to strip.
    """
    shape, dtype = _shape_and_dtype(value)
    if not shape:
        raise ValueError(
            "array_to_spec_strip_leading requires an array with at least one "
            "axis (got a 0-d array)."
        )
    return jax.ShapeDtypeStruct(shape=shape[1:], dtype=dtype)


def _field(path: tuple[Any, ...]) -> str:
    """Render a key path the way JAX error messages do, e.g. ``['image']``."""
    return jax.tree_util.keystr(path) or "<root>"


def _device_dtype(dtype: Any) -> Any | None:
    """The dtype a JAX array holds for data of ``dtype``, or None if it cannot hold it."""
    is_extended = jnp.issubdtype(dtype, jax.dtypes.extended)
    if not is_extended and np.dtype(dtype).kind not in _JAX_ARRAY_KINDS:
        return None
    return jax.dtypes.canonicalize_dtype(dtype, allow_extended_dtype=True)


def _device_dtypes_by_field(
    spec: Any,
) -> Iterator[tuple[tuple[Any, ...], jax.ShapeDtypeStruct, Any | None]]:
    """Each spec leaf with its key path and the dtype a JAX array holds for it."""
    for path, leaf in jax.tree.flatten_with_path(spec)[0]:
        yield path, leaf, _device_dtype(leaf.dtype)


def device_spec(spec: Any) -> Any:
    """Return ``spec`` as it reads once its data is converted to JAX arrays.

    Every leaf keeps its shape; its dtype becomes the one ``jnp.asarray`` gives
    that data under the active x64 setting (``jax.dtypes.canonicalize_dtype``).
    While x64 is off, ``float64`` becomes ``float32``, ``int64`` becomes
    ``int32``, ``uint64`` becomes ``uint32`` and ``complex128`` becomes
    ``complex64``; every other dtype, typed PRNG keys included, is unchanged.

    Args:
        spec: PyTree of ``jax.ShapeDtypeStruct`` describing data as given.

    Returns:
        The same PyTree with each leaf's dtype as a JAX array holds it.

    Raises:
        TypeError: If a leaf's dtype has no JAX array representation (strings,
            bytes, Python objects); the message names every such field.
    """
    converted: list[jax.ShapeDtypeStruct] = []
    unrepresentable: list[str] = []
    for path, leaf, dtype in _device_dtypes_by_field(spec):
        if dtype is None:
            unrepresentable.append(f"{_field(path)}: dtype {leaf.dtype}")
        else:
            converted.append(jax.ShapeDtypeStruct(shape=leaf.shape, dtype=dtype))
    if unrepresentable:
        raise TypeError(
            "These fields have no JAX array representation: " + "; ".join(unrepresentable)
        )
    return jax.tree.unflatten(jax.tree.structure(spec), converted)


def validate_device_dtypes(spec: Any) -> None:
    """Raise if data described by ``spec`` would change dtype on becoming JAX arrays.

    A declared dtype is a request. While x64 is off, JAX turns 64-bit data into
    32-bit arrays; this check refuses such a declaration instead of letting the
    precision change pass silently. Emit the dtype the device holds from the
    source, or enable ``jax_enable_x64``.

    Args:
        spec: PyTree of ``jax.ShapeDtypeStruct`` a source declares.

    Raises:
        SpecMismatchError: Naming every field whose dtype a JAX array cannot hold
            as declared.
    """
    x64_state = "on" if jax.config.jax_enable_x64 else "off"
    problems: list[str] = []
    for path, leaf, dtype in _device_dtypes_by_field(spec):
        if dtype is None:
            problems.append(f"{_field(path)}: dtype {leaf.dtype} has no JAX array representation")
        elif dtype != leaf.dtype:
            problems.append(
                f"{_field(path)}: declared {leaf.dtype} becomes {dtype} as a JAX array "
                f"(jax_enable_x64 is {x64_state})"
            )
    if problems:
        raise SpecMismatchError(
            "The declared element spec has dtypes JAX arrays cannot hold as declared; "
            "emit the dtype the device holds, or enable jax_enable_x64.",
            problems,
        )


def _uncovered(
    paths: Iterable[tuple[Any, ...]], reference: set[tuple[Any, ...]]
) -> list[tuple[Any, ...]]:
    """The shortest prefix of each path that ``reference`` has no field under, in order."""
    reference_prefixes = {path[:end] for path in reference for end in range(len(path) + 1)}
    found: list[tuple[Any, ...]] = []
    for path in paths:
        if path in reference:
            continue
        cut = next(
            (path[:end] for end in range(1, len(path) + 1) if path[:end] not in reference_prefixes),
            path,
        )
        if cut not in found:
            found.append(cut)
    return found


_Fields = dict[tuple[Any, ...], tuple[tuple[int, ...], Any]]


def _spec_fields(spec: Any) -> _Fields:
    """Shape and dtype of every spec leaf, by key path."""
    return {
        path: (tuple(leaf.shape), leaf.dtype) for path, leaf in jax.tree.flatten_with_path(spec)[0]
    }


def _field_mismatches(
    expected: _Fields,
    actual: _Fields,
    ignored: frozenset[tuple[Any, ...]] = frozenset(),
) -> list[str]:
    """Missing, unexpected, and differently shaped or typed fields of ``actual``."""
    problems = [f"{_field(path)}: missing" for path in _uncovered(expected, set(actual) | ignored)]
    problems.extend(f"{_field(path)}: unexpected" for path in _uncovered(actual, set(expected)))
    for path, (shape, dtype) in actual.items():
        if path not in expected:
            continue
        expected_shape, expected_dtype = expected[path]
        if shape != expected_shape:
            problems.append(f"{_field(path)}: shape {shape} != expected {expected_shape}")
        if dtype != expected_dtype:
            problems.append(f"{_field(path)}: dtype {dtype} != expected {expected_dtype}")
    return problems


def spec_mismatches(expected: Any, actual: Any) -> tuple[str, ...]:
    """Return one message per field where ``actual`` differs from ``expected``.

    Fields are matched by key path. A field only one spec has is reported once, at
    the shallowest path the other spec lacks; a field both have is compared by
    shape and dtype. Other ``ShapeDtypeStruct`` attributes (``weak_type``,
    sharding) are not compared. When every field matches but the containers
    differ (a list against a tuple, say), the tree structure difference is
    reported.

    Args:
        expected: The reference spec PyTree.
        actual: The spec PyTree compared with it.

    Returns:
        The mismatch messages; empty when the specs agree.
    """
    problems = _field_mismatches(_spec_fields(expected), _spec_fields(actual))
    expected_tree = jax.tree.structure(expected)
    actual_tree = jax.tree.structure(actual)
    if not problems and expected_tree != actual_tree:
        problems.append(f"tree structure differs: expected {expected_tree}, got {actual_tree}")
    return tuple(problems)


@dataclass(frozen=True, slots=True, kw_only=True)
class _BatchFields:
    """A batch's array leaves read as records, plus leaves that cannot hold records."""

    elements: _Fields
    lengths: dict[tuple[Any, ...], int]
    rejected: frozenset[tuple[Any, ...]]
    problems: tuple[str, ...]


def _without_trailing_indices(path: tuple[Any, ...]) -> tuple[Any, ...]:
    """Drop trailing sequence indices so a list of values reads as one field."""
    end = len(path)
    while end and isinstance(path[end - 1], jax.tree_util.SequenceKey):
        end -= 1
    return path[:end]


def _batch_fields(batch: Any) -> _BatchFields:
    """Read every leaf of ``batch`` as a leading record axis over one element."""
    elements: _Fields = {}
    lengths: dict[tuple[Any, ...], int] = {}
    rejected: list[tuple[Any, ...]] = []
    problems: list[str] = []
    for path, leaf in jax.tree.flatten_with_path(batch)[0]:
        if not isinstance(leaf, _ARRAY_TYPES):
            field = _without_trailing_indices(path)
            if field not in rejected:
                rejected.append(field)
                problems.append(f"{_field(field)}: holds {type(leaf).__name__} values, not arrays")
            continue
        if leaf.ndim == 0:
            rejected.append(path)
            problems.append(f"{_field(path)}: has no leading batch axis")
            continue
        lengths[path] = int(leaf.shape[0])
        elements[path] = (tuple(leaf.shape[1:]), leaf.dtype)
    return _BatchFields(
        elements=elements,
        lengths=lengths,
        rejected=frozenset(rejected),
        problems=tuple(problems),
    )


def _record_axis_problems(fields: _BatchFields) -> list[str]:
    """Every reason the leaves do not share one leading record axis, each naming its field."""
    problems = list(fields.problems)
    if len(set(fields.lengths.values())) > 1:
        listed = ", ".join(f"{_field(path)} has {n}" for path, n in fields.lengths.items())
        problems.append(f"leading axis lengths differ: {listed}")
    return problems


def batch_length(batch: Any) -> int | None:
    """Return the leading-axis length every leaf of ``batch`` shares.

    The common case is a flat pass over the leaves; the field-level report is
    built only when a leaf fails, the way JAX compares tree definitions before
    it explains a structure error.

    Args:
        batch: PyTree of arrays sharing a leading record axis.

    Returns:
        The shared length, or None when the batch has no leaves (an exhausted
        streaming source may return ``{}``).

    Raises:
        SpecMismatchError: If a leaf is not an array, has no leading axis, or the
            leaves disagree on the leading-axis length.
    """
    leaves = jax.tree.leaves(batch)
    if not leaves:
        return None
    counts = {_record_count(leaf) for leaf in leaves}
    if len(counts) == 1 and None not in counts:
        return counts.pop()
    raise SpecMismatchError(
        "The batch does not have one leading record axis.",
        _record_axis_problems(_batch_fields(batch)),
    )


def _record_count(leaf: Any) -> int | None:
    """Leading-axis length of a leaf that can hold records, or None for any other leaf."""
    if isinstance(leaf, _BATCHABLE_TYPES) and leaf.ndim:
        return leaf.shape[0]
    return None


def _batch_matches(batch: Any, element_spec: Any, batch_size: int | None) -> bool:
    """Whether ``batch`` satisfies ``element_spec``: the flat check run before any report."""
    batch_leaves, batch_tree = jax.tree.flatten(batch)
    spec_leaves, spec_tree = jax.tree.flatten(element_spec)
    counts = {_record_count(leaf) for leaf in batch_leaves}
    if batch_tree != spec_tree or None in counts or len(counts) > 1:
        return False
    count = next(iter(counts), None)
    if batch_size is not None and count is not None and not 1 <= count <= batch_size:
        return False
    return all(
        leaf.shape[1:] == expected.shape and leaf.dtype == expected.dtype
        for leaf, expected in zip(batch_leaves, spec_leaves, strict=True)
    )


def _batch_problems(batch: Any, element_spec: Any, batch_size: int | None) -> list[str]:
    """Every field-level reason ``batch`` does not satisfy ``element_spec``."""
    fields = _batch_fields(batch)
    problems = _record_axis_problems(fields)
    lengths = set(fields.lengths.values())
    if batch_size is not None and len(lengths) == 1:
        (length,) = lengths
        if not 1 <= length <= batch_size:
            problems.append(f"batch holds {length} records; expected 1 to {batch_size}")
    problems.extend(_field_mismatches(_spec_fields(element_spec), fields.elements, fields.rejected))
    expected_tree = jax.tree.structure(element_spec)
    batch_tree = jax.tree.structure(batch)
    if not problems and expected_tree != batch_tree:
        problems.append(f"tree structure differs: expected {expected_tree}, got {batch_tree}")
    return problems


def validate_batch(batch: Any, element_spec: Any, *, batch_size: int | None = None) -> None:
    """Check ``batch`` against the element spec its source declares.

    The batch must have the spec's tree structure, with one array per declared
    field shaped ``(n, *element_shape)`` in the declared dtype, and every field
    must hold the same record count ``n``. With ``batch_size``, ``n`` must be
    between 1 and ``batch_size``, so a short final batch passes. Dtypes are
    compared exactly: a ``float64`` batch does not satisfy a ``float32``
    declaration. Only shapes and dtypes are read, so no array is copied or
    converted, and the check runs on tracers while a function is traced without
    adding anything to the compiled graph. A matching batch costs one flat pass
    over its leaves; the field-level report is built only for a mismatch.

    Args:
        batch: PyTree of arrays with a leading record axis.
        element_spec: PyTree of ``jax.ShapeDtypeStruct`` describing one element.
        batch_size: Largest record count the batch may hold, when bounded.

    Raises:
        SpecMismatchError: Listing every problem found, each naming its field.
    """
    if not _batch_matches(batch, element_spec, batch_size):
        raise SpecMismatchError(
            "The batch does not match the declared element spec.",
            _batch_problems(batch, element_spec, batch_size),
        )


# Declared element specs per source, by x64 setting. Reading a spec can open a
# backend iterator (the TFDS and HuggingFace streaming sources peek their first
# record, which fills a shuffle buffer), so it is read once per source and
# precision mode instead of once per pass. Weak keys let entries die with sources.
_DECLARED_SPECS: weakref.WeakKeyDictionary[DataSourceModule, dict[bool, Any]] = (
    weakref.WeakKeyDictionary()
)


def declared_spec(source: DataSourceModule) -> Any:
    """Return ``source.element_spec()``, read once per source and x64 setting.

    Args:
        source: The data source whose declaration is needed.

    Returns:
        The element spec the source declared under the active x64 setting.
    """
    specs = _DECLARED_SPECS.setdefault(source, {})
    x64 = bool(jax.config.read("jax_enable_x64"))
    if x64 not in specs:
        specs[x64] = source.element_spec()
    return specs[x64]


__all__ = [
    "SpecMismatchError",
    "add_leading_dim",
    "array_to_spec",
    "array_to_spec_strip_leading",
    "batch_length",
    "batched_spec",
    "declared_spec",
    "device_spec",
    "scalar_index_spec",
    "spec_mismatches",
    "validate_batch",
    "validate_device_dtypes",
]
