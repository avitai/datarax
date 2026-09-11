"""Contracts for the element-spec helpers and the batch validator in ``datarax.core.spec``.

A spec describes data exactly as it is: ``array_to_spec`` reads shape and dtype
from array metadata without converting or copying anything, ``device_spec``
states what the same data becomes as JAX arrays under the active x64 setting,
and ``validate_batch`` compares a batch with a declared element spec without
coercing it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.spec import (
    array_to_spec,
    array_to_spec_strip_leading,
    batch_length,
    device_spec,
    spec_mismatches,
    SpecMismatchError,
    validate_batch,
    validate_device_dtypes,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _spec(shape: tuple[int, ...], dtype: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


@contextmanager
def _conversion_forbidden() -> Iterator[None]:
    """Fail if array data is converted, copied through ``asarray`` or moved to a device."""
    refuse = AssertionError("array data was converted while only its metadata was needed")
    with (
        patch("jax.numpy.asarray", side_effect=refuse),
        patch("jax.numpy.array", side_effect=refuse),
        patch("numpy.asarray", side_effect=refuse),
        patch("jax.device_put", side_effect=refuse),
    ):
        yield


_ELEMENT_SPEC = {"x": _spec((3,), np.float32), "y": _spec((), np.int32)}


def _batch(
    size: int, *, x_shape: tuple[int, ...] = (3,), x_dtype: Any = np.float32
) -> dict[str, np.ndarray]:
    return {
        "x": np.zeros((size, *x_shape), dtype=x_dtype),
        "y": np.zeros((size,), dtype=np.int32),
    }


class TestArrayToSpec:
    """``array_to_spec`` describes a value as given."""

    def test_host_arrays_keep_their_64_bit_dtype_when_x64_is_off(self) -> None:
        assert not jax.config.jax_enable_x64
        assert array_to_spec(np.zeros((2, 3), dtype=np.float64)) == _spec((2, 3), np.float64)
        assert array_to_spec(np.zeros((4,), dtype=np.int64)) == _spec((4,), np.int64)

    def test_arrays_and_numpy_scalars_are_described_from_metadata(self) -> None:
        jax_array = jnp.zeros((3,), dtype=jnp.float16)
        host_array = np.ones((5, 2), dtype=np.uint8)
        scalar = np.float64(1.5)
        with _conversion_forbidden():
            assert array_to_spec(jax_array) == _spec((3,), np.float16)
            assert array_to_spec(host_array) == _spec((5, 2), np.uint8)
            assert array_to_spec(scalar) == _spec((), np.float64)

    def test_python_values_are_read_the_way_numpy_reads_them(self) -> None:
        assert array_to_spec(1.0) == _spec((), np.float64)
        assert array_to_spec([[1, 2, 3]]) == _spec((1, 3), np.int64)

    def test_traced_values_are_described_inside_jit(self) -> None:
        seen: list[jax.ShapeDtypeStruct] = []

        def describe(x: jax.Array) -> jax.Array:
            seen.append(array_to_spec(x))
            return x

        jax.jit(describe)(jnp.zeros((2, 5), dtype=jnp.int32))
        assert seen == [_spec((2, 5), np.int32)]


class TestArrayToSpecStripLeading:
    """``array_to_spec_strip_leading`` describes one element of leading-batched data."""

    def test_strips_the_leading_axis_and_keeps_the_dtype(self) -> None:
        stored = np.zeros((10, 4, 4), dtype=np.float64)
        with _conversion_forbidden():
            assert array_to_spec_strip_leading(stored) == _spec((4, 4), np.float64)

    def test_scalar_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one axis"):
            array_to_spec_strip_leading(np.float32(1.0))


class TestDeviceSpec:
    """``device_spec`` states the spec of the same data after conversion to JAX arrays."""

    def test_narrows_64_bit_dtypes_when_x64_is_off(self) -> None:
        spec = {
            "x": _spec((3,), np.float64),
            "nested": {
                "i": _spec((), np.int64),
                "u": _spec((2,), np.uint64),
                "c": _spec((), np.complex128),
            },
        }
        assert device_spec(spec) == {
            "x": _spec((3,), np.float32),
            "nested": {
                "i": _spec((), np.int32),
                "u": _spec((2,), np.uint32),
                "c": _spec((), np.complex64),
            },
        }

    def test_keeps_dtypes_the_device_holds(self) -> None:
        spec = {
            "half": _spec((2,), np.float16),
            "flag": _spec((), np.bool_),
            "pixels": _spec((8, 8), np.uint8),
        }
        assert device_spec(spec) == spec

    def test_keeps_64_bit_dtypes_when_x64_is_on(self) -> None:
        spec = {"x": _spec((3,), np.float64)}
        with jax.enable_x64(True):
            assert device_spec(spec) == spec

    def test_typed_key_dtypes_pass_through(self) -> None:
        keys = jax.random.split(jax.random.key(0), 3)
        spec = {"keys": _spec((3,), keys.dtype)}
        assert device_spec(spec) == spec

    def test_dtypes_without_a_jax_representation_are_rejected_by_field(self) -> None:
        spec = {"text": _spec((), np.dtype("<U5")), "x": _spec((2,), np.float32)}
        with pytest.raises(TypeError, match=r"\['text'\].*<U5"):
            device_spec(spec)


class TestValidateDeviceDtypes:
    """A declared dtype the device cannot hold as declared is refused, never narrowed silently."""

    def test_spec_the_device_holds_as_declared_passes(self) -> None:
        validate_device_dtypes({"x": _spec((3,), np.float32), "y": _spec((), np.int32)})

    def test_declared_64_bit_dtypes_that_x64_off_would_narrow_are_rejected(self) -> None:
        spec = {"x": _spec((3,), np.float64), "y": _spec((), np.int32), "i": _spec((), np.int64)}
        with pytest.raises(SpecMismatchError) as caught:
            validate_device_dtypes(spec)
        assert len(caught.value.problems) == 2
        message = str(caught.value)
        assert "['x']" in message
        assert "float64" in message
        assert "float32" in message
        assert "['i']" in message
        assert "jax_enable_x64" in message

    def test_64_bit_dtypes_pass_when_x64_is_on(self) -> None:
        with jax.enable_x64(True):
            validate_device_dtypes({"x": _spec((3,), np.float64)})

    def test_dtype_without_a_jax_representation_is_rejected(self) -> None:
        with pytest.raises(SpecMismatchError, match=r"\['text'\].*no JAX array representation"):
            validate_device_dtypes({"text": _spec((), np.dtype("<U5"))})


class TestSpecMismatches:
    """``spec_mismatches`` names every field where two specs differ."""

    def test_identical_specs_have_no_mismatches(self) -> None:
        spec = {"x": _spec((3,), np.float32), "nested": {"y": _spec((), np.int32)}}
        assert spec_mismatches(spec, spec) == ()

    def test_reports_every_field_that_differs(self) -> None:
        expected = {
            "x": _spec((3,), np.float32),
            "y": _spec((), np.int32),
            "gone": _spec((1,), np.float32),
        }
        actual = {
            "x": _spec((4,), np.float32),
            "y": _spec((), np.int64),
            "extra": {"a": _spec((), np.float32), "b": _spec((), np.float32)},
        }
        problems = spec_mismatches(expected, actual)
        joined = "\n".join(problems)
        assert len(problems) == 4
        assert "['x']" in joined
        assert "(4,)" in joined
        assert "(3,)" in joined
        assert "['y']" in joined
        assert "int64" in joined
        assert "int32" in joined
        assert "['gone']: missing" in joined
        # A whole undeclared subtree is one problem, not one per leaf.
        assert "['extra']: unexpected" in joined

    def test_weak_type_alone_is_not_a_mismatch(self) -> None:
        weak = jax.ShapeDtypeStruct((), np.float32, weak_type=True)
        assert spec_mismatches({"x": _spec((), np.float32)}, {"x": weak}) == ()

    def test_container_type_differences_are_reported(self) -> None:
        problems = spec_mismatches([_spec((), np.float32)], (_spec((), np.float32),))
        assert len(problems) == 1
        assert "structure" in problems[0]


class TestBatchLength:
    """``batch_length`` returns the one leading-axis length every leaf shares."""

    def test_returns_the_shared_leading_axis(self) -> None:
        assert batch_length({"x": np.zeros((5, 3)), "y": {"z": jnp.zeros((5,))}}) == 5

    def test_a_batch_without_leaves_has_no_length(self) -> None:
        assert batch_length({}) is None

    def test_leaves_that_disagree_are_named(self) -> None:
        with pytest.raises(SpecMismatchError) as caught:
            batch_length({"x": np.zeros((2, 3)), "y": np.zeros((5,))})
        message = str(caught.value)
        assert "['x']" in message
        assert "2" in message
        assert "['y']" in message
        assert "5" in message

    def test_a_leaf_without_a_leading_axis_is_rejected(self) -> None:
        with pytest.raises(SpecMismatchError, match=r"\['scalar'\]"):
            batch_length({"x": np.zeros((2,)), "scalar": np.float32(1.0)})

    def test_non_array_values_are_rejected_once_per_field(self) -> None:
        with pytest.raises(SpecMismatchError) as caught:
            batch_length({"x": np.zeros((2,)), "text": ["a", "b"]})
        assert len(caught.value.problems) == 1
        assert "['text']" in caught.value.problems[0]
        assert "str" in caught.value.problems[0]


class TestValidateBatch:
    """``validate_batch`` checks structure, per-element shapes, dtypes and batch length."""

    def test_full_and_short_final_batches_pass_without_converting_data(self) -> None:
        full = _batch(4)
        short = _batch(1)
        with _conversion_forbidden():
            validate_batch(full, _ELEMENT_SPEC, batch_size=4)
            validate_batch(short, _ELEMENT_SPEC, batch_size=4)

    def test_jax_batches_pass(self) -> None:
        validate_batch({"x": jnp.zeros((2, 3)), "y": jnp.zeros((2,), jnp.int32)}, _ELEMENT_SPEC)

    def test_batch_larger_than_batch_size_is_rejected(self) -> None:
        with pytest.raises(SpecMismatchError, match="5 records"):
            validate_batch(_batch(5), _ELEMENT_SPEC, batch_size=4)

    def test_empty_batch_is_rejected_when_a_batch_size_is_expected(self) -> None:
        with pytest.raises(SpecMismatchError, match="0 records"):
            validate_batch(_batch(0), _ELEMENT_SPEC, batch_size=4)

    def test_per_element_shape_mismatch_names_the_field(self) -> None:
        with pytest.raises(SpecMismatchError, match=r"\['x'\].*\(4,\).*\(3,\)"):
            validate_batch(_batch(2, x_shape=(4,)), _ELEMENT_SPEC)

    def test_dtype_is_compared_without_coercion(self) -> None:
        batch = _batch(2, x_dtype=np.float64)
        with pytest.raises(SpecMismatchError, match=r"\['x'\].*float64.*float32"):
            validate_batch(batch, _ELEMENT_SPEC)
        assert batch["x"].dtype == np.float64

    def test_missing_and_undeclared_fields_are_named(self) -> None:
        batch = {"x": np.zeros((2, 3), np.float32), "z": np.zeros((2,), np.float32)}
        with pytest.raises(SpecMismatchError) as caught:
            validate_batch(batch, _ELEMENT_SPEC)
        joined = "\n".join(caught.value.problems)
        assert "['y']: missing" in joined
        assert "['z']: unexpected" in joined

    def test_all_problems_are_reported_together(self) -> None:
        batch = {
            "x": np.zeros((2, 4), np.float64),
            "y": np.zeros((5,), np.int32),
            "z": ["a", "b"],
        }
        with pytest.raises(SpecMismatchError) as caught:
            validate_batch(batch, _ELEMENT_SPEC, batch_size=2)
        assert len(caught.value.problems) >= 4

    def test_error_is_a_value_error_carrying_the_problems(self) -> None:
        with pytest.raises(ValueError) as caught:
            validate_batch(_batch(2, x_shape=(4,)), _ELEMENT_SPEC)
        assert isinstance(caught.value, SpecMismatchError)
        assert isinstance(caught.value.problems, tuple)

    def test_a_scalar_leaf_is_rejected_even_when_the_structure_matches(self) -> None:
        batch = {"x": np.zeros((2, 3), np.float32), "y": np.int32(0)}
        with pytest.raises(SpecMismatchError, match=r"\['y'\].*no leading batch axis"):
            validate_batch(batch, _ELEMENT_SPEC)

    def test_runs_at_trace_time_and_adds_nothing_to_the_compiled_graph(self) -> None:
        def checked(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
            validate_batch(batch, _ELEMENT_SPEC, batch_size=4)
            return batch

        jaxpr = jax.make_jaxpr(checked)({"x": jnp.zeros((4, 3)), "y": jnp.zeros((4,), jnp.int32)})
        assert jaxpr.eqns == []

    def test_mismatch_inside_jit_raises_while_tracing(self) -> None:
        def checked(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
            validate_batch(batch, _ELEMENT_SPEC)
            return batch

        with pytest.raises(SpecMismatchError):
            jax.jit(checked)({"x": jnp.zeros((2, 4)), "y": jnp.zeros((2,), jnp.int32)})

    def test_memory_source_batches_validate_inside_nnx_jit(self) -> None:
        source = MemorySource(
            MemorySourceConfig(),
            {"x": np.random.rand(8, 3), "y": np.arange(8)},
            rngs=nnx.Rngs(0),
        )

        @nnx.jit
        def fetch(src: MemorySource) -> dict[str, jax.Array]:
            batch = src.get_batch_at(0, 4)
            validate_batch(batch, src.element_spec(), batch_size=4)
            return batch

        batch = fetch(source)
        assert batch["x"].dtype == np.float32
        assert batch["y"].dtype == np.int32
