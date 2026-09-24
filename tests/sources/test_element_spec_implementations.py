"""Tests that concrete sources implement ``element_spec()`` correctly.

Each source must declare the per-element output shape/dtype as a PyTree of
``jax.ShapeDtypeStruct``. Downstream consumers (operators, batchers, models)
rely on this contract for buffer pre-allocation and auto-sizing.
"""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.spec import batched_spec, validate_batch
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def test_memory_source_element_spec_dict_data() -> None:
    """A MemorySource over a dict of arrays returns a matching dict of ShapeDtypeStructs.

    The leading dataset-size dimension is stripped from each array; the spec
    describes a single emitted element.
    """
    data = {
        "image": jnp.ones((100, 28, 28, 1), dtype=jnp.float32),
        "label": jnp.arange(100, dtype=jnp.int32),
    }
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    spec = source.element_spec()

    assert isinstance(spec, dict)
    assert set(spec.keys()) == {"image", "label"}

    assert isinstance(spec["image"], jax.ShapeDtypeStruct)
    assert spec["image"].shape == (28, 28, 1)
    assert spec["image"].dtype == jnp.float32

    assert isinstance(spec["label"], jax.ShapeDtypeStruct)
    assert spec["label"].shape == ()
    assert spec["label"].dtype == jnp.int32


def test_memory_source_element_spec_uses_jax_dtypes() -> None:
    """element_spec dtypes compare equal to JAX dtypes.

    ``jax.ShapeDtypeStruct`` stores its dtype as a ``numpy.dtype``, which
    compares equal to the matching ``jnp`` scalar type, so the spec can be
    checked directly against JAX-traced shapes downstream.
    """
    data = {"x": np.ones((10, 5), dtype=np.float64)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    spec = source.element_spec()

    # JAX's default dtype for float64 input on CPU is float32 (x64 disabled by
    # default); the source should canonicalize to JAX's view of the dtype.
    assert spec["x"].dtype in (jnp.float32, jnp.float64)
    assert spec["x"].shape == (5,)


def test_memory_source_element_spec_preserves_pipeline_chain() -> None:
    """A MemorySource's spec composes with the default operator/batcher chain."""
    data = {"image": jnp.ones((50, 4), dtype=jnp.float32)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    elem_spec = source.element_spec()
    bspec = batched_spec(elem_spec, batch_size=8)

    assert set(bspec) == {"image"}
    image_spec = bspec["image"]
    assert isinstance(image_spec, jax.ShapeDtypeStruct)
    assert image_spec.shape == (8, 4)


def test_memory_source_element_spec_describes_get_batch_at_output() -> None:
    """The declared spec is exactly what the Pipeline-facing ``get_batch_at`` emits.

    ``get_batch_at`` converts host storage to JAX arrays, so with x64 off a
    float64 or int64 host array is emitted, and declared, as float32 or int32.
    """
    data = {
        "x": np.random.rand(10, 5),
        "y": np.arange(10, dtype=np.int64),
        "image": np.zeros((10, 2, 2), dtype=np.uint8),
    }
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    spec = source.element_spec()

    assert spec == {
        "x": jax.ShapeDtypeStruct((5,), jnp.float32),
        "y": jax.ShapeDtypeStruct((), jnp.int32),
        "image": jax.ShapeDtypeStruct((2, 2), jnp.uint8),
    }
    validate_batch(source.get_batch_at(0, 4), spec, batch_size=4)


def test_memory_source_element_spec_reads_storage_metadata_without_converting_it() -> None:
    """Deriving the spec never copies the stored dataset to the device."""
    source = MemorySource(MemorySourceConfig(), {"x": np.random.rand(10, 5)}, rngs=nnx.Rngs(0))
    refuse = AssertionError("storage was converted to derive element_spec")

    with (
        patch("jax.numpy.asarray", side_effect=refuse),
        patch("jax.numpy.array", side_effect=refuse),
    ):
        spec = source.element_spec()

    assert spec["x"].shape == (5,)


def test_memory_source_list_mode_element_spec_is_the_device_view_of_element_zero() -> None:
    """List-mode sources declare element zero as the device holds it."""
    data = [{"x": np.ones((3,), dtype=np.float64), "y": 1} for _ in range(4)]
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    assert source.element_spec() == {
        "x": jax.ShapeDtypeStruct((3,), jnp.float32),
        "y": jax.ShapeDtypeStruct((), jnp.int32),
    }
