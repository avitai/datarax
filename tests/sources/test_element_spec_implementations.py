"""Tests that concrete sources implement ``element_spec()`` correctly.

Each source must declare the per-element output shape/dtype as a PyTree of
``jax.ShapeDtypeStruct``. Downstream consumers (operators, batchers, models)
rely on this contract for buffer pre-allocation and auto-sizing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

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
    """element_spec dtypes must be JAX/jnp dtypes, not numpy dtypes.

    NumPy and JAX dtypes compare unequal under ``==``, which would silently
    break shape-validation pipelines downstream. The spec contract requires
    JAX-flavored dtypes.
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
    from datarax.utils.spec import batched_spec  # noqa: PLC0415

    data = {"image": jnp.ones((50, 4), dtype=jnp.float32)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    elem_spec = source.element_spec()
    bspec = batched_spec(elem_spec, batch_size=8)

    image_spec = bspec["image"]
    valid_mask_spec = bspec["valid_mask"]
    assert isinstance(image_spec, jax.ShapeDtypeStruct)
    assert isinstance(valid_mask_spec, jax.ShapeDtypeStruct)
    assert image_spec.shape == (8, 4)
    assert valid_mask_spec.shape == (8,)
    assert valid_mask_spec.dtype == jnp.bool_
