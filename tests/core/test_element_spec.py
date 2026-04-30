"""Tests for the element_spec / output_spec / batch_spec / index_spec contracts.

Each tier of the Three-Tier Architecture (DataSourceModule / OperatorModule /
BatcherModule / SamplerModule) declares its output PyTree shape/dtype as
``jax.ShapeDtypeStruct`` leaves, so downstream consumers can pre-allocate
buffers, auto-size learnable layers, and statically validate operator chains.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.batcher import BatcherModule
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule


@dataclass(frozen=True)
class _MinimalSourceConfig(StructuralConfig):
    pass


class _UnimplementedSource(DataSourceModule):
    """Subclass without overriding element_spec — should raise."""


def test_data_source_module_requires_element_spec() -> None:
    """A DataSourceModule subclass that does not override element_spec must raise.

    The contract is mandatory: data sources must declare their output shape so
    downstream operators can pre-allocate buffers and auto-size layers.
    """
    src = _UnimplementedSource(_MinimalSourceConfig(stochastic=False))
    with pytest.raises(NotImplementedError, match="element_spec"):
        src.element_spec()


@dataclass(frozen=True)
class _IdentityOperatorConfig(OperatorConfig):
    pass


class _IdentityOperator(OperatorModule):
    """Operator that does not change shape — should pass spec through unchanged."""

    def apply(self, data, state, metadata, random_params=None, stats=None):  # type: ignore[override]
        return data, state, metadata


def test_operator_module_output_spec_default_passthrough() -> None:
    """Operators default to passthrough output_spec — input_spec returns unchanged.

    Most operators (normalization, additive noise, simple element-wise transforms)
    do not change shape; making passthrough the default avoids boilerplate while
    still letting shape-changing operators (Resize, Crop) override.
    """
    op = _IdentityOperator(_IdentityOperatorConfig(stochastic=False), rngs=nnx.Rngs(0))
    input_spec = {
        "image": jax.ShapeDtypeStruct(shape=(28, 28, 1), dtype=jnp.float32),
        "label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
    }
    output_spec = op.output_spec(input_spec)

    assert output_spec is input_spec or output_spec == input_spec


@dataclass(frozen=True)
class _MinimalBatcherConfig(StructuralConfig):
    pass


class _MinimalBatcher(BatcherModule):
    """Concrete batcher that uses the base class's default batch_spec."""

    def process(self, elements, *args, batch_size, drop_remainder=False, **kwargs):  # type: ignore[override]
        del elements, args, drop_remainder, kwargs
        return []


def test_batcher_module_batch_spec_adds_leading_dim_and_valid_mask() -> None:
    """Batchers add a leading (batch_size,) dim to every leaf and a top-level valid_mask.

    The valid_mask leaf is what allows mask-weighted loss to ignore padded
    positions in end-of-epoch partial batches without the JIT recompilation
    that variable batch shapes would force.
    """
    batcher = _MinimalBatcher(_MinimalBatcherConfig(stochastic=False))
    element_spec = {
        "image": jax.ShapeDtypeStruct(shape=(28, 28, 1), dtype=jnp.float32),
        "label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
    }
    batch_spec = batcher.batch_spec(element_spec, batch_size=32)

    assert "valid_mask" in batch_spec
    assert batch_spec["valid_mask"].shape == (32,)
    assert batch_spec["valid_mask"].dtype == jnp.bool_

    assert batch_spec["image"].shape == (32, 28, 28, 1)
    assert batch_spec["image"].dtype == jnp.float32
    assert batch_spec["label"].shape == (32,)
    assert batch_spec["label"].dtype == jnp.int32


@dataclass(frozen=True)
class _MinimalSamplerConfig(StructuralConfig):
    pass


class _MinimalSampler(SamplerModule):
    """Concrete sampler that uses the base class's default index_spec."""


def test_sampler_module_index_spec_returns_int_array() -> None:
    """Samplers default to emitting a scalar int32 index per call.

    Specialized samplers (SlidingWindowSampler, BufferSampler) override
    index_spec to declare windowed or vectorized index shapes.
    """
    sampler = _MinimalSampler(_MinimalSamplerConfig(stochastic=False))
    spec = sampler.index_spec()

    assert isinstance(spec, jax.ShapeDtypeStruct)
    assert spec.shape == ()
    assert spec.dtype == jnp.int32
