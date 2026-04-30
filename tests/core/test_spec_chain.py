"""End-to-end test that ``element_spec`` flows through a source → operator → batcher chain.

Verifies the contract: a source's per-element spec, propagated through any
shape-preserving operators via ``output_spec``, must produce a valid batch
spec when handed to a batcher's ``batch_spec``. Downstream consumers
(models, pre-allocated buffers, JIT-aware code) rely on this chain producing
JAX-friendly ``ShapeDtypeStruct`` leaves with consistent shapes and dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.batcher import BatcherModule
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.operator import OperatorModule
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


@dataclass(frozen=True)
class _PassthroughOperatorConfig(OperatorConfig):
    pass


class _PassthroughOperator(OperatorModule):
    """Operator that doesn't change shape — uses base class output_spec default."""

    def apply(self, data, state, metadata, random_params=None, stats=None):  # type: ignore[override]
        return data, state, metadata


@dataclass(frozen=True)
class _MinimalBatcherConfig(StructuralConfig):
    pass


class _MinimalBatcher(BatcherModule):
    """Concrete batcher that uses the base class's default batch_spec."""

    def process(self, elements, *args, batch_size, drop_remainder=False, **kwargs):  # type: ignore[override]
        del elements, args, drop_remainder, kwargs
        return []


def test_spec_chain_source_to_operator_to_batcher() -> None:
    """A complete spec chain produces a JAX-friendly batched spec.

    source: dict-of-arrays MemorySource with image (28×28×1) + label (scalar)
    op1: passthrough (default output_spec inherits identity)
    op2: passthrough
    batcher: default batch_spec adds (batch_size,) leading dim + valid_mask
    """
    data = {
        "image": jnp.ones((100, 28, 28, 1), dtype=jnp.float32),
        "label": jnp.arange(100, dtype=jnp.int32),
    }
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))
    op1 = _PassthroughOperator(_PassthroughOperatorConfig(stochastic=False), rngs=nnx.Rngs(0))
    op2 = _PassthroughOperator(_PassthroughOperatorConfig(stochastic=False), rngs=nnx.Rngs(1))
    batcher = _MinimalBatcher(_MinimalBatcherConfig(stochastic=False))

    elem_spec = source.element_spec()
    after_op1 = op1.output_spec(elem_spec)
    after_op2 = op2.output_spec(after_op1)
    batch_spec = batcher.batch_spec(after_op2, batch_size=8)

    # Image: per-element (28,28,1) → batched (8,28,28,1)
    assert isinstance(batch_spec["image"], jax.ShapeDtypeStruct)
    assert batch_spec["image"].shape == (8, 28, 28, 1)
    assert batch_spec["image"].dtype == jnp.float32

    # Label: per-element () → batched (8,)
    assert isinstance(batch_spec["label"], jax.ShapeDtypeStruct)
    assert batch_spec["label"].shape == (8,)
    assert batch_spec["label"].dtype == jnp.int32

    # Validity mask: present, correctly typed
    assert isinstance(batch_spec["valid_mask"], jax.ShapeDtypeStruct)
    assert batch_spec["valid_mask"].shape == (8,)
    assert batch_spec["valid_mask"].dtype == jnp.bool_


def test_spec_chain_passthrough_operator_preserves_dtype_and_shape() -> None:
    """A chain of passthrough operators produces specs identical to source."""
    data = {"x": jnp.ones((5, 3), dtype=jnp.float32)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))
    op = _PassthroughOperator(_PassthroughOperatorConfig(stochastic=False), rngs=nnx.Rngs(0))

    elem_spec = source.element_spec()
    after = op.output_spec(elem_spec)

    assert after["x"].shape == elem_spec["x"].shape
    assert after["x"].dtype == elem_spec["x"].dtype
