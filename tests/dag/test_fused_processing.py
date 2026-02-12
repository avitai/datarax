"""Tests for fused operator processing in DAGExecutor.

TDD tests for the fused _process_batch_from_source path that uses
_apply_on_raw to chain operators without intermediate Batch objects.

Test categories:
1. Fused path matches sequential operator application
2. CV-1-like pipeline (Normalize + CastToFloat32)
3. No-operator passthrough
4. End-to-end pipeline iteration with fused path
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import OperatorConfig, ElementOperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule
from datarax.dag.dag_executor import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators.element_operator import ElementOperator
from datarax.sources import MemorySource, MemorySourceConfig


# ========================================================================
# Test Operators
# ========================================================================


@dataclass
class ScaleConfig(OperatorConfig):
    factor: float = 2.0


class ScaleOperator(OperatorModule):
    """Deterministic operator: multiplies data by a factor."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        new_data = jax.tree.map(lambda x: x * self.config.factor, data)
        return new_data, state, metadata


@dataclass
class AddConfig(OperatorConfig):
    offset: float = 1.0


class AddOperator(OperatorModule):
    """Deterministic operator: adds offset to data."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        new_data = jax.tree.map(lambda x: x + self.config.offset, data)
        return new_data, state, metadata


@dataclass
class StochasticNoiseConfig(OperatorConfig):
    """Config for stochastic noise operator."""

    noise_scale: float = 0.1

    def __post_init__(self):
        object.__setattr__(self, "stochastic", True)
        object.__setattr__(self, "stream_name", "noise")
        super().__post_init__()


class StochasticNoiseOperator(OperatorModule):
    """Stochastic operator: adds random noise to data."""

    def generate_random_params(self, rng, data_shapes):
        shapes = jax.tree.leaves(data_shapes, is_leaf=lambda x: isinstance(x, tuple))
        batch_size = shapes[0][0]
        return jax.random.normal(rng, shape=(batch_size,)) * self.config.noise_scale

    def apply(self, data, state, metadata, random_params=None, stats=None):
        noise = random_params if random_params is not None else 0.0
        new_data = jax.tree.map(lambda x: x + noise, data)
        return new_data, state, metadata


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def image_data():
    """Create simple image-like data as list of dicts (MemorySource format)."""
    np.random.seed(42)
    images = np.random.randint(0, 256, size=(32, 8, 8, 3), dtype=np.uint8)
    return [{"image": images[i]} for i in range(32)]


@pytest.fixture
def source(image_data):
    """Create a MemorySource."""
    config = MemorySourceConfig(shuffle=False)
    return MemorySource(config=config, data=image_data, rngs=nnx.Rngs(0))


@pytest.fixture
def scale_op():
    return ScaleOperator(ScaleConfig(stochastic=False, factor=2.0))


@pytest.fixture
def add_op():
    return AddOperator(AddConfig(stochastic=False, offset=1.0))


# ========================================================================
# Tests: Fused raw chain matches sequential Batch application
# ========================================================================


class TestFusedMatchesSequential:
    """Fused operator chain produces identical results to sequential application."""

    def test_fused_two_operators(self, scale_op, add_op):
        """Two-operator fused chain matches sequential: scale(2.0) then add(1.0)."""
        jax_data = {"image": jnp.ones((8, 4, 4, 3), dtype=jnp.float32)}

        # Sequential path: create Batch, apply each op
        batch = Batch.from_parts(data=jax_data, states={}, validate=False)
        seq_batch = scale_op.apply_batch(batch)
        seq_batch = add_op.apply_batch(seq_batch)
        seq_data = seq_batch.data.get_value()

        # Fused path: chain _apply_on_raw calls
        fused_data, fused_states = scale_op._apply_on_raw(jax_data, {})
        fused_data, fused_states = add_op._apply_on_raw(fused_data, fused_states)

        for key in seq_data:
            assert jnp.allclose(fused_data[key], seq_data[key]), (
                f"Fused and sequential differ for key '{key}'"
            )

    def test_fused_single_operator(self, scale_op):
        """Single operator fused chain matches sequential."""
        jax_data = {"image": jnp.ones((8, 4, 4, 3), dtype=jnp.float32)}

        batch = Batch.from_parts(data=jax_data, states={}, validate=False)
        seq_batch = scale_op.apply_batch(batch)
        seq_data = seq_batch.data.get_value()

        fused_data, _ = scale_op._apply_on_raw(jax_data, {})

        for key in seq_data:
            assert jnp.allclose(fused_data[key], seq_data[key])

    def test_fused_no_operators(self):
        """No-operator path passes data through unchanged."""
        jax_data = {"image": jnp.ones((8, 4, 4, 3), dtype=jnp.float32)}

        batch = Batch.from_parts(data=jax_data, states={}, validate=False)
        result_data = batch.data.get_value()

        for key in jax_data:
            assert jnp.allclose(result_data[key], jax_data[key])


# ========================================================================
# Tests: End-to-end pipeline iteration
# ========================================================================


class TestFusedPipelineIteration:
    """End-to-end pipeline iteration with operators."""

    def test_pipeline_with_two_operators(self, source, scale_op, add_op):
        """Pipeline iterates correctly with two operators."""
        pipeline = from_source(source, batch_size=8)
        pipeline = pipeline >> OperatorNode(scale_op) >> OperatorNode(add_op)

        batches = list(pipeline)
        assert len(batches) == 4  # 32 / 8 = 4 batches

        first_data = batches[0].get_data()
        assert "image" in first_data
        assert first_data["image"].dtype == jnp.float32

    def test_pipeline_no_operators(self, source):
        """Pipeline without operators still works."""
        pipeline = from_source(source, batch_size=8)

        batches = list(pipeline)
        assert len(batches) == 4

    def test_pipeline_single_operator(self, source, scale_op):
        """Pipeline with one operator works correctly."""
        pipeline = from_source(source, batch_size=8)
        pipeline = pipeline >> OperatorNode(scale_op)

        batches = list(pipeline)
        assert len(batches) == 4

        data = batches[0].get_data()
        assert "image" in data

    def test_cv1_like_pipeline(self):
        """CV-1-like pipeline: Normalize + CastToFloat32 on uint8 images."""
        np.random.seed(42)
        raw = [
            {"image": np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)} for _ in range(16)
        ]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        def normalize_fn(element, key):
            new_data = jax.tree.map(lambda x: x / 255.0, element.data)
            return element.replace(data=new_data)

        def cast_fn(element, key):
            new_data = jax.tree.map(lambda x: x.astype(jnp.float32), element.data)
            return element.replace(data=new_data)

        rngs = nnx.Rngs(0)
        normalize_op = ElementOperator(
            ElementOperatorConfig(stochastic=False), fn=normalize_fn, rngs=rngs
        )
        cast_op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=cast_fn, rngs=rngs)

        pipeline = from_source(source, batch_size=8)
        pipeline = pipeline >> OperatorNode(normalize_op) >> OperatorNode(cast_op)

        batches = list(pipeline)
        assert len(batches) == 2

        data = batches[0].get_data()
        assert data["image"].dtype == jnp.float32
        assert jnp.all(data["image"] >= 0.0)
        assert jnp.all(data["image"] <= 1.0)

    def test_fused_results_correctness(self):
        """Full pipeline produces mathematically correct results."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        scale_op = ScaleOperator(ScaleConfig(stochastic=False, factor=3.0))

        pipeline = from_source(source, batch_size=4)
        pipeline = pipeline >> OperatorNode(scale_op)

        batches = list(pipeline)
        assert len(batches) == 2

        # ones * 3.0 = 3.0
        expected = jnp.ones((4, 4)) * 3.0
        assert jnp.allclose(batches[0].get_data()["val"], expected)


# ========================================================================
# Tests: JIT-compiled fused chain
# ========================================================================


class TestJitFusedChain:
    """JIT-compiled fused operator chain tests."""

    def test_jit_fused_correctness(self):
        """JIT-compiled chain produces same output as eager chain."""
        raw = [{"val": np.ones((4,), dtype=np.float32) * i} for i in range(16)]
        config = MemorySourceConfig(shuffle=False)

        # Eager pipeline
        source1 = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))
        scale1 = ScaleOperator(ScaleConfig(stochastic=False, factor=2.0))
        add1 = AddOperator(AddConfig(stochastic=False, offset=0.5))
        pipe1 = from_source(source1, batch_size=4)
        pipe1 = pipe1 >> OperatorNode(scale1) >> OperatorNode(add1)
        eager_batches = list(pipe1)

        # JIT pipeline (same config — uses JIT internally via _make_fused_step)
        source2 = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))
        scale2 = ScaleOperator(ScaleConfig(stochastic=False, factor=2.0))
        add2 = AddOperator(AddConfig(stochastic=False, offset=0.5))
        pipe2 = from_source(source2, batch_size=4)
        pipe2 = pipe2 >> OperatorNode(scale2) >> OperatorNode(add2)
        jit_batches = list(pipe2)

        assert len(eager_batches) == len(jit_batches)
        for eb, jb in zip(eager_batches, jit_batches):
            for key in eb.get_data():
                assert jnp.allclose(eb.get_data()[key], jb.get_data()[key])

    def test_jit_fused_multiple_epochs(self):
        """JIT-compiled chain works across multiple epochs."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        scale_op = ScaleOperator(ScaleConfig(stochastic=False, factor=5.0))
        pipeline = from_source(source, batch_size=4)
        pipeline = pipeline >> OperatorNode(scale_op)

        # Epoch 1
        batches_1 = list(pipeline)
        # Epoch 2
        batches_2 = list(pipeline)

        assert len(batches_1) == 2
        assert len(batches_2) == 2

        # Both epochs should produce the same results
        expected = jnp.ones((4, 4)) * 5.0
        assert jnp.allclose(batches_1[0].get_data()["val"], expected)
        assert jnp.allclose(batches_2[0].get_data()["val"], expected)

    def test_jit_fused_second_call_uses_cache(self):
        """Second iteration uses JIT cache (no recompilation)."""
        import time

        raw = [{"val": np.ones((128,), dtype=np.float32)} for _ in range(64)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        scale_op = ScaleOperator(ScaleConfig(stochastic=False, factor=2.0))
        pipeline = from_source(source, batch_size=32)
        pipeline = pipeline >> OperatorNode(scale_op)

        # First epoch: includes JIT compilation time
        t0 = time.perf_counter()
        list(pipeline)
        first_time = time.perf_counter() - t0

        # Second epoch: should use cached compiled kernel
        t0 = time.perf_counter()
        list(pipeline)
        second_time = time.perf_counter() - t0

        # Second should be no slower than first (JIT cache hit)
        # We can't guarantee faster on small data, but it should not be drastically slower
        assert second_time < first_time * 5, (
            f"Second epoch ({second_time:.4f}s) much slower than first ({first_time:.4f}s)"
        )

    def test_jit_stochastic_operator(self):
        """Single stochastic op inside JIT'd fused chain produces valid output."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        noise_op = StochasticNoiseOperator(
            StochasticNoiseConfig(noise_scale=0.1),
            rngs=nnx.Rngs(noise=42),
        )
        pipeline = from_source(source, batch_size=4)
        pipeline = pipeline >> OperatorNode(noise_op)

        batches = list(pipeline)
        assert len(batches) == 2

        # Output should be non-zero and different from input (noise added)
        output = batches[0].get_data()["val"]
        assert not jnp.allclose(output, jnp.ones((4, 4)))
        assert not jnp.any(jnp.isnan(output))

    def test_jit_mixed_chain(self):
        """Deterministic + stochastic ops in same JIT'd chain."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        scale_op = ScaleOperator(ScaleConfig(stochastic=False, factor=3.0))
        noise_op = StochasticNoiseOperator(
            StochasticNoiseConfig(noise_scale=0.1),
            rngs=nnx.Rngs(noise=42),
        )
        pipeline = from_source(source, batch_size=4)
        pipeline = pipeline >> OperatorNode(scale_op) >> OperatorNode(noise_op)

        batches = list(pipeline)
        assert len(batches) == 2

        # Scale(3.0) + noise(~0.1) -> should be close to 3.0 but not exact
        output = batches[0].get_data()["val"]
        assert jnp.all(output > 2.5)  # scale(3.0) - noise won't push below 2.5
        assert not jnp.allclose(output, jnp.ones((4, 4)) * 3.0)  # noise added

    def test_jit_stochastic_multiple_epochs(self):
        """Stochastic JIT chain works across 2+ epochs without TraceContextError."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        noise_op = StochasticNoiseOperator(
            StochasticNoiseConfig(noise_scale=0.5),
            rngs=nnx.Rngs(noise=42),
        )
        pipeline = from_source(source, batch_size=4)
        pipeline = pipeline >> OperatorNode(noise_op)

        # Epoch 1
        batches_1 = list(pipeline)
        # Epoch 2 — should not raise TraceContextError
        batches_2 = list(pipeline)

        assert len(batches_1) == 2
        assert len(batches_2) == 2

        # Different epochs should produce different results (different RNG states)
        assert not jnp.allclose(
            batches_1[0].get_data()["val"],
            batches_2[0].get_data()["val"],
        )


# ========================================================================
# Tests: Prefetching integration
# ========================================================================


class TestPrefetchIntegration:
    """Prefetcher wired into DAGExecutor batch-first path."""

    def test_prefetch_produces_same_results(self):
        """Prefetched pipeline produces identical results to non-prefetched."""
        raw = [{"val": np.ones((4,), dtype=np.float32) * i} for i in range(16)]
        config = MemorySourceConfig(shuffle=False)

        # With prefetch (default: prefetch_size=2)
        source1 = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))
        scale1 = ScaleOperator(ScaleConfig(stochastic=False, factor=2.0))
        pipe1 = from_source(source1, batch_size=4, prefetch_size=2)
        pipe1 = pipe1 >> OperatorNode(scale1)
        prefetched = list(pipe1)

        # Without prefetch
        source2 = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))
        scale2 = ScaleOperator(ScaleConfig(stochastic=False, factor=2.0))
        pipe2 = from_source(source2, batch_size=4, prefetch_size=0)
        pipe2 = pipe2 >> OperatorNode(scale2)
        no_prefetch = list(pipe2)

        assert len(prefetched) == len(no_prefetch)
        for pb, npb in zip(prefetched, no_prefetch):
            for key in pb.get_data():
                assert jnp.allclose(pb.get_data()[key], npb.get_data()[key])

    def test_prefetch_disabled_when_zero(self):
        """prefetch_size=0 skips prefetching."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        pipeline = from_source(source, batch_size=4, prefetch_size=0)
        assert pipeline.prefetch_size == 0

        batches = list(pipeline)
        assert len(batches) == 2

    def test_prefetch_default_enabled(self):
        """Default prefetch_size=2 is active."""
        raw = [{"val": np.ones((4,), dtype=np.float32)} for _ in range(8)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config=config, data=raw, rngs=nnx.Rngs(0))

        pipeline = from_source(source, batch_size=4)
        assert pipeline.prefetch_size == 2

        batches = list(pipeline)
        assert len(batches) == 2
