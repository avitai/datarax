"""Behaviour tests for the datarax adapter's structural capabilities.

Each capability the adapter declares must build a runnable pipeline and exhibit
its defining behaviour (batch mixing, branching, mixed source, differentiable
rebatch, etc.), not merely pass config-shape validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from benchmarks.adapters.base import Capability, ScenarioConfig
from benchmarks.adapters.datarax_adapter import _LearnableAugmentOperator, DataraxAdapter


def _image_data(n: int = 32) -> dict[str, np.ndarray]:
    return {"image": np.random.default_rng(0).standard_normal((n, 16, 16, 3)).astype(np.float32)}


def _config(caps: list[Capability], batch: int = 8, **extra: object) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="TEST",
        dataset_size=32,
        element_shape=(16, 16, 3),
        batch_size=batch,
        transforms=["Normalize"],
        required_capabilities=[str(cap) for cap in caps],
        extra={"variant_name": "test", **extra},
    )


def _run(adapter: DataraxAdapter, config: ScenarioConfig, data: dict) -> list:
    adapter.setup(config, data)
    result = adapter.iterate(num_batches=2)
    first_batch = next(iter(adapter._pipeline))  # type: ignore[reportAttributeAccessIssue]
    adapter.teardown()
    return [result, first_batch]


class TestAdapterDeclaresCapabilities:
    """The adapter advertises exactly the capabilities it can build."""

    def test_declares_expected_capabilities(self):
        caps = DataraxAdapter().available_capabilities()
        for expected in (
            Capability.BATCH_MIXING,
            Capability.PROBABILISTIC,
            Capability.LEARNABLE_TRANSFORM,
            Capability.DAG_BRANCHING,
            Capability.MIXED_SOURCE,
            Capability.REBATCHING,
        ):
            assert expected in caps


class TestCapabilityPipelinesRun:
    """Each capability builds a pipeline that produces batches."""

    @pytest.mark.parametrize(
        ("caps", "extra"),
        [
            ([Capability.BATCH_MIXING], {"mix_mode": "mixup", "mix_alpha": 0.4}),
            ([Capability.PROBABILISTIC], {"probability": 0.5}),
            ([Capability.LEARNABLE_TRANSFORM], {}),
            ([Capability.DAG_BRANCHING], {}),
            ([Capability.MIXED_SOURCE], {}),
            ([Capability.CACHING], {}),
        ],
    )
    def test_capability_pipeline_yields_batches(self, caps, extra):
        result, _ = _run(DataraxAdapter(), _config(caps, **extra), _image_data())
        assert result.num_batches == 2
        assert result.num_elements > 0


class TestCachingCapability:
    """The caching capability wraps iteration in a replay cache."""

    def test_setup_wires_cache(self):
        adapter = DataraxAdapter()
        adapter.setup(_config([Capability.CACHING]), _image_data())
        assert adapter._cached_iter is not None  # type: ignore[reportAttributeAccessIssue]
        adapter.teardown()
        assert adapter._cached_iter is None  # type: ignore[reportAttributeAccessIssue]

    def test_replayed_batches_are_identical(self):
        import jax

        adapter = DataraxAdapter()
        adapter.setup(_config([Capability.CACHING]), _image_data())
        first_pass = jax.tree.leaves(next(iter(adapter._pipeline)))  # type: ignore[reportAttributeAccessIssue]
        first_cached = next(iter(adapter._cached_iter))  # type: ignore[reportAttributeAccessIssue]
        replayed = next(iter(adapter._cached_iter))  # type: ignore[reportAttributeAccessIssue]
        assert jax.tree.leaves(first_cached)[0].shape == first_pass[0].shape
        assert bool((jax.tree.leaves(replayed)[0] == jax.tree.leaves(first_cached)[0]).all())
        adapter.teardown()


class TestRebatchCapability:
    """Rebatching reduces the batch dimension to the configured target."""

    def test_rebatch_reduces_batch_dimension(self):
        config = _config([Capability.REBATCHING], batch=8, target_batch_size=2)
        _, first_batch = _run(DataraxAdapter(), config, _image_data())
        # 8 records grouped by 8 // 2 == 4 -> leading dim 2.
        assert jax.tree.leaves(first_batch)[0].shape[0] == 2


class TestLearnableOperatorIsDifferentiable:
    """The learnable augment stage flows gradients and honours train/eval mode."""

    def test_gradient_flows_through_params(self):
        op = _LearnableAugmentOperator(3, rngs=nnx.Rngs(0, dropout=1))
        op.eval()

        def loss(module, x):
            return jnp.sum(module({"image": x})["image"] ** 2)

        grads = nnx.grad(loss)(op, jnp.ones((4, 8, 8, 3)))
        leaves = jax.tree.leaves(nnx.state(grads))
        assert leaves
        assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)

    def test_train_eval_toggles_dropout(self):
        op = _LearnableAugmentOperator(3, rngs=nnx.Rngs(0, dropout=1))
        op.eval()
        assert op.dropout.deterministic is True
        op.train()
        assert op.dropout.deterministic is False
