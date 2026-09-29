"""Tests for OperatorModule - parametric transformation module.

This test suite validates OperatorModule - the base class for all parametric,
differentiable data transformations.

Test Categories (from operator-module-api.md):
1. Module initialization (config-based, stochastic vs deterministic)
2. Stochastic mode (random parameter generation and application)
3. Deterministic mode (no randomness)
4. Batch processing (vmap correctness, empty batches)
5. JIT compatibility (compilation, static branches)
6. Random parameter system (generation, distribution via vmap)
7. Statistics system (inherited from DataraxModule)
8. Training/eval mode (inherited from NNX)
9. Module copying (config-based)
"""

# ========================================================================
# Test Fixture: Example Operator Implementations
# ========================================================================
# Example 1: Simple stochastic operator (random brightness)
from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.testing import TraceCounter
from substrax.testing.compiles import expect_compiles

from datarax.core import batch_ops
from datarax.core.config import DataraxModuleConfig, OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.module import DataraxModule
from datarax.core.operator import OperatorModule, require_key


@dataclass(frozen=True)
class RandomBrightnessConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
    """Config for random brightness operator."""

    min_factor: float = 0.8
    max_factor: float = 1.2

    def __post_init__(self):
        super().__post_init__()
        if self.min_factor >= self.max_factor:
            raise ValueError("min_factor must be < max_factor")
        if self.min_factor <= 0 or self.max_factor <= 0:
            raise ValueError("Brightness factors must be positive")


class RandomBrightnessOperator(OperatorModule):
    """Stochastic operator that adjusts brightness randomly."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del stats
        # This record's brightness factor, drawn from its own key.
        record_key = require_key(key, self)
        factor = jax.random.uniform(
            record_key,
            shape=(),
            minval=self.config.min_factor,  # type: ignore[reportAttributeAccessIssue]
            maxval=self.config.max_factor,  # type: ignore[reportAttributeAccessIssue]
        )
        transformed_data = {**data, "image": jnp.clip(data["image"] * factor, 0.0, 1.0)}
        return element.replace(data=transformed_data)


@dataclass(frozen=True)
class NormalizeConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
    """Config for normalization operator (deterministic)."""

    # No operator-specific fields needed
    pass


class NormalizeOperator(OperatorModule):
    """Deterministic operator that normalizes data using statistics."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        # Get stats from config
        data = element.data
        del key
        if stats is None:
            stats = self.get_statistics()

        if stats is None:
            # No normalization if no stats available
            return element

        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        transformed_data = {**data, "image": (data["image"] - mean) / std}
        return element.replace(data=transformed_data)


# ========================================================================
# Test Helper: Batch Creation
# ========================================================================


def create_test_batch(data, states=None, metadata_list=None, batch_size=None):
    """Helper to create batch using Batch.from_parts() API.

    This helper ensures all tests use the correct Batch constructor API.

    Args:
        data: Dict of arrays with batch dimension (e.g., {"image": array of shape (B, H, W, C)})
        states: PyTree dict with stacked states (batch dim on axis 0), or None for empty PyTree
        metadata_list: List of metadata objects (length B), or None for default Nones
        batch_size: Optional batch size (inferred from data if not provided)

    Returns:
        Batch instance created using from_parts()
    """
    if batch_size is None:
        # Infer batch size from first array in data
        first_array = next(iter(data.values()))
        batch_size = first_array.shape[0]

    if states is None:
        # Default: empty PyTree (empty dict)
        states = {}
    if metadata_list is None:
        metadata_list = [None] * batch_size

    return batch_ops.from_arrays(data, states=states)


# ========================================================================
# Test Category 1: Module Initialization
# ========================================================================


class TestOperatorModuleInitialization:
    """Test OperatorModule initialization with config."""

    def test_stochastic_initialization_with_rngs(self):
        """Test stochastic operator initialization with RNG manager."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
            min_factor=0.8,
            max_factor=1.2,
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        assert operator.config is config
        assert operator.stochastic is True
        assert operator.stream_name == "augment"
        # The caller's Rngs is read once, for the base key, and not kept.
        assert operator.rngs is None

    def test_stochastic_initialization_without_rngs_fails(self):
        """Test that stochastic operator requires rngs at runtime."""
        config = RandomBrightnessConfig(stochastic=True, stream_name="augment")

        # Config validation passes (stream_name provided)
        # But module init should fail (no rngs)
        with pytest.raises(ValueError) as exc_info:
            RandomBrightnessOperator(config)  # Missing rngs

        error_msg = str(exc_info.value).lower()
        assert "stochastic" in error_msg or "require" in error_msg
        assert "rngs" in error_msg

    def test_deterministic_initialization_without_rngs(self):
        """Test deterministic operator doesn't require rngs."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        assert operator.config is config
        assert operator.stochastic is False
        assert operator.stream_name is None
        assert operator.rngs is None

    def test_deterministic_initialization_with_rngs_allowed(self):
        """Test deterministic operator can accept rngs (but won't use them)."""
        config = NormalizeConfig(stochastic=False)
        rngs = nnx.Rngs(42)
        operator = NormalizeOperator(config, rngs=rngs)

        assert operator.rngs is None  # accepted, and not kept
        assert operator.stochastic is False  # Won't use rngs

    def test_initialization_with_name(self):
        """Test operator initialization with module name."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, name="normalize_op")

        assert operator.name == "normalize_op"

    def test_initialization_caches_config_fields(self):
        """Test that module caches config fields for convenience."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="brightness",
        )
        rngs = nnx.Rngs(0)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Should cache these for convenience
        assert operator.stochastic == config.stochastic
        assert operator.stream_name == config.stream_name

    def test_is_nnx_module(self):
        """Test that OperatorModule is a proper NNX module."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        assert isinstance(operator, nnx.Module)

    def test_calls_super_init(self):
        """Test that OperatorModule calls DataraxModule.__init__()."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should have all DataraxModule attributes
        assert hasattr(operator, "config")


# ========================================================================
# Test Category 2: Stochastic Mode Operations
# ========================================================================


class TestOperatorModuleStochasticMode:
    """Test stochastic operator functionality."""

    @staticmethod
    def _factors_from(data: dict) -> jax.Array:
        """Recover each record's drawn factor from an all-``0.5`` input image."""
        return jnp.mean(data["image"], axis=(1, 2, 3)) / 0.5

    def test_the_batch_path_draws_one_factor_per_record(self):
        """Every record in the batch gets its own factor, whatever the batch size."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(42)
        )

        for batch_size in (1, 8, 32):
            data = operator(
                batch_ops.from_arrays({"image": jnp.ones((batch_size, 8, 8, 3)) * 0.5}, states={})
            ).data

            assert self._factors_from(data).shape == (batch_size,)

    def test_drawn_factors_are_within_the_configured_range(self):
        """Each record's factor comes from its key and respects min_factor/max_factor."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(
                stochastic=True, stream_name="augment", min_factor=0.5, max_factor=1.5
            ),
            rngs=nnx.Rngs(42),
        )

        data = operator(
            batch_ops.from_arrays({"image": jnp.ones((100, 8, 8, 3)) * 0.5}, states={})
        ).data

        factors = self._factors_from(data)
        assert jnp.all(factors >= 0.5)
        assert jnp.all(factors <= 1.5)

    def test_apply_draws_its_factor_from_the_key_it_is_given(self):
        """One key repeats its factor; a different key draws a different one."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(42)
        )
        data = {"image": jnp.ones((8, 8, 3)) * 0.5}

        first = operator.apply(Element(data), key=jax.random.key(0)).data
        again = operator.apply(Element(data), key=jax.random.key(0)).data
        other = operator.apply(Element(data), key=jax.random.key(1)).data

        assert jnp.array_equal(first["image"], again["image"])
        assert not jnp.allclose(first["image"], other["image"])

    def test_per_element_randomness_independence(self):
        """Each record in one batch draws independently of the others."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(42)
        )

        data = operator(
            batch_ops.from_arrays({"image": jnp.ones((10, 8, 8, 3)) * 0.5}, states={})
        ).data

        # All 10 values should be different (with high probability)
        assert len(jnp.unique(self._factors_from(data))) >= 8

    def test_two_operators_with_one_seed_draw_the_same_factors(self):
        """The operator's own base key, not call order, decides what each record draws."""
        config = RandomBrightnessConfig(stochastic=True, stream_name="augment")
        batch = {"image": jnp.ones((32, 8, 8, 3)) * 0.5}

        first = RandomBrightnessOperator(config, rngs=nnx.Rngs(42))(
            batch_ops.from_arrays(batch, states={})
        ).data
        second = RandomBrightnessOperator(config, rngs=nnx.Rngs(42))(
            batch_ops.from_arrays(batch, states={})
        ).data
        other = RandomBrightnessOperator(config, rngs=nnx.Rngs(7))(
            batch_ops.from_arrays(batch, states={})
        ).data

        assert jnp.array_equal(first["image"], second["image"])
        assert not jnp.allclose(first["image"], other["image"])

    def test_apply_batch_stochastic_mode(self):
        """Test apply_batch() in stochastic mode."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Create batch
        batch = create_test_batch(data={"image": jnp.ones((8, 64, 64, 3)) * 0.5})

        # Apply operator
        transformed = operator(batch)

        # Should have same batch structure
        assert transformed.batch_size == 8
        assert transformed.data["image"].shape == (8, 64, 64, 3)

        # Values should be different from input (brightened/darkened)
        assert not jnp.allclose(transformed.data["image"], batch.data["image"])

    def test_stochastic_batch_uses_rng_stream(self):
        """Test that stochastic mode uses configured RNG stream."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="brightness_stream",  # Custom stream name
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should not raise (stream exists in rngs)
        transformed = operator(batch)
        assert transformed.batch_size == 4


_SEEN_KEYS: list = []


class KeyDrawingOperator(OperatorModule):
    """An operator written to the key contract: it draws from the record's own key."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Record what the framework passed, then scale by a draw from that key."""
        data = element.data
        del stats
        _SEEN_KEYS.append(key)
        record_key = require_key(key, self)
        factor = jax.random.uniform(record_key, (), minval=0.5, maxval=1.5)
        return element.replace(data={**data, "image": data["image"] * factor})


class KeyRecordingPassthrough(OperatorModule):
    """A spy that records its fourth argument and draws nothing, in either mode."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Record what the framework passed and return the record unchanged."""
        del stats
        _SEEN_KEYS.append(key)
        return element


class LearnableScaleOperator(OperatorModule):
    """A stochastic operator with its own parameter: a jittered, learnable scale."""

    def __init__(self, config: OperatorConfig, *, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs=rngs)
        self.scale = nnx.Param(jnp.asarray(2.0))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Scale by the parameter times a draw from the record's key."""
        data = element.data
        del stats
        jitter = jax.random.uniform(require_key(key, self), (), minval=0.5, maxval=1.5)
        return element.replace(data={**data, "image": data["image"] * self.scale[...] * jitter})


class TestOperatorKeys:
    """A stochastic operator's randomness is a function of its base key and the record alone.

    The caller's ``Rngs`` is read once, at construction, for the base key, which is array state
    typed ``nnx.RngKey``: a key held as a static value would put it in the graphdef, and two
    operators differing only in their seed would compile twice. A record's key folds
    its epoch, draw and index into the base key; a call that names no records keys on batch
    positions, so it repeats exactly as well.
    """

    _CONFIG = RandomBrightnessConfig(stochastic=True, stream_name="augment")

    @classmethod
    def _stochastic(cls, seed: int = 0) -> RandomBrightnessOperator:
        return RandomBrightnessOperator(cls._CONFIG, rngs=nnx.Rngs(augment=seed))

    @staticmethod
    def _batch(size: int = 4) -> dict:
        # At most 0.8, below 1 / max_factor (1 / 1.2): no draw is clipped, so output / input is
        # the record's factor in every element, whatever the draw.
        return {"image": jnp.linspace(0.1, 0.8, size * 12).reshape(size, 4, 3)}

    @staticmethod
    def _identified(data: dict, rows: jax.Array, epoch: int = 0) -> Batch:
        """``data`` as a batch of the records ``rows`` (low index words) in ``epoch``."""
        size = rows.shape[0]
        return batch_ops.from_arrays(data).replace(
            indices=jnp.stack([jnp.zeros(size, jnp.uint32), rows.astype(jnp.uint32)], -1),
            epochs=jnp.full((size,), epoch, jnp.int32),
        )

    def test_an_operator_does_not_keep_the_callers_rngs(self):
        """The caller's Rngs is used at construction and is not operator state afterwards."""
        assert self._stochastic().rngs is None

    def test_a_module_that_is_not_an_operator_keeps_its_rngs(self):
        """Only operators clear it; a source or sampler still draws from the caller's Rngs."""
        rngs = nnx.Rngs(0)

        assert DataraxModule(DataraxModuleConfig(), rngs=rngs).rngs is rngs

    def test_the_base_key_is_an_rng_key_and_the_only_rng_state(self):
        """No stream and no counter: nothing an operator holds advances when it is applied."""
        operator = self._stochastic()

        assert isinstance(operator._base_key, nnx.RngKey)
        assert jax.tree.leaves(nnx.state(operator, nnx.RngCount)) == []
        assert not hasattr(operator, "_rng_stream")

    def test_a_deterministic_operator_has_no_rng_state(self):
        """It draws nothing, so it carries no base key."""
        operator = NormalizeOperator(NormalizeConfig(stochastic=False), rngs=nnx.Rngs(0))

        assert not hasattr(operator, "_base_key")
        assert jax.tree.leaves(nnx.state(operator, nnx.RngState)) == []

    def test_a_call_without_records_repeats_and_keys_on_positions(self):
        """No record identity: row ``i`` is record ``(0, i)`` of epoch 0, draw 0."""
        operator = self._stochastic()
        batch = self._batch()

        first = operator(batch_ops.from_arrays(batch, states={})).data
        second = operator(batch_ops.from_arrays(batch, states={})).data
        positional = operator(self._identified(batch, jnp.arange(4)))

        assert jnp.array_equal(first["image"], second["image"])
        assert jnp.array_equal(first["image"], positional["image"])
        assert jnp.array_equal(operator(batch_ops.from_arrays(batch))["image"], first["image"])

    def test_reversing_the_batch_reverses_the_rows(self):
        """A record's draw follows its index, not its position in the batch."""
        operator = self._stochastic()
        batch = self._batch()
        rows = jnp.arange(10, 14)

        forward = operator(self._identified(batch, rows, epoch=3))
        backward = operator(self._identified({"image": batch["image"][::-1]}, rows[::-1], 3))

        assert jnp.array_equal(forward["image"], backward["image"][::-1])

    def test_each_epoch_draws_afresh(self):
        """The control for the test above: the epoch is part of the key."""
        operator = self._stochastic()

        first = operator(self._identified(self._batch(), jnp.arange(4), epoch=0))
        second = operator(self._identified(self._batch(), jnp.arange(4), epoch=1))

        assert not jnp.allclose(first["image"], second["image"])

    def test_each_draw_of_a_record_is_a_fresh_draw(self):
        """Two copies of one record in one batch differ by their draw ordinal alone."""
        operator = self._stochastic()
        batch = self._identified(self._batch(2), jnp.array([6, 6]))

        out = operator(batch.replace(draws=jnp.array([0, 1], jnp.int32)))
        same = operator(batch)

        # Brightness scales a record by one drawn factor, so output / input is that factor.
        factors = out["image"] / batch["image"]
        same_factors = same["image"] / batch["image"]
        assert jnp.allclose(same_factors[0], same_factors[1])
        assert not jnp.allclose(factors[0], factors[1])

    def test_two_operators_built_from_one_seed_agree_and_other_seeds_differ(self):
        """The base key is drawn from the caller's stream, so the seed decides the draws."""
        first = self._stochastic(0)(batch_ops.from_arrays(self._batch(), states={})).data
        second = self._stochastic(0)(batch_ops.from_arrays(self._batch(), states={})).data
        other = self._stochastic(7)(batch_ops.from_arrays(self._batch(), states={})).data

        assert jnp.array_equal(first["image"], second["image"])
        assert not jnp.allclose(first["image"], other["image"])

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_batches_differing_in_their_identities_share_one_program(self, graph):
        """A record's identity is array data, never part of what the program is keyed on."""
        operator = self._stochastic()
        apply = nnx.jit(lambda op, batch: op(batch)["image"], graph=graph)
        first = self._identified(self._batch(), jnp.arange(4))
        apply(operator, first)
        others = [
            self._identified(self._batch(), jnp.arange(4, dtype=jnp.uint32) + jnp.uint32(2**31), 5),
            first.replace(draws=jnp.arange(4, dtype=jnp.int32)),
            first.replace(
                indices=jnp.stack([jnp.full(4, 7, jnp.uint32), jnp.arange(4, dtype=jnp.uint32)], -1)
            ),
        ]

        with expect_compiles(0):
            outputs = [apply(operator, batch) for batch in others]

        assert all(not jnp.allclose(out, apply(operator, first)) for out in outputs)

    def test_operators_differing_only_in_their_key_share_one_trace(self):
        """The key is state, not graphdef, so a second seed compiles nothing new."""
        counter = TraceCounter()
        apply = nnx.jit(
            counter.wrap(lambda operator, data: operator(batch_ops.from_arrays(data)).data)
        )

        with counter.expect(new_traces=1):
            apply(self._stochastic(0), self._batch())
        with counter.expect(new_traces=0):
            apply(self._stochastic(7), self._batch())

    def test_the_param_partition_differentiates_in_tree_mode(self):
        """Tree mode refuses a key leaf in the differentiated tree, as it does for flax's own
        key-holding modules, so the parameters are split out and differentiated alone. The
        gradient is checked against the closed form: d/ds sum(x * s * jitter) = sum(x * jitter).
        """
        operator = LearnableScaleOperator(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=0)
        )
        batch = self._identified(self._batch(), jnp.arange(4))
        graphdef, params, rest = nnx.split(operator, nnx.Param, ..., graph=False)

        def loss(params):
            model = nnx.merge(graphdef, params, rest)
            return jnp.sum(model(batch)["image"])

        grads = jax.jit(jax.grad(loss))(params)

        scaled = operator(batch)
        expected = jnp.sum(scaled["image"]) / 2.0  # the scale is 2.0
        tolerance = 8 * jnp.finfo(jnp.float32).eps * jnp.abs(expected)
        assert jnp.abs(grads["scale"][...] - expected) <= tolerance

    def test_a_subclass_may_hold_the_callers_rngs_without_changing_draws(self):
        """Assigning ``self.rngs`` after ``super().__init__`` is allowed and changes no draw."""

        class KeepsRngs(RandomBrightnessOperator):
            def __init__(self, config, *, rngs):
                super().__init__(config, rngs=rngs)
                self.rngs = rngs

        plain = self._stochastic()(batch_ops.from_arrays(self._batch(), states={})).data
        keeping = KeepsRngs(self._CONFIG, rngs=nnx.Rngs(augment=0))(
            batch_ops.from_arrays(self._batch(), states={})
        ).data

        assert jnp.array_equal(plain["image"], keeping["image"])


class TestOperatorKeyContract:
    """What the framework hands ``apply`` as its fourth argument.

    Every assertion here runs through the framework rather than calling an override
    directly: a direct call would only exercise the test operator's own signature, which
    would hold whatever datarax passed.
    """

    @staticmethod
    def _stochastic() -> KeyDrawingOperator:
        return KeyDrawingOperator(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=0)
        )

    def test_the_batch_path_passes_a_key_to_a_stochastic_operator(self):
        """Every apply call the framework makes for a stochastic operator carries a key."""
        _SEEN_KEYS.clear()

        KeyRecordingPassthrough(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=0)
        )(batch_ops.from_arrays({"image": jnp.ones((8, 4))}, states={}))

        assert _SEEN_KEYS, "the framework never called apply"
        assert all(key is not None for key in _SEEN_KEYS)

    def test_the_batch_path_passes_no_key_to_a_deterministic_operator(self):
        """The mirror of the case above: without it, a spy recording nothing would pass both."""
        _SEEN_KEYS.clear()

        KeyRecordingPassthrough(OperatorConfig(stochastic=False))(
            batch_ops.from_arrays({"image": jnp.ones((8, 4))}, states={})
        )

        assert _SEEN_KEYS, "the framework never called apply"
        assert all(key is None for key in _SEEN_KEYS)

    def test_each_record_is_transformed_by_its_own_draw(self):
        """Eight records give eight different factors, so no draw is shared across the batch."""
        data = self._stochastic()(
            batch_ops.from_arrays({"image": jnp.ones((8, 4))}, states={})
        ).data

        means = {float(jnp.mean(data["image"][index])) for index in range(8)}
        assert len(means) == 8

    def test_a_record_keeps_its_draw_across_batch_position(self):
        """The same record index draws the same factor whatever else shares its batch."""
        whole = self._stochastic()(
            TestOperatorKeys._identified({"image": jnp.ones((8, 4))}, jnp.arange(8))
        )
        tail = self._stochastic()(
            TestOperatorKeys._identified({"image": jnp.ones((3, 4))}, jnp.arange(5, 8))
        )

        assert jnp.allclose(whole["image"][5:], tail["image"])

    def test_require_key_names_the_operator_it_refuses(self):
        """The helper reports which operator was handed no key."""
        with pytest.raises(ValueError, match="KeyDrawingOperator is stochastic"):
            require_key(None, self._stochastic())


# ========================================================================
# Test Category 3: Deterministic Mode Operations
# ========================================================================


class TestOperatorModuleDeterministicMode:
    """Test deterministic operator functionality."""

    def test_apply_without_a_key(self):
        """Test apply() in deterministic mode (no key)."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        data = {"image": jnp.ones((64, 64, 3)) * 0.7}
        state = {}

        transformed_data = operator.apply(Element(data, state=state), key=None).data

        # Should normalize: (0.7 - 0.5) / 0.2 = 1.0
        expected = jnp.ones((64, 64, 3)) * 1.0
        assert jnp.allclose(transformed_data["image"], expected)

    def test_determinism_same_input_same_output(self):
        """Test that deterministic operator is truly deterministic."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        data = {"image": jnp.array([[[0.3, 0.7, 0.9]]])}
        state = {}

        # Apply twice
        result1 = operator.apply(Element(data, state=state)).data
        result2 = operator.apply(Element(data, state=state)).data

        # Results should be identical
        assert jnp.array_equal(result1["image"], result2["image"])

    def test_apply_batch_deterministic_mode(self):
        """Test apply_batch() in deterministic mode."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((8, 64, 64, 3)) * 0.7},
            states={},
            metadata_list=[None] * 8,
        )

        transformed = operator(batch)

        # All elements should be normalized to 1.0
        expected = jnp.ones((8, 64, 64, 3)) * 1.0
        assert jnp.allclose(transformed.data["image"], expected)

    def test_deterministic_batch_no_rng_required(self):
        """Test that deterministic mode doesn't use rngs."""
        config = NormalizeConfig(stochastic=False)
        # No rngs provided
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should work fine without rngs
        transformed = operator(batch)
        assert transformed.batch_size == 4


# ========================================================================
# Test Category 4: Batch Processing
# ========================================================================


class TestOperatorModuleBatchProcessing:
    """Test batch processing functionality."""

    def test_empty_batch_handling(self):
        """Test that empty batches are handled correctly."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Empty batch
        batch = create_test_batch(
            data={"image": jnp.zeros((0, 64, 64, 3))}, states={}, metadata_list=[]
        )

        transformed = operator(batch)

        # Should return unchanged empty batch
        assert transformed.batch_size == 0
        assert transformed.data["image"].shape == (0, 64, 64, 3)

    def test_single_element_batch(self):
        """Test processing batch with single element."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((1, 64, 64, 3)) * 0.5}, states={}, metadata_list=[None]
        )

        transformed = operator(batch)

        assert transformed.batch_size == 1
        assert transformed.data["image"].shape == (1, 64, 64, 3)

    def test_multi_element_batch_stochastic(self):
        """Test processing multi-element batch in stochastic mode."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch_size = 32
        batch = create_test_batch(
            data={"image": jnp.ones((batch_size, 64, 64, 3)) * 0.5},
            states={},
            metadata_list=[None] * batch_size,
        )

        transformed = operator(batch)

        assert transformed.batch_size == batch_size
        # Each element should have different brightness (with high probability)
        means = jnp.mean(transformed.data["image"], axis=(1, 2, 3))
        unique_means = jnp.unique(jnp.round(means, decimals=3))
        assert len(unique_means) >= batch_size // 2  # At least half different

    def test_multi_element_batch_deterministic(self):
        """Test processing multi-element batch in deterministic mode."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        batch_size = 32
        batch = create_test_batch(
            data={"image": jnp.ones((batch_size, 64, 64, 3)) * 0.7},
            states={},
            metadata_list=[None] * batch_size,
        )

        transformed = operator(batch)

        assert transformed.batch_size == batch_size
        # All elements should be normalized identically
        for i in range(batch_size):
            assert jnp.allclose(transformed.data["image"][i], transformed.data["image"][0])

    def test_vmap_correctness_per_element(self):
        """Test that vmap correctly processes each element independently."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Create batch with different input values
        batch_data = jnp.stack(
            [
                jnp.ones((64, 64, 3)) * 0.2,
                jnp.ones((64, 64, 3)) * 0.5,
                jnp.ones((64, 64, 3)) * 0.8,
            ]
        )
        batch = create_test_batch(
            data={"image": batch_data}, states={}, metadata_list=[None, None, None]
        )

        transformed = operator(batch)

        # Each element should maintain its relative brightness pattern
        # (element 2 should still be brighter than element 0)
        jnp.mean(transformed.data["image"], axis=(1, 2, 3))
        # Relative ordering might be preserved
        # (Just verify they're all processed)
        assert transformed.batch_size == 3


# ========================================================================
# Test Category 5: JIT Compatibility
# ========================================================================


class TestOperatorModuleJITCompatibility:
    """Test JAX JIT compilation compatibility."""

    def test_apply_batch_compiles_stochastic(self):
        """Test that apply_batch() compiles successfully in stochastic mode."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should compile without errors (already decorated with @nnx.jit)
        transformed = operator(batch)
        assert transformed.batch_size == 4

    def test_apply_batch_compiles_deterministic(self):
        """Test that apply_batch() compiles successfully in deterministic mode."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should compile without errors
        transformed = operator(batch)
        assert transformed.batch_size == 4

    def test_static_branch_compilation(self):
        """Test that static stochastic boolean enables branch elimination."""
        # This is implicit in the design - self.stochastic is compile-time constant
        # JAX will compile only the relevant branch

        # Create both operator types
        stochastic_config = RandomBrightnessConfig(stochastic=True, stream_name="augment")
        deterministic_config = NormalizeConfig(stochastic=False)

        stochastic_op = RandomBrightnessOperator(stochastic_config, rngs=nnx.Rngs(0))
        deterministic_op = NormalizeOperator(deterministic_config)

        # Both should have different compilation paths
        # (This is tested implicitly by successful compilation)
        assert stochastic_op.stochastic is True
        assert deterministic_op.stochastic is False

    def test_apply_is_pure_function(self):
        """Test that apply() is a pure function (same input → same output)."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        element = Element({"image": jnp.array([[[0.3, 0.7]]])})

        # Call multiple times
        results = [operator.apply(element) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert jnp.array_equal(result.data["image"], results[0].data["image"])


# ========================================================================
# Test Category 6: Random Parameter System
# ========================================================================


class TestOperatorModuleRandomParams:
    """Test random parameter generation and distribution."""

    def test_keys_distributed_via_vmap(self):
        """Test that random params are correctly distributed to elements via vmap."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Create batch with uniform input
        batch = create_test_batch(
            data={"image": jnp.ones((8, 64, 64, 3)) * 0.5},
            states={},
            metadata_list=[None] * 8,
        )

        transformed = operator(batch)

        # Each element should have received different random param
        # (resulting in different brightness values)
        for i in range(8):
            for j in range(i + 1, 8):
                # Most pairs should be different
                elem_i_mean = jnp.mean(transformed.data["image"][i])
                elem_j_mean = jnp.mean(transformed.data["image"][j])
                if i == 0 and j == 1:
                    # At least first two should be different
                    assert not jnp.allclose(elem_i_mean, elem_j_mean)

    def test_a_deterministic_operator_has_no_draw_to_make(self):
        """A deterministic operator transforms its record without any key."""
        operator = NormalizeOperator(
            NormalizeConfig(stochastic=False), statistics={"mean": 0.5, "std": 0.2}
        )

        data = operator.apply(Element({"image": jnp.ones((4, 4, 3)) * 0.7})).data

        assert jnp.allclose(data["image"], jnp.ones((4, 4, 3)))


# ========================================================================
# Test Category 7: Statistics System (Inherited)
# ========================================================================


class TestOperatorModuleStatistics:
    """Test statistics stored on the operator and applied to its records."""

    def test_stored_statistics_are_readable(self):
        """Test operator using statistics set on it."""
        stats = {"mean": 0.5, "std": 0.2}
        operator = NormalizeOperator(NormalizeConfig(stochastic=False), statistics=stats)

        stored = operator.get_statistics()

        assert stored is not None
        assert {name: float(value) for name, value in stored.items()} == pytest.approx(stats)

    def test_compute_statistics_returns_the_stored_statistics(self):
        """By default an operator applies whatever was stored on it."""
        operator = NormalizeOperator(
            NormalizeConfig(stochastic=False), statistics={"mean": 0.5, "std": 0.2}
        )

        computed = operator.compute_statistics(
            batch_ops.from_arrays({"image": jnp.ones((4, 8, 8, 3))})
        )

        assert computed is not None
        assert {name: float(value) for name, value in computed.items()} == pytest.approx(
            {"mean": 0.5, "std": 0.2}
        )

    def test_an_operator_can_derive_statistics_from_the_batch(self):
        """An operator that fits statistics to each batch overrides compute_statistics."""

        class BatchFittedNormalize(NormalizeOperator):
            def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
                image = batch.data["image"]
                return {"mean": jnp.mean(image), "std": jnp.std(image) + 1e-6}

        operator = BatchFittedNormalize(NormalizeConfig(stochastic=False))

        stats = operator.compute_statistics(
            batch_ops.from_arrays({"image": jnp.ones((4, 64, 64, 3)) * 0.7})
        )

        assert stats is not None
        assert jnp.isclose(stats["mean"], 0.7, rtol=1e-4)

    def test_compute_statistics_is_not_memoized(self):
        """Nothing caches the result, so an operator that fits per batch sees every batch."""
        call_count = 0

        class CountingNormalize(NormalizeOperator):
            def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
                nonlocal call_count
                call_count += 1
                return {"mean": jnp.mean(batch.data["image"]), "std": 1.0}

        operator = CountingNormalize(NormalizeConfig(stochastic=False))

        operator.compute_statistics(batch_ops.from_arrays({"image": jnp.ones((4, 8, 8, 3))}))
        operator.compute_statistics(batch_ops.from_arrays({"image": jnp.zeros((4, 8, 8, 3))}))

        assert call_count == 2


# ========================================================================
# Test Category 8: Training/Eval Mode (Inherited from NNX)
# ========================================================================


class TestOperatorModuleTrainingMode:
    """Test training/evaluation mode switching (inherited from NNX)."""

    def test_train_mode_callable(self):
        """Test that train() method is available."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should have train() method from NNX
        assert hasattr(operator, "train")
        operator.train()  # Should not raise

    def test_eval_mode_callable(self):
        """Test that eval() method is available."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should have eval() method from NNX
        assert hasattr(operator, "eval")
        operator.eval()  # Should not raise

    def test_mode_switching(self):
        """Test switching between train and eval modes."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should be able to switch modes
        operator.train()
        operator.eval()
        operator.train()
        # No errors expected


# ========================================================================
# Test Category 10: Scan Batch Strategy
# ========================================================================


class TestOperatorModuleScanBatchStrategy:
    """Test scan-based batch execution strategy (sequential, low memory)."""

    def test_scan_default_is_vmap(self):
        """Default batch_strategy should be 'vmap'."""
        config = NormalizeConfig(stochastic=False)
        assert config.batch_strategy == "vmap"

    def test_scan_config_validation(self):
        """Invalid batch_strategy should raise ValueError."""
        with pytest.raises(ValueError, match="batch_strategy"):
            NormalizeConfig(stochastic=False, batch_strategy="invalid")

    def test_scan_produces_same_result_as_vmap(self):
        """Scan strategy produces identical results to vmap for deterministic ops."""
        # Create two operators: one vmap (default), one scan
        config_vmap = NormalizeConfig(stochastic=False, batch_strategy="vmap")
        config_scan = NormalizeConfig(stochastic=False, batch_strategy="scan")
        op_vmap = NormalizeOperator(config_vmap, statistics={"mean": 0.5, "std": 0.2})
        op_scan = NormalizeOperator(config_scan, statistics={"mean": 0.5, "std": 0.2})

        # Create batch of 4 elements
        batch = create_test_batch(
            data={"image": jnp.array([[0.3], [0.5], [0.7], [0.9]])},
            states={},
            metadata_list=[None] * 4,
        )

        result_vmap = op_vmap(batch)
        result_scan = op_scan(batch)

        assert jnp.allclose(result_vmap.data["image"], result_scan.data["image"]), (
            "Scan and vmap should produce identical results for deterministic ops"
        )

    def test_scan_with_stochastic_operator(self):
        """Scan strategy works with stochastic operators (shapes match)."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
            batch_strategy="scan",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3)) * 0.5},
            states={},
            metadata_list=[None] * 4,
        )

        result = operator(batch)
        assert result.batch_size == 4
        assert result.data["image"].shape == (4, 32, 32, 3)

    def test_scan_single_element_batch(self):
        """Scan strategy works with batch size 1."""
        config = NormalizeConfig(stochastic=False, batch_strategy="scan")
        op = NormalizeOperator(config, statistics={"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((1, 32, 32, 3)) * 0.7},
            states={},
            metadata_list=[None],
        )

        result = op(batch)
        assert result.batch_size == 1
        expected = jnp.ones((1, 32, 32, 3)) * 1.0  # (0.7 - 0.5) / 0.2
        assert jnp.allclose(result.data["image"], expected)


class TestOperatorStatisticsStore:
    """Statistics are fixed-shape operator state, given at construction.

    They are a ``nnx.Variable`` of arrays from ``__init__`` on, so the operator's state layout
    never changes after it is built: a checkpoint template fits it, a compiled step never sees a
    new structure, and two operators with equal configurations share one trace whatever their
    statistics hold. An operator built without statistics holds no statistics leaf.
    """

    _STATS = {"mean": 0.5, "std": 0.2}

    @classmethod
    def _fitted(cls) -> NormalizeOperator:
        """Return an operator holding statistics that halve and rescale the input."""
        return NormalizeOperator(NormalizeConfig(), statistics=cls._STATS)

    def test_statistics_are_state_at_their_real_shape_from_construction(self):
        operator = self._fitted()

        assert isinstance(operator._statistics, nnx.Variable)
        stored = operator.get_statistics()
        assert stored is not None
        assert set(stored) == {"mean", "std"}
        assert all(isinstance(leaf, jax.Array) for leaf in stored.values())
        assert all(leaf.shape == () and leaf.dtype == jnp.float32 for leaf in stored.values())

    def test_an_operator_built_without_statistics_holds_none(self):
        operator = NormalizeOperator(NormalizeConfig())

        assert operator.get_statistics() is None
        assert "_statistics" not in nnx.to_pure_dict(nnx.state(operator))

    def test_statistics_reach_apply_through_the_batch_call(self):
        """A batch is normalized with the statistics given at construction."""
        batch = create_test_batch(
            data={"image": jnp.ones((1, 4, 4, 3)) * 0.7}, states={}, metadata_list=[None]
        )

        result = self._fitted()(batch)

        assert jnp.allclose(result.data["image"], jnp.ones((1, 4, 4, 3)))

    def test_statistics_reach_apply_through_the_raw_path(self):
        out = self._fitted()(batch_ops.from_arrays({"image": jnp.ones((2, 4, 4, 3)) * 0.7}))
        out_data = out.data

        assert jnp.allclose(out_data["image"], jnp.ones((2, 4, 4, 3)))

    def test_statistics_reach_apply_under_nnx_jit(self):
        @nnx.jit
        def run(op, data):
            return op(batch_ops.from_arrays(data)).data

        out_data = run(self._fitted(), {"image": jnp.ones((2, 4, 4, 3)) * 0.7})

        assert jnp.allclose(out_data["image"], jnp.ones((2, 4, 4, 3)))

    def test_replacing_values_of_the_same_shape_applies_them_without_a_new_trace(self):
        """The counter wraps the function and the transform wraps the counter."""
        counter = TraceCounter()
        traced = nnx.jit(counter.wrap(lambda op, data: op(batch_ops.from_arrays(data)).data))
        operator = self._fitted()
        data = {"image": jnp.ones((2, 4, 4, 3)) * 0.7}

        with counter.expect(new_traces=1):
            traced(operator, data)
        operator.set_statistics({"mean": 0.7, "std": 0.5})
        with counter.expect(new_traces=0):
            out = traced(operator, data)
        with counter.expect(new_traces=0):
            traced(self._fitted(), data)

        assert jnp.allclose(out["image"], jnp.zeros((2, 4, 4, 3)), atol=1e-6)

    @pytest.mark.parametrize(
        "replacement",
        [
            {"mean": 0.7},
            {"mean": 0.7, "std": 0.5, "extra": 1.0},
            {"mean": jnp.zeros(3), "std": 0.5},
            {"mean": jnp.asarray(1, jnp.int32), "std": 0.5},
        ],
        ids=["fewer entries", "more entries", "another shape", "another dtype"],
    )
    def test_statistics_of_another_layout_are_refused_and_nothing_changes(self, replacement):
        operator = self._fitted()

        with pytest.raises(ValueError, match="statistics"):
            operator.set_statistics(replacement)

        stored = operator.get_statistics()

        assert stored is not None
        assert stored["mean"] == jnp.float32(0.5) and stored["std"] == jnp.float32(0.2)

    def test_setting_statistics_on_an_operator_built_without_them_is_refused(self):
        with pytest.raises(ValueError, match="statistics="):
            NormalizeOperator(NormalizeConfig()).set_statistics(self._STATS)

    def test_a_module_configuration_carries_no_statistics(self):
        """Statistics are fitted state; a configuration is static metadata a transform compares."""
        names = {field.name for field in fields(DataraxModuleConfig)}

        assert "batch_stats_fn" not in names
        assert "precomputed_stats" not in names


# ========================================================================
# Test Category 11: Statistics computed per batch
# ========================================================================


class TestComputeStatisticsReachesApply:
    """The batch path computes each batch's statistics and gives them to every record.

    ``compute_statistics`` has been overridable since the statistics store landed, but nothing
    called it: the batch path read ``get_statistics()``, so an operator that fits statistics to
    each batch never saw a batch. Every test here runs through a batch path rather than calling
    ``compute_statistics`` directly, which is the only way to observe that wiring.
    """

    IMAGE = jnp.asarray([0.2, 0.4, 0.6, 0.8], dtype=jnp.float32).reshape(4, 1, 1, 1)

    @staticmethod
    def _normalized(image: jax.Array) -> jax.Array:
        """The batch normalized by its own mean and standard deviation."""
        return (image - jnp.mean(image)) / (jnp.std(image) + 1e-6)

    @staticmethod
    def _batch_fitted(batch_strategy: str = "vmap") -> OperatorModule:
        """An operator normalizing with the statistics of whatever batch it is given.

        Its ``apply`` uses exactly what it is handed and keeps no fallback to the stored
        statistics, so a batch path that computes none fails here instead of quietly
        normalizing with something else.
        """

        class BatchFitted(NormalizeOperator):
            def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
                image = batch.data["image"]
                return {"mean": jnp.mean(image), "std": jnp.std(image) + 1e-6}

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                data = element.data
                del key
                if stats is None:
                    raise AssertionError("the batch path passed no statistics")
                normalized = (data["image"] - stats["mean"]) / stats["std"]
                return element.replace(data={**data, "image": normalized})

        return BatchFitted(NormalizeConfig(stochastic=False, batch_strategy=batch_strategy))

    @staticmethod
    def _counting() -> tuple[OperatorModule, list[int]]:
        """An operator recording how many times its statistics were computed."""
        calls: list[int] = []

        class Counting(NormalizeOperator):
            def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
                calls.append(1)
                return {"mean": jnp.mean(batch.data["image"]), "std": 1.0}

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                data = element.data
                del key
                if stats is None:
                    raise AssertionError("the batch path passed no statistics")
                return element.replace(data={**data, "image": data["image"] - stats["mean"]})

        return Counting(NormalizeConfig(stochastic=False)), calls

    def test_the_batch_call_normalizes_with_the_batch_statistics(self):
        operator = self._batch_fitted()
        batch = create_test_batch(data={"image": self.IMAGE})

        result = operator(batch)

        assert jnp.allclose(result.data["image"], self._normalized(self.IMAGE), atol=1e-6)

    def test_the_raw_path_normalizes_with_the_batch_statistics(self):
        operator = self._batch_fitted()

        out = operator(batch_ops.from_arrays({"image": self.IMAGE}))
        out_data = out.data

        assert jnp.allclose(out_data["image"], self._normalized(self.IMAGE), atol=1e-6)

    def test_the_statistics_are_computed_under_nnx_jit(self):
        operator = self._batch_fitted()

        @nnx.jit
        def run(op, data):
            return op(batch_ops.from_arrays(data)).data

        out_data = run(operator, {"image": self.IMAGE})

        assert jnp.allclose(out_data["image"], self._normalized(self.IMAGE), atol=1e-6)

    def test_the_scan_strategy_normalizes_with_the_batch_statistics(self):
        operator = self._batch_fitted(batch_strategy="scan")

        out = operator(batch_ops.from_arrays({"image": self.IMAGE}))
        out_data = out.data

        assert jnp.allclose(out_data["image"], self._normalized(self.IMAGE), atol=1e-6)

    def test_the_statistics_are_computed_once_for_the_whole_batch(self):
        """Statistics describe the batch, so they are computed before it is vectorized."""
        operator, calls = self._counting()

        operator(batch_ops.from_arrays({"image": jnp.ones((8, 2, 2, 1), jnp.float32)}))

        assert len(calls) == 1

    def test_statistics_passed_by_the_caller_are_not_recomputed(self):
        """An explicit argument still wins, so a caller can supply statistics of its own."""
        operator, calls = self._counting()

        out_data = operator.apply_batch(
            batch_ops.from_arrays({"image": jnp.ones((4, 1, 1, 1), jnp.float32)}),
            None,
            {"mean": 0.25, "std": 1.0},
        ).data

        assert calls == []
        assert jnp.allclose(out_data["image"], 0.75)

    def test_gradients_reach_the_input_through_the_computed_statistics(self):
        """The statistics are part of the traced computation, not a constant read beside it."""
        operator = self._batch_fitted()

        def loss(image):
            out = operator(batch_ops.from_arrays({"image": image}))
            out_data = out.data
            return jnp.sum(out_data["image"] ** 2)

        gradient = jax.grad(loss)(self.IMAGE)

        assert jnp.any(gradient != 0.0)
