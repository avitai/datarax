"""The operator contract: one per-record method, everything else inherited (design D6).

An operator implements ``apply(element, key, stats) -> Element``; the base class's ``__call__``
computes statistics, derives one key per record from its identity, applies the mode, and maps
``apply`` over the records by ``batch_strategy``. A whole-batch operator overrides
``apply_batch(batch, keys, stats)`` instead and takes its first record's key.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

from datarax.core import batch_ops
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import to_words
from datarax.core.operator import OperatorModule, require_key
from datarax.core.prng import per_record_keys, record_key
from datarax.core.state_keys import WEIGHT
from datarax.pipeline.dag import name_records


B = 6


def _batch(values: jax.Array | None = None, first: int = 10, epoch: int = 1) -> Batch:
    x = jnp.linspace(0.0, 1.0, B * 3, dtype=jnp.float32).reshape(B, 3) if values is None else values
    return name_records(batch_ops.from_arrays({"x": x}), to_words(jnp.arange(B) + first), epoch)


def _stochastic(strategy: str = "vmap") -> OperatorConfig:
    return OperatorConfig(stochastic=True, stream_name="augment", batch_strategy=strategy)


class Jitter(OperatorModule):
    """Adds learnable-scale Gaussian noise; defined by ``apply`` alone."""

    def __init__(self, config: OperatorConfig, *, rngs: nnx.Rngs) -> None:
        super().__init__(config, rngs=rngs)
        self.scale = nnx.Param(jnp.asarray(0.1, jnp.float32))

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        noise = jax.random.normal(require_key(key, self), element.data["x"].shape)
        return element.replace(data={"x": element.data["x"] + self.scale[...] * noise})


class Swap(OperatorModule):
    """Reverses the batch's rows; a whole-batch operator defined by ``apply_batch`` alone.

    It records the random bits of the one key it was handed, so a test can compare that key.
    """

    def apply_batch(
        self, batch: Batch, keys: jax.Array | None, stats: dict[str, Any] | None
    ) -> Batch:
        assert keys is not None
        bits = jax.random.bits(keys[0], (2,))
        return batch.replace(
            data={"x": batch.data["x"][::-1]}, batch_state={**batch.batch_state, "bits": bits}
        )


def _gradient_value(state: Any, name: str) -> np.ndarray:
    """The array the variable ``name`` holds in a gradient state."""
    variable = state[name]
    assert isinstance(variable, nnx.Variable)
    return np.asarray(variable[...])


def _jitter(strategy: str = "vmap") -> Jitter:
    return Jitter(_stochastic(strategy), rngs=nnx.Rngs(augment=3))


@pytest.mark.parametrize("strategy", ["vmap", "scan"])
class TestAnOperatorDefinedByApplyAlone:
    """``apply`` alone gives an operator every transform, mode and batch strategy."""

    def test_records_draw_their_own_values(self, strategy: str) -> None:
        out = _jitter(strategy)(_batch(jnp.zeros((B, 3), jnp.float32)))
        rows = np.asarray(out.data["x"])
        assert len({tuple(row) for row in rows.round(6)}) == B

    def test_every_transform_gives_the_eager_result(self, strategy: str) -> None:
        op, batch = _jitter(strategy), _batch()
        eager = op(batch).data["x"]
        graph = nnx.jit(lambda o, b: o(b))(op, batch).data["x"]
        tree = nnx.jit(lambda o, b: o(b), graph=False)(op, batch).data["x"]
        partial = nnx.jit_partial(lambda o, b: o(b), op, graph=False)(batch).data["x"]
        for result in (graph, tree, partial):
            np.testing.assert_allclose(result, eager, rtol=0, atol=1e-6)

    def test_vmap_and_scan_over_a_chunk_equal_each_batch(self, strategy: str) -> None:
        op = _jitter(strategy)
        batches = [_batch(first=10 + B * k) for k in range(3)]
        chunk = batch_ops.stack(batches)
        each = jnp.stack([op(b).data["x"] for b in batches])
        vmapped = jax.vmap(lambda b: op(b).data["x"])(chunk)
        _, scanned = jax.lax.scan(lambda c, b: (c, op(b).data["x"]), None, chunk)
        np.testing.assert_allclose(vmapped, each, rtol=0, atol=1e-6)
        np.testing.assert_allclose(scanned, each, rtol=0, atol=1e-6)

    def test_eval_mode_returns_the_records_unchanged(self, strategy: str) -> None:
        op, batch = _jitter(strategy), _batch()
        op.eval()
        np.testing.assert_array_equal(op(batch).data["x"], batch.data["x"])
        op.train()
        assert not np.array_equal(op(batch).data["x"], batch.data["x"])

    def test_batches_differing_in_values_compile_once(self, strategy: str) -> None:
        op = _jitter(strategy)
        step = nnx.jit(lambda o, b: o(b), graph=False)
        step(op, _batch())
        other = _batch(jnp.ones((B, 3), jnp.float32), first=99, epoch=4)
        with expect_compiles(0):
            step(op, other)

    def test_parameter_gradient_checks_against_finite_differences(self, strategy: str) -> None:
        op, batch = _jitter(strategy), _batch()
        check_parameter_gradients(op, lambda o: jnp.sum(o(batch).data["x"] ** 2))

    def test_input_gradient_checks_against_finite_differences(self, strategy: str) -> None:
        op, batch = _jitter(strategy), _batch()
        check_input_gradients(
            op,
            lambda o, x: jnp.sum(o(batch.replace(data={"x": x})).data["x"] ** 2),
            batch.data["x"],
        )


class TestKeys:
    """A record's key depends on its identity only, in the batch form and the per-record form."""

    def test_the_batch_form_keys_equal_the_record_key_of_each_record(self) -> None:
        base = jax.random.key(7)
        batch = _batch()
        keys = per_record_keys(base, batch.indices, batch.epochs, batch.draws)
        for i in range(B):
            record = batch_ops.element(batch, i)
            assert record.index is not None
            assert jnp.array_equal(
                jax.random.key_data(keys[i]),
                jax.random.key_data(record_key(base, record.index, record.epoch, record.draw)),
            )

    def test_apply_record_draws_what_the_batch_form_draws(self) -> None:
        op, batch = _jitter(), _batch()
        stacked = jnp.stack(
            [op.apply_record(batch_ops.element(batch, i)).data["x"] for i in range(B)]
        )
        np.testing.assert_allclose(stacked, op(batch).data["x"], rtol=0, atol=1e-6)

    def test_a_record_draws_the_same_values_in_any_batch(self) -> None:
        op, batch = _jitter(), _batch()
        alone = op(batch_ops.slice_rows(batch, 2, 3)).data["x"][0]
        np.testing.assert_allclose(alone, op(batch).data["x"][2], rtol=0, atol=1e-6)

    def test_a_stochastic_operator_refuses_an_element_without_identity(self) -> None:
        with pytest.raises(ValueError, match="identity"):
            _jitter().apply_record(Element({"x": jnp.zeros(3)}))

    def test_a_direct_apply_call_takes_the_callers_key(self) -> None:
        op, key = _jitter(), jax.random.key(0)
        out = op.apply(Element({"x": jnp.zeros(3)}), key)
        expected = 0.1 * jax.random.normal(key, (3,))
        np.testing.assert_allclose(out.data["x"], expected, rtol=0, atol=1e-7)


class TestAWholeBatchOperatorDefinedByApplyBatchAlone:
    """Overriding ``apply_batch`` alone gives an operator every transform and the mode."""

    def test_its_one_key_is_its_first_records_key(self) -> None:
        op = Swap(_stochastic(), rngs=nnx.Rngs(augment=5))
        batch = _batch()
        first = batch_ops.element(batch, 0)
        assert first.index is not None
        expected = jax.random.bits(
            record_key(op._base_key[...], first.index, first.epoch, first.draw), (2,)
        )
        assert jnp.array_equal(op(batch).batch_state["bits"], expected)

    def test_every_transform_gives_the_eager_result(self) -> None:
        op, batch = Swap(_stochastic(), rngs=nnx.Rngs(augment=5)), _batch()
        eager = op(batch)
        for result in (
            nnx.jit(lambda o, b: o(b))(op, batch),
            nnx.jit(lambda o, b: o(b), graph=False)(op, batch),
        ):
            np.testing.assert_array_equal(result.data["x"], eager.data["x"])
            np.testing.assert_array_equal(result.batch_state["bits"], eager.batch_state["bits"])

    def test_eval_mode_returns_the_batch_unchanged(self) -> None:
        op, batch = Swap(_stochastic(), rngs=nnx.Rngs(augment=5)), _batch()
        op.eval()
        assert op(batch) is batch

    def test_it_has_no_per_record_form(self) -> None:
        op = Swap(_stochastic(), rngs=nnx.Rngs(augment=5))
        assert not op.has_record_form
        with pytest.raises(TypeError, match="apply_batch"):
            op.apply_record(batch_ops.element(_batch(), 0))


class TestOperatorRules:
    """What ``apply`` may do (design D6 operator rules)."""

    def test_a_module_state_write_inside_apply_is_refused_naming_the_whole_batch_form(
        self,
    ) -> None:
        class Counter(OperatorModule):
            def __init__(self) -> None:
                super().__init__(OperatorConfig())
                self.count = nnx.Variable(jnp.zeros((), jnp.int32))

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                self.count[...] += 1
                return element

        with pytest.raises(TypeError, match="apply_batch"):
            Counter()(_batch())

    @pytest.mark.parametrize("strategy", ["vmap", "scan"])
    def test_batchnorm_in_training_is_refused_and_its_running_average_form_accepted(
        self, strategy: str
    ) -> None:
        class Normed(OperatorModule):
            def __init__(self, running_average: bool | None) -> None:
                super().__init__(OperatorConfig(batch_strategy=strategy))
                self.norm = nnx.BatchNorm(3, rngs=nnx.Rngs(0))
                # Not named use_running_average: train() and eval() set every attribute of that
                # name, which would overwrite the call-site override this test needs.
                self.running_average = running_average

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                x = self.norm(element.data["x"][None], use_running_average=self.running_average)
                return element.replace(data={"x": x[0]})

        training = Normed(None)
        training.train()
        with pytest.raises(TypeError, match="apply_batch"):
            training(_batch())
        overridden = Normed(True)
        overridden.train()
        assert overridden(_batch()).data["x"].shape == (B, 3)
        training.eval()
        assert training(_batch()).data["x"].shape == (B, 3)

    def test_a_dropout_inside_apply_draws_from_the_records_key(self) -> None:
        class Dropped(OperatorModule):
            def __init__(self) -> None:
                super().__init__(_stochastic(), rngs=nnx.Rngs(augment=1))
                self.dropout = nnx.Dropout(0.5)

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                x = self.dropout(element.data["x"], rngs=require_key(key, self))
                return element.replace(data={"x": x})

        op, batch = Dropped(), _batch(jnp.ones((B, 64), jnp.float32))
        masks = np.asarray(op(batch).data["x"] == 0.0)
        assert len({row.tobytes() for row in masks}) == B
        alone = np.asarray(op(batch_ops.slice_rows(batch, 4, 5)).data["x"][0] == 0.0)
        np.testing.assert_array_equal(alone, masks[4])

    def test_statistics_computed_at_batch_level_reach_apply(self) -> None:
        class Center(OperatorModule):
            def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
                return {"mean": jnp.mean(batch.data["x"], axis=0)}

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                assert stats is not None
                return element.replace(data={"x": element.data["x"] - stats["mean"]})

        out = Center(OperatorConfig())(_batch())
        np.testing.assert_allclose(jnp.mean(out.data["x"], axis=0), 0.0, atol=1e-6)

    def test_an_identity_change_inside_apply_is_refused(self) -> None:
        class Rename(OperatorModule):
            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                return element.replace(epoch=element.epoch + 1)

        with pytest.raises(ValueError, match="identity"):
            Rename(OperatorConfig())(_batch())

    def test_a_state_value_computed_from_params_carries_their_gradient(self) -> None:
        class Weigh(OperatorModule):
            def __init__(self) -> None:
                super().__init__(OperatorConfig())
                self.temperature = nnx.Param(jnp.asarray(2.0, jnp.float32))

            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                weight = jax.nn.sigmoid(self.temperature[...] * jnp.sum(element.data["x"]))
                return element.replace(state={**element.state, WEIGHT: weight})

        # Dyadic records (k/8): every float32 row sum is exact in any reduction order, so the
        # jitted float64 gradient check and the float64 closed form compute the same numbers.
        op, batch = Weigh(), _batch(jnp.arange(B * 3, dtype=jnp.float32).reshape(B, 3) / 8)

        def loss(o: Weigh) -> jax.Array:
            out = o(batch)
            return jnp.sum(out.states[WEIGHT] * jnp.sum(out.data["x"], axis=1))

        gradient = check_parameter_gradients(op, loss)
        s = np.sum(np.asarray(batch.data["x"], np.float64), axis=1)
        sigma = 1.0 / (1.0 + np.exp(-2.0 * s))
        expected = np.sum(sigma * (1 - sigma) * s * s)
        np.testing.assert_allclose(
            _gradient_value(gradient, "temperature"),
            expected,
            rtol=4 * float(np.finfo(np.float64).eps),
        )


class TestEvalModeWithADeterministicForm:
    """In eval mode a stochastic operator applies its ``apply_deterministic`` to every record."""

    def test_an_overridden_deterministic_form_is_mapped_over_the_records(self) -> None:
        class JitterOrCenter(Jitter):
            def apply_deterministic(
                self, element: Element, stats: dict[str, Any] | None = None
            ) -> Element:
                return element.replace(data={"x": element.data["x"] - 0.5})

        op = JitterOrCenter(_stochastic(), rngs=nnx.Rngs(augment=3))
        batch = _batch()
        op.eval()
        np.testing.assert_allclose(op(batch).data["x"], batch.data["x"] - 0.5, rtol=0, atol=0)
        op.train()
        assert not np.allclose(op(batch).data["x"], batch.data["x"] - 0.5)
