"""Missing values: a field that may be missing is ``Maybe(value, present)`` in ``data`` (design D8).

A missing value's slot holds zeros and ``Maybe`` has no arithmetic, so the fill value is read only
through ``value_or(fill)`` or with ``present`` in hand. A value an operator hides but keeps as the
target is marked in ``state[MASKED][field]``; a value an operator fills is marked in
``state[IMPUTED][field]`` with ``present`` set. Padding rows are a third thing (``WEIGHT`` 0).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from substrax.spmd import place_batch_on_shards
from substrax.testing.compiles import expect_compiles

from datarax.core import batch_ops, Maybe
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule, require_key
from datarax.core.spec import array_to_spec, batched_spec, SpecMismatchError, validate_batch
from datarax.core.state_keys import IMPUTED, MASKED
from datarax.pipeline.dag import name_records


B = 8
H, W, C, T = 4, 4, 3, 6  # image and text-feature shapes of the fill tests
PRESENT = np.array([True, False, True, True, False, True, False, True])


def depth(present: np.ndarray = PRESENT, seed: int = 0) -> Maybe:
    """A missing-capable ``(B, 4, 4)`` field whose missing slots hold zeros."""
    values = np.random.default_rng(seed).random((present.shape[0], 4, 4)).astype(np.float32) + 0.5
    return Maybe(jnp.asarray(np.where(present[:, None, None], values, 0.0)), jnp.asarray(present))


def batch_with_depth(present: np.ndarray = PRESENT, seed: int = 0) -> Batch:
    size = present.shape[0]
    return batch_ops.from_arrays(
        {
            "image": jnp.asarray(np.random.default_rng(seed).random((size, 2, 2)), jnp.float32),
            "depth": depth(present, seed),
        }
    )


class TestLayout:
    """A registered frozen dataclass of two arrays: a pytree node, not an nnx type."""

    def test_its_leaves_are_value_and_present(self) -> None:
        field = depth()

        leaves = jax.tree.leaves(field)

        assert len(leaves) == 2
        assert leaves[0] is field.value and leaves[1] is field.present

    def test_it_is_frozen_slotted_and_not_an_nnx_type(self) -> None:
        field = depth()

        with pytest.raises(dataclasses.FrozenInstanceError):
            field.value = jnp.zeros(3)  # pyright: ignore[reportAttributeAccessIssue]
        assert not hasattr(field, "__dict__")
        assert not isinstance(field, nnx.Module | nnx.Pytree | nnx.Variable)

    def test_presence_patterns_share_one_treedef(self) -> None:
        """Nothing is aux: which records are present never keys a compiled program."""
        assert jax.tree.structure(depth(PRESENT)) == jax.tree.structure(depth(~PRESENT))

    def test_presence_costs_one_byte_per_record(self) -> None:
        field = depth()

        assert field.present.dtype == jnp.bool_
        assert field.present.nbytes == B


class TestNoArithmetic:
    """A fill value cannot be used by accident: every implicit use of the value raises."""

    @pytest.mark.parametrize(
        "misuse",
        [
            pytest.param(lambda m: m * 2, id="times"),
            pytest.param(lambda m: m + 1.0, id="plus"),
            pytest.param(lambda m: 1.0 - m, id="reflected-minus"),
            pytest.param(lambda m: -m, id="negate"),
            pytest.param(lambda m: m > 0, id="greater"),
            pytest.param(lambda m: m == 0, id="equals-a-number"),
            pytest.param(lambda m: m != 0, id="differs-from-a-number"),
            pytest.param(lambda m: m[0], id="index"),
            pytest.param(jnp.mean, id="jnp.mean"),
            pytest.param(jnp.sum, id="jnp.sum"),
            pytest.param(np.asarray, id="np.asarray"),
            pytest.param(jnp.asarray, id="jnp.asarray"),
            pytest.param(lambda m: np.ones(B) * m, id="numpy-array-times"),
            pytest.param(lambda m: jnp.ones(B) * m, id="jax-array-times"),
        ],
    )
    def test_implicit_use_raises_type_error(self, misuse: Any) -> None:
        with pytest.raises(TypeError):
            misuse(depth())

    def test_array_conversion_names_the_explicit_reads(self) -> None:
        with pytest.raises(TypeError, match=r"value_or"):
            np.asarray(depth())

    def test_the_same_fields_compare_equal_as_specs(self) -> None:
        """Two descriptions of one field compare by their fields, as a spec comparison needs."""
        spec = Maybe(jax.ShapeDtypeStruct((4,), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_))
        same = Maybe(jax.ShapeDtypeStruct((4,), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_))
        other = Maybe(jax.ShapeDtypeStruct((5,), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_))

        assert spec == same
        assert spec != other
        assert hash(spec) == hash(same)


class TestValueOr:
    """The explicit read: the value where present, the fill where missing."""

    def test_under_jit_over_a_batch(self) -> None:
        field = depth()

        filled = jax.jit(lambda m: m.value_or(-1.0))(field)

        expected = np.where(PRESENT[:, None, None], np.asarray(field.value), -1.0)
        np.testing.assert_array_equal(filled, expected)
        assert filled.dtype == jnp.float32

    def test_per_record_under_vmap(self) -> None:
        field = depth()

        corner = jax.vmap(lambda m: m.value_or(-1.0)[0, 0])(field)

        np.testing.assert_array_equal(corner, np.where(PRESENT, field.value[:, 0, 0], -1.0))

    def test_an_array_fill_broadcasts_over_the_trailing_axes(self) -> None:
        field = depth()
        fill = jnp.arange(4, dtype=jnp.float32)

        filled = field.value_or(fill)

        np.testing.assert_array_equal(filled[1], np.broadcast_to(fill, (4, 4)))
        np.testing.assert_array_equal(filled[0], field.value[0])

    def test_present_must_lead_the_value_shape(self) -> None:
        field = Maybe(jnp.zeros((B, 4)), jnp.ones((B + 1,), bool))

        with pytest.raises(ValueError, match="present"):
            field.value_or(0.0)


class TestFillValueAndGradients:
    """Zeros, not NaN: a NaN fill hidden behind ``jnp.where`` still makes the gradient NaN."""

    @staticmethod
    def _loss(values: jax.Array, present: jax.Array) -> jax.Array:
        return jnp.sum(jnp.where(present[:, None, None], jnp.sqrt(values), 0.0))

    def test_a_zero_fill_keeps_the_gradient_finite(self) -> None:
        field = depth()

        grad = jax.grad(self._loss)(field.value + 1e-3, field.present)

        assert bool(jnp.isfinite(grad).all())
        np.testing.assert_array_equal(np.asarray(grad)[~PRESENT], 0.0)

    def test_a_nan_fill_makes_the_gradient_nan_while_the_loss_stays_finite(self) -> None:
        """The reason missing slots hold zeros: this failure is silent in the loss."""
        field = depth()
        nan_filled = jnp.where(field.present[:, None, None], field.value, jnp.nan)

        loss = self._loss(nan_filled, field.present)
        grad = jax.grad(self._loss)(nan_filled, field.present)

        assert bool(jnp.isfinite(loss))
        assert not bool(jnp.isfinite(grad).all())


class MaskTokens(OperatorModule):
    """Hides each token with probability 0.5 and keeps the tokens as the target."""

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        hidden = jax.random.bernoulli(require_key(key, self), 0.5, element.data["tokens"].shape)
        return element.update_state({MASKED: {"tokens": hidden}})


def test_a_masked_value_keeps_its_target() -> None:
    tokens = jnp.arange(B * 5, dtype=jnp.int32).reshape(B, 5)
    batch = name_records(batch_ops.from_arrays({"tokens": tokens}), jnp.arange(B), 0)
    op = MaskTokens(OperatorConfig(stochastic=True, stream_name="mask"), rngs=nnx.Rngs(mask=0))

    out = jax.jit(lambda b: op(b))(batch)

    np.testing.assert_array_equal(out.data["tokens"], tokens)
    hidden = np.asarray(out.states[MASKED]["tokens"])
    assert hidden.dtype == np.bool_ and hidden.shape == (B, 5)
    assert 0 < hidden.sum() < hidden.size
    # A masked-prediction loss reads the hidden positions' targets from data, unchanged.
    np.testing.assert_array_equal(
        np.asarray(out.data["tokens"])[hidden], np.asarray(tokens)[hidden]
    )


class CrossModalFill(OperatorModule):
    """Fills a missing image from the text and a missing text from the image, where the other is
    present; a record missing both stays missing. Marks what it filled in ``state[IMPUTED]``."""

    def __init__(self, *, rngs: nnx.Rngs, strategy: str = "vmap") -> None:
        super().__init__(OperatorConfig(stochastic=False, batch_strategy=strategy))
        self.text_to_image = nnx.Linear(T, H * W * C, rngs=rngs)
        self.image_to_text = nnx.Linear(H * W * C, T, rngs=rngs)

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        image, text = element.data["image"], element.data["text"]
        fill_image = ~image.present & text.present
        fill_text = ~text.present & image.present
        image_value = jnp.where(
            fill_image, self.text_to_image(text.value).reshape(H, W, C), image.value
        )
        text_value = jnp.where(fill_text, self.image_to_text(image_value.reshape(-1)), text.value)
        return element.replace(
            data={
                "image": Maybe(image_value, image.present | fill_image),
                "text": Maybe(text_value, text.present | fill_text),
            },
            state={IMPUTED: {"image": fill_image, "text": fill_text}},
        )


def paired_batch(seed: int, has_image: np.ndarray | None = None) -> Batch:
    """Paired image/text records, either of which may be missing (zeros in its slot)."""
    rng = np.random.default_rng(seed)
    if has_image is None:
        has_image = rng.random(B) < 0.6
    has_text = rng.random(B) < 0.6
    image = np.where(has_image[:, None, None, None], rng.random((B, H, W, C)), 0.0)
    text = np.where(has_text[:, None], rng.random((B, T)), 0.0)
    return batch_ops.from_arrays(
        {
            "image": Maybe(jnp.asarray(image, jnp.float32), jnp.asarray(has_image)),
            "text": Maybe(jnp.asarray(text, jnp.float32), jnp.asarray(has_text)),
        }
    )


def four_cases_batch() -> Batch:
    """Both present, image only, text only, neither: two records of each."""
    has_image = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    has_text = np.array([1, 1, 0, 0, 1, 1, 0, 0], bool)
    batch = paired_batch(0, has_image)
    text = batch.data["text"]
    return batch.replace(
        data={
            "image": batch.data["image"],
            "text": Maybe(
                jnp.where(has_text[:, None], text.value + 0.5, 0.0), jnp.asarray(has_text)
            ),
        }
    )


def _imputed_loss(model: CrossModalFill, batch: Batch) -> jax.Array:
    """A loss over every record's filled values; only imputed values depend on the filler."""
    out = model(batch)
    return jnp.sum(out.data["image"].value_or(0.0) ** 2) + jnp.sum(
        out.data["text"].value_or(0.0) ** 2
    )


class TestAFill:
    """An operator that fills a missing value sets ``present`` and marks ``IMPUTED``."""

    def test_the_four_presence_cases(self) -> None:
        batch = four_cases_batch()

        out = CrossModalFill(rngs=nnx.Rngs(0))(batch)

        image, text = out.data["image"], out.data["text"]
        imputed = out.states[IMPUTED]
        # both present: kept, nothing imputed
        np.testing.assert_array_equal(image.value[:2], batch.data["image"].value[:2])
        np.testing.assert_array_equal(text.value[:2], batch.data["text"].value[:2])
        # image only: text imputed
        np.testing.assert_array_equal(imputed["text"], [0, 0, 1, 1, 0, 0, 0, 0])
        assert not np.array_equal(text.value[2:4], batch.data["text"].value[2:4])
        # text only: image imputed
        np.testing.assert_array_equal(imputed["image"], [0, 0, 0, 0, 1, 1, 0, 0])
        assert not np.array_equal(image.value[4:6], batch.data["image"].value[4:6])
        # present afterwards everywhere but where both were missing
        np.testing.assert_array_equal(image.present, [1, 1, 1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(text.present, [1, 1, 1, 1, 1, 1, 0, 0])
        # neither: still missing, slots still zero
        np.testing.assert_array_equal(image.value[6:], 0.0)
        np.testing.assert_array_equal(text.value[6:], 0.0)

    def test_gradients_reach_the_filler_only_from_imputed_records(self) -> None:
        model = CrossModalFill(rngs=nnx.Rngs(0))
        batch = four_cases_batch()
        imputed_rows = np.array([2, 3, 4, 5])

        grads = nnx.grad(_imputed_loss)(model, batch)
        only_imputed = nnx.grad(_imputed_loss)(model, batch_ops.take(batch, imputed_rows))

        for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(only_imputed), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        assert all(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads))

    def test_no_gradient_reaches_the_filler_when_nothing_is_missing(self) -> None:
        model = CrossModalFill(rngs=nnx.Rngs(0))
        batch = paired_batch(1)
        complete = batch.replace(
            data={name: Maybe(field.value, jnp.ones(B, bool)) for name, field in batch.data.items()}
        )

        grads = nnx.grad(_imputed_loss)(model, complete)

        assert all(float(jnp.abs(g).sum()) == 0.0 for g in jax.tree.leaves(grads))

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_one_trace_over_batches_with_different_presence(self, graph: bool) -> None:
        model = CrossModalFill(rngs=nnx.Rngs(0))
        step = nnx.jit(
            lambda m, b: nnx.value_and_grad(_imputed_loss, graph=graph)(m, b), graph=graph
        )
        step(model, paired_batch(0))
        patterns = {tuple(np.asarray(paired_batch(s).data["image"].present)) for s in range(1, 6)}
        assert len(patterns) > 1

        with expect_compiles(0):
            for seed in range(1, 6):
                step(model, paired_batch(seed))

    @pytest.mark.parametrize("strategy", ["vmap", "scan"])
    def test_both_batch_strategies_fill_alike(self, strategy: str) -> None:
        batch = four_cases_batch()
        reference = CrossModalFill(rngs=nnx.Rngs(0))(batch)

        out = CrossModalFill(rngs=nnx.Rngs(0), strategy=strategy)(batch)

        for got, want in zip(jax.tree.leaves(out), jax.tree.leaves(reference), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-6)


def _depth_sum(batch: Batch) -> jax.Array:
    """What a step reads: the filled depth per record, plus how many records have one."""
    field = batch["depth"]
    return jnp.sum(field.value_or(0.0), axis=(1, 2)) + jnp.sum(field.present)


def _expected_depth_sum(batch: Batch) -> np.ndarray:
    value, present = np.asarray(batch["depth"].value), np.asarray(batch["depth"].present)
    return np.where(present, value.sum(axis=(1, 2)), 0.0) + present.sum()


class TestTransforms:
    """A batch carrying a ``Maybe`` through the transforms a step meets, against NumPy."""

    def test_lax_scan_over_a_chunk(self) -> None:
        batches = [batch_with_depth(np.roll(PRESENT, k), seed=k) for k in range(3)]
        chunk = batch_ops.stack(batches)

        _, per_batch = jax.lax.scan(lambda c, b: (c, _depth_sum(b)), None, chunk)

        for k, batch in enumerate(batches):
            np.testing.assert_allclose(per_batch[k], _expected_depth_sum(batch), rtol=1e-6)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_nnx_vmap_over_records(self, graph: bool) -> None:
        batch = batch_with_depth()

        @nnx.vmap(in_axes=0, graph=graph)
        def per_record(field: Maybe) -> jax.Array:
            return jnp.sum(field.value_or(-1.0))

        expected = np.where(PRESENT, np.asarray(batch["depth"].value).sum(axis=(1, 2)), -16.0)
        np.testing.assert_allclose(per_record(batch["depth"]), expected, rtol=1e-6)

    def test_jit_partial_checkpoint_and_cond(self) -> None:
        model = CrossModalFill(rngs=nnx.Rngs(0))
        batch = batch_with_depth()

        bound = nnx.jit_partial(lambda m, b: _depth_sum(b), model, graph=False)
        checkpointed = jax.jit(jax.checkpoint(_depth_sum))
        branched = jax.jit(lambda b, c: jax.lax.cond(c, _depth_sum, lambda x: -_depth_sum(x), b))

        expected = _expected_depth_sum(batch)
        np.testing.assert_allclose(bound(batch), expected, rtol=1e-6)
        np.testing.assert_allclose(checkpointed(batch), expected, rtol=1e-6)
        np.testing.assert_allclose(branched(batch, False), -expected, rtol=1e-6)

    def test_shard_map_over_a_data_mesh(self) -> None:
        mesh = Mesh(np.array(jax.devices()[:4]), ("data",))
        batch = batch_with_depth()
        specs = jax.tree.map(lambda _: P("data"), batch.replace(batch_state=None))
        specs = specs.replace(batch_state={})

        @jax.jit
        @jax.shard_map(mesh=mesh, in_specs=(specs,), out_specs=P())
        def present_count(shard: Batch) -> jax.Array:
            return jax.lax.psum(jnp.sum(shard["depth"].present, dtype=jnp.int32), "data")

        assert int(present_count(batch)) == int(PRESENT.sum())


class TestOnTheDevice:
    """Presence is placed with the batch; reading it inside a step transfers nothing."""

    @staticmethod
    def _step(batch: Batch) -> jax.Array:
        field = batch["depth"]
        return jnp.sum(field.value_or(0.0)) + jnp.sum(field.present)

    def test_a_jitted_step_reading_presence_transfers_nothing(self) -> None:
        step = jax.jit(self._step)
        placed = jax.device_put(batch_with_depth())
        step(placed)
        other = batch_with_depth(~PRESENT, seed=1)

        with jax.transfer_guard("disallow"), expect_compiles(0):
            total = step(other)

        assert np.isfinite(float(total))

    def test_the_transfer_guard_positive_control_raises(self) -> None:
        step = jax.jit(self._step)
        step(jax.device_put(batch_with_depth()))
        host = jax.tree.map(np.asarray, batch_with_depth())

        with jax.transfer_guard("disallow"), pytest.raises(RuntimeError, match="[Dd]isallowed"):
            step(host)


class TestSpecs:
    """A spec describes a ``Maybe`` field as a ``Maybe`` of two ``ShapeDtypeStruct``s."""

    @staticmethod
    def _element_spec() -> dict[str, Any]:
        return {
            "image": jax.ShapeDtypeStruct((2, 2), jnp.float32),
            "depth": Maybe(
                jax.ShapeDtypeStruct((4, 4), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_)
            ),
        }

    def test_batched_spec_lifts_value_and_present(self) -> None:
        spec = batched_spec(self._element_spec(), B)

        assert isinstance(spec["depth"], Maybe)
        assert spec["depth"].value.shape == (B, 4, 4)
        assert spec["depth"].present.shape == (B,)
        assert spec["depth"].present.dtype == jnp.bool_

    def test_a_batch_validates_against_its_spec(self) -> None:
        validate_batch(batch_with_depth().data, self._element_spec(), batch_size=B)

    def test_presence_of_the_wrong_dtype_is_named(self) -> None:
        data = batch_with_depth().data
        wrong = {
            **data,
            "depth": Maybe(data["depth"].value, data["depth"].present.astype(jnp.int8)),
        }

        with pytest.raises(SpecMismatchError, match=r"\['depth'\]\.present"):
            validate_batch(wrong, self._element_spec(), batch_size=B)

    def test_a_record_spec_reads_off_a_record(self) -> None:
        record = batch_ops.element(batch_with_depth(), 0)

        spec = jax.tree.map(array_to_spec, record.data)

        assert spec == self._element_spec()

    def test_an_operator_output_spec_by_eval_shape(self) -> None:
        model = CrossModalFill(rngs=nnx.Rngs(0))

        out = nnx.eval_shape(lambda m, b: m(b), model, paired_batch(0))

        assert isinstance(out.data["image"], Maybe)
        assert out.data["image"].present.shape == (B,)
        assert out.states[IMPUTED]["text"].dtype == jnp.bool_


def test_placement_splits_presence_with_its_rows() -> None:
    """Presence is a per-record leaf: the row sharding places it like the value."""
    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",))
    rows, replicated = NamedSharding(mesh, P("data")), NamedSharding(mesh, P())
    present = np.arange(len(devices) * 2) % 3 != 0
    batch = jax.tree.map(np.asarray, batch_with_depth(present))

    placed = place_batch_on_shards(batch, batch_ops.shardings(batch, rows, replicated))

    field = placed["depth"]
    assert isinstance(field, Maybe)
    assert field.value.sharding == rows and field.present.sharding == rows
    np.testing.assert_array_equal(field.present, present)
    np.testing.assert_array_equal(field.value, batch.data["depth"].value)
