"""The ``Element`` and ``Batch`` contract: frozen registered dataclasses of arrays.

A record is an ``Element`` and a batch of records a ``Batch``. Every field is a pytree child, so
both pass through every JAX and Flax NNX transform, a batch's values never enter the treedef, and
two batches differing only in their values share one compiled program.
"""

import dataclasses
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from substrax.testing.compiles import expect_compiles

from datarax.core.element_batch import Batch, Element, PADDING_INDEX


B = 8


def make_batch(seed: int = 0, size: int = B, epoch: int = 0) -> Batch:
    """A batch with nested data, per-record state, identities and a 0-d batch-level leaf."""
    rng = np.random.default_rng(seed)
    return Batch(
        {
            "image": jnp.asarray(rng.standard_normal((size, 4, 4, 3)), jnp.float32),
            "text": {"tokens": jnp.asarray(rng.integers(0, 50, (size, 6)), jnp.int32)},
            "label": jnp.asarray(rng.integers(0, 10, (size,)), jnp.int32),
        },
        states={"score": jnp.asarray(rng.standard_normal(size), jnp.float32)},
        indices=jnp.stack(
            [jnp.zeros(size, jnp.uint32), jnp.arange(size, dtype=jnp.uint32) + seed * size], -1
        ),
        epochs=jnp.full((size,), epoch, jnp.int32),
        draws=jnp.zeros((size,), jnp.int32),
        batch_state={"seen": jnp.zeros((), jnp.int32)},
    )


def row_axes() -> Batch:
    """A ``vmap`` prefix: every per-record field on axis 0, the batch-level state broadcast."""
    axes: Batch = jax.tree.map(lambda _: 0, make_batch().replace(batch_state=None))
    return axes


class _Linear(nnx.Module):
    """A one-layer model reading a batch the way flax's examples read a dict."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(48, 10, rngs=rngs)

    def __call__(self, image: jax.Array) -> jax.Array:
        return self.dense(image.reshape(image.shape[0], -1))


def _loss(model: _Linear, batch: Batch) -> jax.Array:
    """A flax-example loss: the step reads ``batch["image"]`` as it would read a dict."""
    logits = model(batch["image"])
    labels = batch["label"]
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(batch.batch_size), labels])


def _expected_loss(model: _Linear, batch: Batch) -> jax.Array:
    """The same loss over the plain data dict: the reference every transform reproduces."""
    logits = model(batch.data["image"])
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(B), batch.data["label"]])


class TestLayout:
    """Frozen, slotted, registered dataclasses whose every field is a child."""

    def test_a_batch_is_frozen_and_slotted(self) -> None:
        batch = make_batch()

        with pytest.raises(dataclasses.FrozenInstanceError):
            batch.data = {}  # pyright: ignore[reportAttributeAccessIssue]
        assert not hasattr(batch, "__dict__")

    def test_an_element_is_frozen_and_slotted(self) -> None:
        element = Element({"x": jnp.ones(3)})

        with pytest.raises(dataclasses.FrozenInstanceError):
            element.data = {}  # pyright: ignore[reportAttributeAccessIssue]
        assert not hasattr(element, "__dict__")

    def test_neither_is_an_nnx_module_or_pytree(self) -> None:
        assert not isinstance(make_batch(), nnx.Module | nnx.Pytree)
        assert not isinstance(Element({"x": jnp.ones(3)}), nnx.Module | nnx.Pytree)

    def test_fields_after_data_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            Batch({}, {}, None, None, None, {})  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            Element({}, {})  # pyright: ignore[reportCallIssue]

    def test_every_field_is_a_leaf_and_nothing_is_aux(self) -> None:
        """Two batches differing in every value have one treedef: values never key a cache."""
        first, second = make_batch(0), make_batch(1, epoch=3)

        assert jax.tree.structure(first) == jax.tree.structure(second)
        # image, tokens, label, score, indices, epochs, draws, seen
        assert len(jax.tree.leaves(first)) == 8

    def test_an_element_defaults_to_no_state_no_index_and_epoch_zero(self) -> None:
        element = Element({"x": jnp.ones(3)})

        assert element.state == {}
        assert element.index is None
        assert element.epoch.dtype == np.int32 and int(element.epoch) == 0
        assert element.draw.dtype == np.int32 and int(element.draw) == 0

    def test_the_padding_index_is_all_ones_in_both_words(self) -> None:
        assert PADDING_INDEX.dtype == np.uint32
        assert np.array_equal(PADDING_INDEX, [2**32 - 1, 2**32 - 1])


class TestUpdates:
    """Updates return new objects and leave the original as it was."""

    def test_batch_replace_returns_a_new_batch(self) -> None:
        batch = make_batch()
        states = {"score": jnp.zeros(B)}

        replaced = batch.replace(states=states)

        assert replaced.states is states
        assert batch.states is not states
        assert replaced.data is batch.data

    def test_element_replace_update_data_and_update_state(self) -> None:
        element = Element({"x": jnp.ones(3), "y": jnp.zeros(2)}, state={"a": jnp.ones(())})

        with_data = element.update_data({"x": jnp.full(3, 2.0)})
        with_state = element.update_state({"b": jnp.zeros(())})
        moved = element.replace(epoch=np.array(4, np.int32))

        assert float(with_data.data["x"][0]) == 2.0 and "y" in with_data.data
        assert set(with_state.state) == {"a", "b"}
        assert int(moved.epoch) == 4
        assert float(element.data["x"][0]) == 1.0 and set(element.state) == {"a"}

    def test_update_data_refuses_data_that_is_not_a_mapping(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            Element(jnp.ones(3)).update_data({"x": jnp.ones(3)})


class TestReads:
    """``[name]``, ``get`` and ``in`` read ``data``; nothing else pretends to be a mapping."""

    def test_reads_delegate_to_data(self) -> None:
        batch = make_batch()
        sentinel = jnp.zeros(())

        assert batch["image"] is batch.data["image"]
        assert batch["text"]["tokens"] is batch.data["text"]["tokens"]
        assert batch.get("label") is batch.data["label"]
        assert batch.get("missing") is None
        assert batch.get("missing", sentinel) is sentinel
        assert "image" in batch and "score" not in batch and "missing" not in batch

    def test_batch_size_is_the_static_record_count(self) -> None:
        batch = make_batch()

        @jax.jit
        def size(batch: Batch) -> jax.Array:
            assert isinstance(batch.batch_size, int)
            return jnp.asarray(batch.batch_size)

        assert batch.batch_size == B
        assert int(size(batch)) == B

    @pytest.mark.parametrize(
        "misuse",
        [
            pytest.param(lambda b: b[0], id="batch[0]"),
            pytest.param(lambda b: dict(b), id="dict(batch)"),
            pytest.param(lambda b: {**b}, id="{**batch}"),
            pytest.param(lambda b: list(b), id="list(batch)"),
            pytest.param(lambda b: len(b), id="len(batch)"),
        ],
    )
    def test_mapping_misuse_raises_instead_of_dropping_fields(
        self, misuse: Callable[[Batch], object]
    ) -> None:
        """Full delegation would return only the data keys and silently drop the rest."""
        with pytest.raises(TypeError):
            misuse(make_batch())


class TestTransforms:
    """Every transform the Trainer and the operators meet, each checked against a reference."""

    def test_jax_jit(self) -> None:
        model = _Linear(rngs=nnx.Rngs(0))
        batch = make_batch()
        graphdef, params = nnx.split(model)

        loss = jax.jit(lambda p, b: _loss(nnx.merge(graphdef, p), b))(params, batch)

        np.testing.assert_allclose(loss, _expected_loss(model, batch), rtol=1e-6)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_nnx_jit_and_grad_in_graph_and_tree_mode(self, graph: bool) -> None:
        model = _Linear(rngs=nnx.Rngs(0))
        batch = make_batch()

        @nnx.jit(graph=graph)
        def step(model: _Linear, batch: Batch) -> tuple[jax.Array, nnx.State]:
            return nnx.value_and_grad(_loss, graph=graph)(model, batch)

        loss, grads = step(model, batch)
        expected, expected_grads = nnx.value_and_grad(_expected_loss, graph=graph)(model, batch)

        np.testing.assert_allclose(loss, expected, rtol=1e-6)
        for got, want in zip(jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True):
            # A few float32 ULPs at the gradient's scale: jitted and eager reduce in other orders.
            bound = 16 * float(jnp.finfo(jnp.float32).eps) * float(jnp.max(jnp.abs(want)))
            np.testing.assert_allclose(got, want, rtol=0, atol=bound)

    def test_nnx_jit_partial_binds_the_model_and_takes_the_batch(self) -> None:
        """The Trainer's binding refuses a Variable in the runtime arguments; a Batch has none."""
        model = _Linear(rngs=nnx.Rngs(0))
        batch = make_batch()

        step = nnx.jit_partial(_loss, model, graph=False)

        np.testing.assert_allclose(step(batch), _expected_loss(model, batch), rtol=1e-6)

    def test_jax_vmap_over_a_batch_prefix(self) -> None:
        batch = make_batch()

        def per_record(row: Batch) -> jax.Array:
            return jnp.sum(row["image"]) + row.indices[1].astype(jnp.float32) + row.epochs

        out = jax.vmap(per_record, in_axes=(row_axes(),))(batch)

        expected = jnp.sum(batch.data["image"], axis=(1, 2, 3)) + jnp.arange(B)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_nnx_vmap_over_a_batch_prefix(self, graph: bool) -> None:
        batch = make_batch()

        @nnx.vmap(in_axes=(row_axes(),), graph=graph)
        def per_record(row: Batch) -> jax.Array:
            return jnp.sum(row["image"]) * 2.0

        np.testing.assert_allclose(
            per_record(batch), 2.0 * jnp.sum(batch.data["image"], axis=(1, 2, 3)), rtol=1e-6
        )

    def test_lax_scan_over_stacked_batches(self) -> None:
        """A ``(K, B, ...)`` chunk scans one batch per step, batch-level leaves included."""
        batches = [make_batch(k) for k in range(3)]
        chunk = jax.tree.map(lambda *x: jnp.stack(x), *batches)

        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
            value = jnp.sum(batch["image"]) + batch.batch_state["seen"]
            return total + value, value

        total, per_batch = jax.lax.scan(body, jnp.zeros(()), chunk)

        expected = jnp.stack([jnp.sum(b.data["image"]) for b in batches])
        np.testing.assert_allclose(per_batch, expected, rtol=1e-5)
        np.testing.assert_allclose(total, expected.sum(), rtol=1e-5)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_nnx_scan_over_stacked_batches(self, graph: bool) -> None:
        model = _Linear(rngs=nnx.Rngs(0))
        batches = [make_batch(k) for k in range(3)]
        chunk = jax.tree.map(lambda *x: jnp.stack(x), *batches)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0), graph=graph)
        def body(model: _Linear, batch: Batch) -> tuple[_Linear, jax.Array]:
            return model, _loss(model, batch)

        _, losses = body(model, chunk)

        expected = jnp.stack([_expected_loss(model, b) for b in batches])
        np.testing.assert_allclose(losses, expected, rtol=1e-5)

    def test_checkpoint_under_grad_of_a_named_float_leaf(self) -> None:
        batch = make_batch()

        def loss(image: jax.Array) -> jax.Array:
            return jnp.sum(jnp.sin(batch.replace(data={**batch.data, "image": image})["image"]))

        grad = jax.grad(jax.checkpoint(loss))(batch.data["image"])

        np.testing.assert_allclose(grad, jnp.cos(batch.data["image"]), rtol=1e-6)

    def test_cond_returns_a_batch_from_either_branch(self) -> None:
        batch = make_batch()

        def flip(b: Batch) -> Batch:
            return b.replace(data={**b.data, "image": b.data["image"][:, ::-1]})

        taken = jax.jit(lambda b, c: jax.lax.cond(c, flip, lambda x: x, b))

        np.testing.assert_array_equal(
            taken(batch, True).data["image"], batch.data["image"][:, ::-1]
        )
        np.testing.assert_array_equal(taken(batch, False).data["image"], batch.data["image"])

    def test_shard_map_over_a_data_mesh(self) -> None:
        """Rows split over the data axis; the batch-level leaf replicated on every device."""
        mesh = Mesh(np.array(jax.devices()[:4]), ("data",))
        batch = make_batch()
        row = P("data")
        rows: Batch = jax.tree.map(lambda _: row, batch.replace(batch_state=None))
        specs = rows.replace(batch_state=jax.tree.map(lambda _: P(), batch.batch_state))

        @jax.jit
        @jax.shard_map(mesh=mesh, in_specs=(specs,), out_specs=P())
        def total(shard: Batch) -> jax.Array:
            local = jnp.sum(shard["image"]) + shard.batch_state["seen"].astype(jnp.float32)
            return jax.lax.psum(local, "data")

        np.testing.assert_allclose(total(batch), jnp.sum(batch.data["image"]), rtol=1e-5)
        placed = jax.device_put(batch, jax.tree.map(lambda s: NamedSharding(mesh, s), specs))
        assert placed.indices.sharding.spec == row


class TestCompiles:
    """A batch's values are never part of what a compiled program is keyed on."""

    def test_batches_differing_in_values_share_one_program(self) -> None:
        model = _Linear(rngs=nnx.Rngs(0))
        step = nnx.jit(_loss)
        step(model, make_batch(0))

        with expect_compiles(0):
            for seed in range(1, 4):
                step(model, make_batch(seed, epoch=seed))


def _median_launch_us(fn: Callable[[object], jax.Array], arg: object, calls: int = 200) -> float:
    """Median host time, in microseconds, to launch ``fn(arg)`` without waiting for the device."""
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e6


def _launch_ratio(
    fn: Callable[[object], jax.Array],
    arg: object,
    reference: Callable[[object], jax.Array],
    reference_arg: object,
    rounds: int = 9,
) -> tuple[float, float, float]:
    """Median over rounds of ``fn``'s launch time over ``reference``'s, the two timed in turn.

    Each round times both back to back, alternating which goes first, so a burst of load on the
    machine lands on both sides of one ratio instead of on one side of the comparison.

    Returns:
        The median ratio, and the median launch times of ``fn`` and ``reference`` in microseconds.
    """
    ratios, fn_us, reference_us = [], [], []
    for round_index in range(rounds):
        if round_index % 2:
            reference_time = _median_launch_us(reference, reference_arg)
            fn_time = _median_launch_us(fn, arg)
        else:
            fn_time = _median_launch_us(fn, arg)
            reference_time = _median_launch_us(reference, reference_arg)
        ratios.append(fn_time / reference_time)
        fn_us.append(fn_time)
        reference_us.append(reference_time)
    return statistics.median(ratios), statistics.median(fn_us), statistics.median(reference_us)


def _batch_and_dict_steps() -> tuple[
    Callable[[object], jax.Array], Batch, Callable[[object], jax.Array], dict[str, jax.Array]
]:
    """A jitted step over a Batch and one over a dict of the same leaves, both compiled."""
    batch = make_batch()
    as_dict = {f"leaf{i}": leaf for i, leaf in enumerate(jax.tree.leaves(batch))}
    step_batch = jax.jit(lambda b: jnp.sum(b["image"]))
    step_dict = jax.jit(lambda d: jnp.sum(d["leaf0"]))
    step_batch(batch).block_until_ready()
    step_dict(as_dict).block_until_ready()
    return step_batch, batch, step_dict, as_dict


@pytest.mark.performance
def test_launch_cost_is_within_noise_of_a_dict_with_the_same_leaves() -> None:
    """Launch cost follows leaf count, not the container: a Batch launches like a dict of 8.

    Both are timed the same way in one process, so the bound is a ratio: a coverage tracer
    slows both alike.
    """
    step_batch, batch, step_dict, as_dict = _batch_and_dict_steps()

    with expect_compiles(0):
        ratio, batch_us, dict_us = _launch_ratio(step_batch, batch, step_dict, as_dict)

    assert ratio <= 1.5, f"Batch {batch_us:.1f} us vs dict {dict_us:.1f} us (ratio {ratio:.2f})"


@pytest.mark.performance
def test_the_launch_ratio_detects_a_slower_launch() -> None:
    """The bound above can fail: a launch that does 50 us more host work exceeds it."""
    step_batch, batch, step_dict, as_dict = _batch_and_dict_steps()

    def slower_step(b: object) -> jax.Array:
        deadline = time.perf_counter() + 50e-6
        while time.perf_counter() < deadline:
            pass
        return step_batch(b)

    with expect_compiles(0):
        ratio, _, _ = _launch_ratio(slower_step, batch, step_dict, as_dict)

    assert ratio > 1.5
