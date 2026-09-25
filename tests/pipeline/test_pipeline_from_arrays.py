"""``Pipeline.from_arrays``: a pipeline over in-memory arrays in one call.

It is the two-step construction written once -- a ``MemorySource`` over the arrays and a
``Pipeline`` with no stages -- so a caller holding a dict of arrays needs neither, and every
guarantee of a pipeline (one permutation per epoch, the final-batch rule, resumable position,
one compiled step) holds for it unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from datarax import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


_N = 10
_BATCH = 4


def _data() -> dict[str, np.ndarray]:
    return {
        "x": np.arange(2 * _N, dtype=np.float32).reshape(_N, 2),
        "y": np.arange(_N, dtype=np.int32),
    }


def _two_step(*, seed: int, shuffle: bool, drop_last: bool = False) -> Pipeline:
    source = MemorySource(MemorySourceConfig(shuffle=shuffle), data=_data(), rngs=nnx.Rngs(seed))
    return Pipeline(
        source=source, stages=[], batch_size=_BATCH, rngs=nnx.Rngs(seed), drop_last=drop_last
    )


def _epoch(pipeline: Pipeline, key: str = "y") -> list[np.ndarray]:
    batches = [np.asarray(batch[key]) for batch in pipeline]
    pipeline.reset()
    return batches


def test_it_serves_the_batches_the_two_step_construction_serves() -> None:
    one_call = Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=3, shuffle=True)
    two_step = _two_step(seed=3, shuffle=True)

    for _ in range(2):  # the second epoch takes a new permutation in both
        for ours, theirs in zip(_epoch(one_call, "x"), _epoch(two_step, "x"), strict=True):
            np.testing.assert_array_equal(ours, theirs)


def test_without_shuffle_the_records_come_in_order() -> None:
    pipeline = Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=0)

    np.testing.assert_array_equal(np.concatenate(_epoch(pipeline)), np.arange(_N))


def test_a_shuffled_epoch_is_a_permutation_and_the_seed_chooses_it() -> None:
    first = np.concatenate(
        _epoch(Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=1, shuffle=True))
    )
    second = np.concatenate(
        _epoch(Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=2, shuffle=True))
    )

    np.testing.assert_array_equal(np.sort(first), np.arange(_N))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("drop_last", "num_epochs", "sizes"),
    [
        (False, 1, [4, 4, 2]),
        (True, 1, [4, 4]),
        (False, 2, [4, 4, 4, 4, 4]),
    ],
)
def test_the_final_batch_rule_and_the_epoch_count_reach_the_pipeline(
    drop_last: bool, num_epochs: int, sizes: list[int]
) -> None:
    pipeline = Pipeline.from_arrays(
        _data(), batch_size=_BATCH, seed=0, drop_last=drop_last, num_epochs=num_epochs
    )

    assert [len(batch) for batch in _epoch(pipeline)] == sizes


def test_its_position_resumes_where_it_was_saved() -> None:
    pipeline = Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=5, shuffle=True)
    session = pipeline.session()
    next(session)
    state = pipeline.get_state()

    restored = Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=5, shuffle=True)
    restored.set_state(state)

    np.testing.assert_array_equal(
        np.asarray(next(restored.session())["x"]), np.asarray(next(session)["x"])
    )


def test_a_training_step_over_it_compiles_once() -> None:
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    traces = {"count": 0}

    @nnx.jit
    def step(model: nnx.Linear, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]) -> jax.Array:
        traces["count"] += 1

        def loss(model: nnx.Linear) -> jax.Array:
            return jnp.mean((model(batch["x"])[:, 0] - batch["y"]) ** 2)

        value, grads = nnx.value_and_grad(loss)(model)
        optimizer.update(model, grads)
        return value

    pipeline = Pipeline.from_arrays(
        _data(), batch_size=_BATCH, seed=0, shuffle=True, drop_last=True
    )
    for _ in range(3):
        for batch in pipeline:
            step(model, optimizer, batch)
        pipeline.reset()

    assert traces["count"] == 1


def test_it_scans_an_epoch_with_modules() -> None:
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    pipeline = Pipeline.from_arrays(_data(), batch_size=_BATCH, seed=0, drop_last=True)

    def step(model: nnx.Linear, batch: dict[str, jax.Array]) -> jax.Array:
        return jnp.sum(model(batch["x"]))

    outputs = pipeline.scan(step, length=2, modules=(model,))

    assert outputs.shape == (2,)


def test_arrays_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match=r"(?i)length|size"):
        Pipeline.from_arrays({"x": np.zeros((4, 2)), "y": np.zeros(3)}, batch_size=_BATCH, seed=0)
