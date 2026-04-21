"""Grain-backed rebatching contracts."""

from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.dag.nodes import RebatchNode
from datarax.dag.nodes.rebatch import rebatch_iterable


def _input_batches() -> list[dict[str, jax.Array]]:
    return [
        {"x": jnp.array([0, 1])},
        {"x": jnp.array([2, 3])},
        {"x": jnp.array([4])},
    ]


def test_rebatch_iterable_emits_target_batches() -> None:
    dataset = rebatch_iterable(_input_batches(), target_batch_size=3)

    batches = list(dataset)

    assert [batch["x"].shape[0] for batch in batches] == [3, 2]
    np.testing.assert_array_equal(batches[0]["x"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(batches[1]["x"], np.array([3, 4]))


def test_rebatch_iterable_drops_remainder() -> None:
    dataset = rebatch_iterable(
        _input_batches(),
        target_batch_size=3,
        drop_remainder=True,
    )

    batches = list(dataset)

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0]["x"], np.array([0, 1, 2]))


def test_rebatch_iterable_pads_remainder_with_grain_batch_and_pad() -> None:
    dataset = rebatch_iterable(
        _input_batches(),
        target_batch_size=3,
        pad=True,
        pad_value=0,
    )

    batches = list(dataset)

    assert [batch["x"].shape[0] for batch in batches] == [3, 3]
    np.testing.assert_array_equal(batches[1]["x"], np.array([3, 4, 0]))


def test_rebatch_iterable_rejects_one_shot_python_iterators() -> None:
    iterator: Iterator[dict[str, jax.Array]] = iter(_input_batches())

    with pytest.raises(TypeError, match="Grain IterDataset or Sequence"):
        rebatch_iterable(iterator, target_batch_size=3)


def test_rebatch_node_fast_mode_is_removed() -> None:
    with pytest.raises(ValueError, match="fast rebatching moved outside RebatchNode"):
        RebatchNode(target_batch_size=3, mode=cast(Any, "fast"))
