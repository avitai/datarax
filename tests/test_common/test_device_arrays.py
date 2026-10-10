"""``arrays_made_since`` reports the device arrays made in a window, whatever a collection frees.

A check that a call makes no device array compares the live arrays before and after it. A garbage
collection that runs inside the call frees arrays the call never touched (an nnx transform leaves
its flattened Variables in a reference cycle until the cyclic collector runs), so comparing the
two live sets for equality fails on arrays freed, not made. The snapshot is held, so nothing in it
is freed and no id in it is reused by a new array.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest

from tests.test_common.device_arrays import arrays_made_since


@pytest.fixture
def collector_held() -> Iterator[None]:
    """The cyclic collector runs only where a test calls ``gc.collect()``."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def _drop_a_cycle_holding_an_array() -> weakref.ref[jax.Array]:
    """An unreachable reference cycle that alone keeps a device array alive."""
    array = jnp.zeros(3)
    cycle: list[object] = [array]
    cycle.append(cycle)
    return weakref.ref(array)


@pytest.mark.usefixtures("collector_held")
def test_an_array_a_collection_frees_in_the_window_is_not_reported_as_made() -> None:
    freed = _drop_a_cycle_holding_an_array()
    before = jax.live_arrays()
    assert any(array is freed() for array in before), "the cycle's array is live at the snapshot"

    gc.collect()  # inside the window: frees the cycle

    assert arrays_made_since(before) == []
    del before
    gc.collect()
    assert freed() is None, "control: only the cycle and the snapshot kept the array alive"


@pytest.mark.usefixtures("collector_held")
def test_an_array_made_in_the_window_is_reported() -> None:
    _drop_a_cycle_holding_an_array()
    before = jax.live_arrays()

    gc.collect()
    made = jnp.ones(3)

    assert sum(array is made for array in arrays_made_since(before)) == 1


def test_an_array_made_before_the_snapshot_is_not_reported() -> None:
    kept = jnp.ones(3)
    before = jax.live_arrays()

    assert all(array is not kept for array in arrays_made_since(before))
