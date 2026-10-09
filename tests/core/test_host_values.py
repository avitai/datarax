"""Host values: what sources keep on the host, out of NNX state and out of every program.

The holders live in ``datarax.core.host_values``, beside the other contract types sources and the
pipeline share, so the pipeline reads them without importing a source module.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from flax import nnx

from datarax.core.host_values import HostArray, HostArrays, HostProvenance, HostValue


@pytest.mark.parametrize("holder", [HostValue, HostProvenance, HostArray, HostArrays])
def test_each_holder_is_defined_once_in_core(holder: type) -> None:
    assert holder.__module__ == "datarax.core.host_values"


def _npy_map(path) -> HostArray:
    """A map of the array a ``.npy`` file stores: its offset is the header's length."""
    stored = np.load(path, mmap_mode="r")
    return HostArray(
        str(path), offset=int(stored.offset), dtype=stored.dtype.str, shape=stored.shape
    )


def test_host_values_of_a_kind_are_equal_whatever_they_hold() -> None:
    assert HostValue(1) == HostValue("other")
    assert hash(HostValue(1)) == hash(HostValue([1, 2]))
    assert HostProvenance(()) == HostProvenance(({"id": "a"},))


def test_host_values_of_different_kinds_differ() -> None:
    assert HostValue(()) != HostProvenance(())


def test_a_host_array_is_equal_only_to_itself(tmp_path) -> None:
    path = tmp_path / "rows.npy"
    np.save(path, np.arange(6, dtype=np.float32))
    one, other = _npy_map(path), _npy_map(path)
    assert one == one
    assert one != other
    assert HostArrays((one,)) != HostArrays((one,))


def test_a_host_array_opens_its_map_at_first_use_and_pickles_as_its_place_on_disk(tmp_path) -> None:
    path = tmp_path / "rows.npy"
    np.save(path, np.arange(6, dtype=np.float32))
    held = _npy_map(path)
    np.testing.assert_array_equal(held.array, np.arange(6, dtype=np.float32))
    copy = pickle.loads(pickle.dumps(held))  # the holder this test built, as a worker receives it
    assert copy.path == str(path)
    assert copy.__getstate__() == {
        "path": str(path),
        "offset": held.offset,
        "dtype": "<f4",
        "shape": (6,),
    }
    np.testing.assert_array_equal(copy.array, held.array)


class _Holding(nnx.Module):
    def __init__(self, held: object) -> None:
        self.held = held
        self.weight = nnx.Param(np.zeros((2,), np.float32))


def test_a_holder_is_static_to_nnx_and_left_out_of_state() -> None:
    module = _Holding(HostValue(np.arange(4)))
    state_leaves = [leaf for _, leaf in nnx.to_flat_state(nnx.state(module))]
    assert len(state_leaves) == 1
    assert nnx.graphdef(module) == nnx.graphdef(_Holding(HostValue("anything")))
