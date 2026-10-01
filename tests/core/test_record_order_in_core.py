"""The record order lives in core: ``datarax.core.index_shuffle`` provides it, samplers do not.

Sources and the pipeline name records through the keyed order, and core sits below samplers in
the layered architecture, so the order is a core module that samplers, sources and core import.
"""

from __future__ import annotations

import importlib
import importlib.util

import jax
import numpy as np

from datarax.core.index_words import from_words, to_words


def test_core_provides_the_record_order() -> None:
    order = importlib.import_module("datarax.core.index_shuffle")
    length, seed, epoch = 37, 5, 2
    positions = np.arange(length, dtype=np.uint64)
    host = order.shuffle_positions_host(positions, length, seed, epoch)
    device = order.shuffle_positions(
        to_words(positions), length, jax.random.fold_in(jax.random.key(seed), epoch)
    )
    assert sorted(int(record) for record in host) == list(range(length))
    np.testing.assert_array_equal(from_words(np.asarray(device)), host)
    assert [order.index_shuffle(p, seed, length, epoch) for p in range(length)] == list(host)


def test_the_samplers_package_holds_no_record_order() -> None:
    assert importlib.util.find_spec("datarax.samplers.index_shuffle") is None
