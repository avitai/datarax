"""Tests for JAX process sharding aligned with Grain even-split semantics."""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from datarax.sharding.jax_process_sharder import (
    JaxProcessSharderConfig,
    JaxProcessSharderModule,
)


def _shard_for_process(data, *, process_index: int, drop_remainder: bool):
    with (
        patch("jax.process_count", return_value=3),
        patch("jax.process_index", return_value=process_index),
        patch("jax.local_device_count", return_value=1),
    ):
        sharder = JaxProcessSharderModule(JaxProcessSharderConfig(drop_remainder=drop_remainder))
    return sharder.shard_data(data)


def test_jax_process_sharder_even_split_with_remainder() -> None:
    """drop_remainder=False should distribute remainders across early processes."""
    expected_for_10 = [list(range(0, 4)), list(range(4, 7)), list(range(7, 10))]
    expected_for_11 = [list(range(0, 4)), list(range(4, 8)), list(range(8, 11))]

    for process_index, expected in enumerate(expected_for_10):
        assert (
            _shard_for_process(
                list(range(10)),
                process_index=process_index,
                drop_remainder=False,
            )
            == expected
        )

    for process_index, expected in enumerate(expected_for_11):
        assert (
            _shard_for_process(
                list(range(11)),
                process_index=process_index,
                drop_remainder=False,
            )
            == expected
        )


def test_jax_process_sharder_drop_remainder_keeps_equal_splits() -> None:
    """drop_remainder=True should keep equal shards and drop trailing remainder."""
    expected_for_10 = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9))]

    for process_index, expected in enumerate(expected_for_10):
        assert (
            _shard_for_process(
                list(range(10)),
                process_index=process_index,
                drop_remainder=True,
            )
            == expected
        )


def test_jax_process_sharder_array_even_split_matches_list_boundaries() -> None:
    """Array sharding should use the same bounds as list sharding."""
    array = np.arange(11)
    jax_array = jnp.arange(11)

    np.testing.assert_array_equal(
        _shard_for_process(array, process_index=1, drop_remainder=False),
        np.array([4, 5, 6, 7]),
    )
    np.testing.assert_array_equal(
        np.asarray(_shard_for_process(jax_array, process_index=2, drop_remainder=False)),
        np.array([8, 9, 10]),
    )


def test_shard_bounds_match_grain_even_split_exactly() -> None:
    """SH1: datarax shard bounds must equal Grain's even_split for all topologies.

    The interval math (`even_split`) is private in Grain, so datarax keeps its
    own implementation; this test pins it to Grain's real algorithm across many
    (num_examples, shard_count, shard_index, drop_remainder) combinations,
    including drop_remainder=False (the case Grain 0.2.18 fixed).
    """
    from grain._src.core.sharding import even_split, ShardOptions

    for num_examples in (0, 1, 7, 10, 11, 100, 101):
        for shard_count in (1, 2, 3, 4):
            for shard_index in range(shard_count):
                for drop_remainder in (True, False):
                    options = ShardOptions(
                        shard_index=shard_index,
                        shard_count=shard_count,
                        drop_remainder=drop_remainder,
                    )
                    expected = even_split(num_examples, options)

                    with (
                        patch("jax.process_count", return_value=shard_count),
                        patch("jax.process_index", return_value=shard_index),
                        patch("jax.local_device_count", return_value=1),
                    ):
                        sharder = JaxProcessSharderModule(
                            JaxProcessSharderConfig(drop_remainder=drop_remainder)
                        )
                        actual = sharder._shard_bounds(num_examples)

                    assert actual == expected, (
                        f"mismatch for num={num_examples} count={shard_count} "
                        f"index={shard_index} drop={drop_remainder}: "
                        f"{actual} != {expected}"
                    )
