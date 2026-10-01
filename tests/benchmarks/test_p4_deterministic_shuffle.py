"""P4: Deterministic multi-worker shuffling target tests.

Target: Same epoch output regardless of worker count. A shuffled epoch's order is the one the
pipeline's epoch key selects (``record_indices_at(start, size, key)``); worker ``k`` of ``n``
serves positions ``[k::n]`` of it.
"""

import jax
import numpy as np
import pytest

from datarax.core.index_shuffle import index_shuffle
from datarax.core.index_words import from_words
from datarax.sources import MemorySource, MemorySourceConfig


_KEY = jax.random.key(42)


def _epoch(length: int, *, num_workers: int = 1, shard_id: int | None = None) -> list[int]:
    """The records one worker serves in the epoch the key orders."""
    config = MemorySourceConfig(num_workers=num_workers, shard_id=shard_id)
    source = MemorySource(config, {"x": np.arange(length)})
    return from_words(source.record_indices_at(0, len(source), _KEY)).astype(int).tolist()


@pytest.mark.benchmark
class TestP4MultiWorkerPartitioning:
    """P4: Multi-worker partitioning via num_workers/shard_id."""

    def test_the_epoch_is_a_shuffled_permutation(self):
        order = _epoch(500)
        assert sorted(order) == list(range(500)), "Shuffle lost or duplicated elements"
        assert order != list(range(500)), "Shuffle produced identity permutation"

    def test_worker_partitions_cover_all_elements(self):
        """Verify all elements are covered across 4 workers (no gaps)."""
        all_values: list[int] = []
        for worker_id in range(4):
            all_values.extend(_epoch(100, num_workers=4, shard_id=worker_id))

        assert sorted(all_values) == list(range(100)), "Workers missed some elements"

    def test_worker_partitions_are_disjoint(self):
        """Verify no element appears in more than one worker's partition."""
        seen: set[int] = set()

        for worker_id in range(4):
            worker_values = set(_epoch(200, num_workers=4, shard_id=worker_id))

            overlap = seen & worker_values
            assert not overlap, f"Worker {worker_id} overlaps with previous workers: {overlap}"
            seen |= worker_values

    def test_single_worker_equals_no_partitioning(self):
        """Verify num_workers=1 produces the same output as default (no partitioning)."""
        assert _epoch(500) == _epoch(500, num_workers=1, shard_id=0)

    def test_worker_count_invariant_global_order(self):
        """Verify that the global shuffled order is the same regardless of worker count.

        Collecting all workers' outputs (in order) and interleaving by global
        position must reconstruct the same global permutation.
        """
        # Get global order from single worker
        global_order = _epoch(120, num_workers=1, shard_id=0)

        # Get partitioned orders from 4 workers and reconstruct global order
        worker_orders = [_epoch(120, num_workers=4, shard_id=worker_id) for worker_id in range(4)]

        # Reconstruct: worker k gets global positions [k::4]
        reconstructed = [0] * 120
        for worker_id, order in enumerate(worker_orders):
            for local_idx, value in enumerate(order):
                global_idx = worker_id + local_idx * 4
                reconstructed[global_idx] = value

        assert reconstructed == global_order

    def test_uneven_partition_handles_remainder(self):
        """Verify partitioning works when N is not divisible by num_workers."""
        all_values: list[int] = []  # 103 % 4 != 0
        for worker_id in range(4):
            all_values.extend(_epoch(103, num_workers=4, shard_id=worker_id))

        assert sorted(all_values) == list(range(103))


class TestFeistelIndexShuffle:
    """Test the Feistel cipher index_shuffle utility directly."""

    def test_is_permutation(self):
        """Verify index_shuffle produces a valid permutation."""
        n = 100
        shuffled = [index_shuffle(i, 42, n) for i in range(n)]
        assert sorted(shuffled) == list(range(n)), "Not a valid permutation"

    def test_deterministic(self):
        """Verify same seed produces same permutation."""
        n = 200
        perm1 = [index_shuffle(i, 123, n) for i in range(n)]
        perm2 = [index_shuffle(i, 123, n) for i in range(n)]
        assert perm1 == perm2

    def test_different_seeds(self):
        """Verify different seeds produce different permutations."""
        n = 100
        perm1 = [index_shuffle(i, 42, n) for i in range(n)]
        perm2 = [index_shuffle(i, 99, n) for i in range(n)]
        assert perm1 != perm2

    def test_small_n(self):
        """Verify index_shuffle works for very small N."""
        for n in [1, 2, 3, 5]:
            shuffled = [index_shuffle(i, 42, n) for i in range(n)]
            assert sorted(shuffled) == list(range(n)), f"Failed for n={n}"

    def test_large_n(self):
        """Verify index_shuffle works for large N."""
        n = 100_000
        # Just test a sample (full permutation check would be slow)
        indices = [index_shuffle(i, 42, n) for i in range(100)]
        assert all(0 <= idx < n for idx in indices)
        # Should have some variety (not all same value)
        assert len(set(indices)) > 50
