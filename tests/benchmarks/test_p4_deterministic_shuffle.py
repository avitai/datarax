"""P4: Deterministic multi-worker shuffling target tests.

Target: Same epoch output regardless of worker count.
"""

import pytest
import flax.nnx as nnx

from datarax.samplers.index_shuffle import index_shuffle
from datarax.sources import MemorySource, MemorySourceConfig


@pytest.mark.benchmark
class TestP4DeterministicShuffle:
    """P4: Shuffle order must be worker-count invariant."""

    def test_deterministic_across_restarts(self):
        """Verify same seed produces same order across process restarts."""
        data = [{"x": i} for i in range(1000)]

        config = MemorySourceConfig(shuffle=True)
        run1 = [item["x"] for item in MemorySource(config, data, rngs=nnx.Rngs(42))]
        run2 = [item["x"] for item in MemorySource(config, data, rngs=nnx.Rngs(42))]

        assert run1 == run2, "Same seed produced different shuffle orders"

    def test_shuffle_produces_permutation(self):
        """Verify shuffle produces a valid permutation (all elements present)."""
        data = [{"x": i} for i in range(500)]
        config = MemorySourceConfig(shuffle=True)
        source = MemorySource(config, data, rngs=nnx.Rngs(42))

        values = sorted(item["x"] for item in source)
        assert values == list(range(500)), "Shuffle lost or duplicated elements"

    def test_shuffle_changes_order(self):
        """Verify shuffle actually changes the order (not identity permutation)."""
        data = [{"x": i} for i in range(1000)]
        config = MemorySourceConfig(shuffle=True)
        source = MemorySource(config, data, rngs=nnx.Rngs(42))

        values = [item["x"] for item in source]
        # Extremely unlikely to be in order with 1000 elements
        assert values != list(range(1000)), "Shuffle produced identity permutation"


@pytest.mark.benchmark
class TestP4MultiWorkerPartitioning:
    """P4: Multi-worker partitioning via num_workers/shard_id."""

    def test_worker_partitions_cover_all_elements(self):
        """Verify all elements are covered across 4 workers (no gaps)."""
        data = [{"x": i} for i in range(100)]
        all_values: list[int] = []

        for worker_id in range(4):
            config = MemorySourceConfig(shuffle=True, num_workers=4, shard_id=worker_id)
            source = MemorySource(config, data, rngs=nnx.Rngs(42))
            all_values.extend(item["x"] for item in source)

        assert sorted(all_values) == list(range(100)), "Workers missed some elements"

    def test_worker_partitions_are_disjoint(self):
        """Verify no element appears in more than one worker's partition."""
        data = [{"x": i} for i in range(200)]
        seen: set[int] = set()

        for worker_id in range(4):
            config = MemorySourceConfig(shuffle=True, num_workers=4, shard_id=worker_id)
            source = MemorySource(config, data, rngs=nnx.Rngs(42))
            worker_values = {item["x"] for item in source}

            overlap = seen & worker_values
            assert not overlap, f"Worker {worker_id} overlaps with previous workers: {overlap}"
            seen |= worker_values

    def test_single_worker_equals_no_partitioning(self):
        """Verify num_workers=1 produces the same output as default (no partitioning)."""
        data = [{"x": i} for i in range(500)]

        config_default = MemorySourceConfig(shuffle=True)
        source = MemorySource(config_default, data, rngs=nnx.Rngs(42))
        default_order = [item["x"] for item in source]

        config_single = MemorySourceConfig(shuffle=True, num_workers=1, shard_id=0)
        single_order = [item["x"] for item in MemorySource(config_single, data, rngs=nnx.Rngs(42))]

        assert default_order == single_order

    def test_worker_count_invariant_global_order(self):
        """Verify that the global shuffled order is the same regardless of worker count.

        Collecting all workers' outputs (in order) and interleaving by global
        position must reconstruct the same global permutation.
        """
        data = [{"x": i} for i in range(120)]

        # Get global order from single worker
        config_1 = MemorySourceConfig(shuffle=True, num_workers=1, shard_id=0)
        global_order = [item["x"] for item in MemorySource(config_1, data, rngs=nnx.Rngs(42))]

        # Get partitioned orders from 4 workers and reconstruct global order
        worker_orders: list[list[int]] = []
        for worker_id in range(4):
            config = MemorySourceConfig(shuffle=True, num_workers=4, shard_id=worker_id)
            source = MemorySource(config, data, rngs=nnx.Rngs(42))
            worker_orders.append([item["x"] for item in source])

        # Reconstruct: worker k gets global positions [k::4]
        reconstructed = [0] * 120
        for worker_id, order in enumerate(worker_orders):
            for local_idx, value in enumerate(order):
                global_idx = worker_id + local_idx * 4
                reconstructed[global_idx] = value

        assert reconstructed == global_order

    def test_uneven_partition_handles_remainder(self):
        """Verify partitioning works when N is not divisible by num_workers."""
        data = [{"x": i} for i in range(103)]  # 103 % 4 != 0
        all_values: list[int] = []

        for worker_id in range(4):
            config = MemorySourceConfig(shuffle=True, num_workers=4, shard_id=worker_id)
            source = MemorySource(config, data, rngs=nnx.Rngs(42))
            all_values.extend(item["x"] for item in source)

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
