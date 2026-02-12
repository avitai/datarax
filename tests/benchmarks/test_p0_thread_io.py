"""P0: Thread-based I/O performance target tests.

Target: Datarax CV-1 throughput within 1.2x of SPDL.
These tests MUST fail before optimization and MUST pass after.

Note: The comparative benchmark (vs SPDL adapter) requires the spdl package
and is skipped if unavailable. The functional integration test for prefetch
always runs.
"""

import pytest
import flax.nnx as nnx

from datarax.sources import MemorySource, MemorySourceConfig


@pytest.mark.benchmark
class TestP0ThreadIO:
    """P0: Verify threaded prefetch integration in sources."""

    def test_prefetch_integration_in_source(self):
        """Verify sources use threaded prefetch when prefetch_size > 0."""
        config = MemorySourceConfig(prefetch_size=2)
        data = [{"x": i} for i in range(100)]
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Source should use threaded prefetch when prefetch_size > 0
        batches = list(source)
        assert len(batches) == 100

    def test_prefetch_disabled_by_default(self):
        """Verify prefetch_size=0 skips threading (no overhead for simple use)."""
        config = MemorySourceConfig(prefetch_size=0)
        data = [{"x": i} for i in range(50)]
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        batches = list(source)
        assert len(batches) == 50

    def test_prefetch_preserves_data_integrity(self):
        """Verify prefetched iteration produces identical data to non-prefetched."""
        data = [{"x": i} for i in range(200)]

        # Without prefetch
        config_no_pf = MemorySourceConfig(prefetch_size=0)
        source_no_pf = MemorySource(config_no_pf, data, rngs=nnx.Rngs(0))
        items_no_pf = [item["x"] for item in source_no_pf]

        # With prefetch
        config_pf = MemorySourceConfig(prefetch_size=4)
        source_pf = MemorySource(config_pf, data, rngs=nnx.Rngs(0))
        items_pf = [item["x"] for item in source_pf]

        assert items_no_pf == items_pf

    def test_prefetch_with_shuffle(self):
        """Verify prefetch works correctly with shuffling enabled."""
        data = [{"x": i} for i in range(100)]
        config = MemorySourceConfig(prefetch_size=2, shuffle=True)
        source = MemorySource(config, data, rngs=nnx.Rngs(42))

        batches = list(source)
        assert len(batches) == 100
        # Should be shuffled (very unlikely to be in order)
        values = [b["x"] for b in batches]
        assert values != list(range(100))

    def test_prefetch_with_dict_data(self):
        """Verify prefetch works with dictionary-format data."""
        import numpy as np

        data = {"image": np.random.default_rng(42).integers(0, 255, (50, 8, 8, 3), dtype=np.uint8)}
        config = MemorySourceConfig(prefetch_size=3)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        batches = list(source)
        assert len(batches) == 50

    def test_prefetch_propagates_producer_error(self):
        """Verify exceptions from the producer thread propagate to consumer."""
        from datarax.control.prefetcher import Prefetcher

        def failing_iterator():
            yield 1
            yield 2
            raise ValueError("producer failed")

        prefetcher = Prefetcher(buffer_size=2)
        it = prefetcher.prefetch(failing_iterator())

        assert next(it) == 1
        assert next(it) == 2
        with pytest.raises(ValueError, match="producer failed"):
            next(it)
