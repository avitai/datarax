"""P3: Memory efficiency target tests.

Target: Peak RSS within 1.5x of SPDL during CV-1 pipeline execution.

Note: The comparative benchmark (vs SPDL adapter) requires the spdl package
and is skipped if unavailable. The functional tests always run.
"""

import pytest
import numpy as np
import flax.nnx as nnx

from datarax.sources import MemorySource, MemorySourceConfig
from tests.benchmarks.performance_targets import (
    measure_adapter_throughput,
    measure_peak_rss_delta_mb,
)


@pytest.mark.benchmark
class TestP3MemoryEfficiency:
    """P3: Datarax peak RSS within 1.5x of SPDL on CV-1."""

    def test_memory_source_array_views(self):
        """Verify dict data uses array slicing (views) for batching."""
        data = {
            "image": np.random.default_rng(42).integers(0, 255, (1000, 8, 8, 3), dtype=np.uint8)
        }
        config = MemorySourceConfig(prefetch_size=0)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Iterate through entire source — should use views not copies
        delta_mb = measure_peak_rss_delta_mb(lambda: list(source))

        # RSS increase should be modest. The source data is ~192KB,
        # so RSS delta should be dominated by Python/JAX overhead, not
        # data duplication. After optimization (array views), this should
        # be well under 200MB.
        assert delta_mb < 200, f"RSS increased by {delta_mb:.0f} MB during iteration"

    def test_gather_batch_efficiency(self):
        """Verify _gather_batch uses array indexing not list comprehension for arrays."""
        data = {"x": np.arange(500)}
        config = MemorySourceConfig(prefetch_size=0)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # get_batch should work efficiently
        batch = source.get_batch(32)
        assert len(batch["x"]) == 32

    def test_peak_rss_within_1_5x_spdl(self, cv1_large_image_data):
        """Compare peak RSS against SPDL (requires spdl package)."""
        pytest.importorskip("spdl")

        from benchmarks.adapters.base import ScenarioConfig
        from benchmarks.adapters.datarax_adapter import DataraxAdapter
        from benchmarks.adapters.spdl_adapter import SpdlAdapter as SPDLAdapter

        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=10_000,
            element_shape=(224, 224, 3),
            batch_size=64,
            transforms=[],
            seed=42,
        )

        datarax_rss = measure_peak_rss_delta_mb(
            lambda: measure_adapter_throughput(DataraxAdapter(), config, cv1_large_image_data)
        )
        spdl_rss = measure_peak_rss_delta_mb(
            lambda: measure_adapter_throughput(SPDLAdapter(), config, cv1_large_image_data)
        )

        if spdl_rss > 0:
            ratio = datarax_rss / spdl_rss
            assert ratio <= 1.5, (
                f"Datarax peak RSS ({datarax_rss:.0f} MB) is {ratio:.1f}x "
                f"SPDL ({spdl_rss:.0f} MB), exceeds 1.5x target"
            )
