"""ImageNet ArrayRecord I/O scaling benchmark.

Uses TimingCollector for measurement (replaces AdvancedProfiler).
"""

import pytest

from datarax.utils.console import emit


grain = pytest.importorskip("grain")
pytest.importorskip("grain.samplers")
pytest.importorskip("grain.sharding")
pytest.importorskip("grain.transforms")

from pathlib import Path

from calibrax.profiling import TimingCollector


# Path to the converted dataset directory (in gitignored tests/data)
DATASET_DIR = Path("tests/data/imagenet64_arrayrecord")


@pytest.mark.benchmark
@pytest.mark.skipif(
    not DATASET_DIR.exists(), reason="Real ImageNet ArrayRecord directory not found."
)
class TestRealIO:
    def test_array_record_throughput(self):
        """Measures Grain's throughput reading the ArrayRecord files with varying worker counts."""
        shards = sorted(list(DATASET_DIR.glob("*.array_record")))
        if not shards:
            pytest.skip("No .array_record files found in dataset directory.")

        results = {}

        for workers in [0, 4, 8, 16]:
            records = grain.sources.ArrayRecordDataSource([str(p) for p in shards])

            loader = grain.DataLoader(
                data_source=records,
                sampler=grain.samplers.SequentialSampler(
                    num_records=len(records),
                    shard_options=grain.sharding.ShardOptions(
                        shard_index=0, shard_count=1, drop_remainder=True
                    ),
                ),
                worker_count=workers,
                operations=[grain.transforms.Batch(128, drop_remainder=True)],
            )

            # Warmup
            warmup_it = iter(loader)
            for _ in range(5):
                try:
                    next(warmup_it)
                except StopIteration:
                    break

            # Measure with TimingCollector
            collector = TimingCollector()
            sample = collector.measure_iteration(iter(loader), num_batches=100)

            throughput = (
                sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
            )
            latency_ms = (
                (sample.wall_clock_sec / sample.num_batches * 1000) if sample.num_batches > 0 else 0
            )

            results[workers] = throughput
            emit(f"Workers {workers}: {throughput:.1f} batches/s (Latency: {latency_ms:.2f} ms)")

        best_throughput = max(results.values())
        emit(f"Peak Throughput: {best_throughput:.1f} batches/s")
