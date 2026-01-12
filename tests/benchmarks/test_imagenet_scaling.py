import pytest
import grain.python as grain
from pathlib import Path
from datarax.benchmarking.profiler import AdvancedProfiler, ProfilerConfig
from datarax.sources.array_record_source import ArrayRecordSourceModule, ArrayRecordSourceConfig

# Path to the converted dataset directory
# Path to the converted dataset directory (in gitignored tests/data)
DATASET_DIR = Path("tests/data/imagenet64_arrayrecord")


@pytest.mark.benchmark
@pytest.mark.skipif(
    not DATASET_DIR.exists(), reason="Real ImageNet ArrayRecord directory not found."
)
class TestRealIO:
    @pytest.fixture(autouse=True)
    def setup_profiler(self):
        self.profiler = AdvancedProfiler(
            config=ProfilerConfig(
                warmup_steps=5,
                measure_steps=50,
                enable_trace=False,  # Enable trace manually if needed, disable for throughput tests
            )
        )

    def test_array_record_throughput(self, benchmark):
        """Measures throughput of ArrayRecord reading with varying worker counts."""

        # Get all sharded files
        shards = sorted(list(DATASET_DIR.glob("*.array_record")))
        if not shards:
            pytest.skip("No .array_record files found in dataset directory.")

        # Manually calling the param logic via a loop for the 'test' body
        # or defining properly with pytest parametrization.
        # Let's stick to a single test that sweeps.

        results = {}
        # Get all sharded files
        shards = sorted(list(DATASET_DIR.glob("*.array_record")))

        for workers in [0, 4, 8, 16]:
            # Re-instantiate profiler clean
            config = ProfilerConfig(warmup_steps=10, measure_steps=100)
            self.profiler = AdvancedProfiler(config)

            # Setup Source
            src_config = ArrayRecordSourceConfig(shuffle_files=False)
            source = ArrayRecordSourceModule(config=src_config, paths=[str(p) for p in shards])

            # Setup Loader
            loader = grain.DataLoader(
                data_source=source.grain_source,
                sampler=grain.SequentialSampler(
                    num_records=len(source.grain_source),
                    shard_options=grain.ShardOptions(
                        shard_index=0, shard_count=1, drop_remainder=True
                    ),
                ),
                worker_count=workers,
                operations=[grain.Batch(128, drop_remainder=True)],
            )

            def payload():
                it = iter(loader)
                for _ in range(100):
                    next(it)

            res = self.profiler.profile(payload, f"workers_{workers}")

            # Access metrics correctly
            throughput = res.timing_metrics.get("iterations_per_second", 0.0)
            latency = res.timing_metrics.get("mean_time_s", 0.0) * 1000.0  # Convert to ms

            results[workers] = throughput
            print(f"Workers {workers}: {throughput:.1f} batches/s (Latency: {latency:.2f} ms)")

        # Check scaling (roughly)
        # 0 workers (main process) vs 4 workers should show improvement or similar if IO bound
        # Note: 0 workers in Grain might mean synchronous main thread.

        best_throughput = max(results.values())
        print(f"Peak Throughput: {best_throughput:.1f} batches/s")
