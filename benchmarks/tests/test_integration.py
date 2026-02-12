"""Integration test: full benchmark pipeline end-to-end.

Verifies foundational exit criteria (Section 11.1):
SyntheticDataGenerator → DataraxAdapter.setup() → warmup() → iterate()
→ BenchmarkResult → BaselineStore.save() → BaselineStore.compare()

Design ref: Section 11.1 of the benchmark report.
"""

from pathlib import Path

import pytest

from benchmarks.adapters.base import IterationResult, ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.core.baselines import BaselineStore
from benchmarks.core.environment import capture_environment
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.timing import TimingSample


class TestFullPipelineIntegration:
    """End-to-end integration test for the benchmark infrastructure."""

    def test_cv1_small_full_pipeline(self, tmp_path: Path):
        """Run CV-1 small scenario through the complete pipeline.

        This is the foundational smoke test: all components compose correctly.
        """
        # 1. Generate synthetic data
        gen = SyntheticDataGenerator(seed=42)
        images = gen.images(100, 32, 32, c=3, dtype="uint8")
        data = {"image": images}

        # 2. Configure scenario
        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=100,
            element_shape=(32, 32, 3),
            batch_size=10,
            transforms=[],
            seed=42,
        )

        # 3. Setup adapter
        adapter = DataraxAdapter()
        assert adapter.is_available()
        adapter.setup(config, data)

        # 4. Warmup
        adapter.warmup(num_batches=2)

        # 5. Iterate and collect timing
        iteration_result = adapter.iterate(num_batches=5)
        assert isinstance(iteration_result, IterationResult)
        assert iteration_result.num_batches > 0
        assert iteration_result.num_elements > 0
        assert iteration_result.wall_clock_sec > 0

        # 6. Build BenchmarkResult from iteration data
        env = capture_environment()
        result = BenchmarkResult(
            framework=adapter.name,
            scenario_id=config.scenario_id,
            variant="small",
            timing=TimingSample(
                wall_clock_sec=iteration_result.wall_clock_sec,
                per_batch_times=iteration_result.per_batch_times,
                first_batch_time=iteration_result.first_batch_time,
                num_batches=iteration_result.num_batches,
                num_elements=iteration_result.num_elements,
            ),
            resources=None,
            environment=env,
            config={
                "batch_size": config.batch_size,
                "dataset_size": config.dataset_size,
            },
        )

        assert result.throughput_elements_sec() > 0

        # 7. Save as baseline
        store = BaselineStore(tmp_path / "baselines")
        store.save("CV-1_small", result)
        assert (tmp_path / "baselines" / "CV-1_small.json").exists()

        # 8. Compare against baseline (same result → pass)
        verdict = store.compare("CV-1_small", result)
        assert verdict is not None
        assert verdict["status"] == "pass"
        assert verdict["throughput_ratio"] == pytest.approx(1.0, abs=0.01)

        # 9. Teardown
        adapter.teardown()

    def test_nlp1_full_pipeline(self, tmp_path: Path):
        """Run NLP-1 scenario through the complete pipeline."""
        # 1. Generate synthetic token data
        gen = SyntheticDataGenerator(seed=99)
        tokens = gen.token_sequences(200, 128, vocab_size=32000)
        data = {"tokens": tokens}

        # 2. Configure scenario
        config = ScenarioConfig(
            scenario_id="NLP-1",
            dataset_size=200,
            element_shape=(128,),
            batch_size=20,
            transforms=[],
            seed=99,
        )

        # 3. Full lifecycle
        adapter = DataraxAdapter()
        adapter.setup(config, data)
        adapter.warmup(num_batches=1)
        iteration_result = adapter.iterate(num_batches=5)

        # 4. Build result
        result = BenchmarkResult(
            framework="Datarax",
            scenario_id="NLP-1",
            variant="small",
            timing=TimingSample(
                wall_clock_sec=iteration_result.wall_clock_sec,
                per_batch_times=iteration_result.per_batch_times,
                first_batch_time=iteration_result.first_batch_time,
                num_batches=iteration_result.num_batches,
                num_elements=iteration_result.num_elements,
            ),
            resources=None,
            environment={},
            config={"batch_size": 20},
        )

        # 5. Save and compare
        store = BaselineStore(tmp_path / "baselines")
        store.save("NLP-1_small", result)
        verdict = store.compare("NLP-1_small", result)

        assert verdict["status"] == "pass"

        adapter.teardown()

    def test_regression_detection_integration(self, tmp_path: Path):
        """Test that BaselineStore detects a simulated regression."""
        # Save a fast baseline
        fast_result = BenchmarkResult(
            framework="Datarax",
            scenario_id="CV-1",
            variant="small",
            timing=TimingSample(
                wall_clock_sec=1.0,
                per_batch_times=[0.02] * 50,
                first_batch_time=0.04,
                num_batches=50,
                num_elements=5000,
            ),
            resources=None,
            environment={},
            config={"batch_size": 100},
        )

        store = BaselineStore(tmp_path / "baselines")
        store.save("regression_test", fast_result)

        # Create a "regressed" result (2x slower)
        slow_result = BenchmarkResult(
            framework="Datarax",
            scenario_id="CV-1",
            variant="small",
            timing=TimingSample(
                wall_clock_sec=2.0,
                per_batch_times=[0.04] * 50,
                first_batch_time=0.08,
                num_batches=50,
                num_elements=5000,
            ),
            resources=None,
            environment={},
            config={"batch_size": 100},
        )

        verdict = store.compare("regression_test", slow_result)
        assert verdict["status"] == "failure"
        assert verdict["throughput_ratio"] < 1.0

    def test_deterministic_data_generation(self):
        """Verify SyntheticDataGenerator is deterministic (exit criterion)."""
        import numpy as np

        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)

        # Images
        np.testing.assert_array_equal(gen1.images(10, 32, 32), gen2.images(10, 32, 32))

        # Tokens
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)
        np.testing.assert_array_equal(
            gen1.token_sequences(10, 128),
            gen2.token_sequences(10, 128),
        )

        # Tabular
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)
        np.testing.assert_array_equal(
            gen1.tabular(100, 50),
            gen2.tabular(100, 50),
        )
