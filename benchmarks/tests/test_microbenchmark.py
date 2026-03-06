"""Tests for the internal microbenchmark decomposition module."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.core.microbenchmark import MicrobenchmarkResult, run_microbenchmark


@pytest.fixture()
def cv1_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="CV-1",
        dataset_size=200,
        element_shape=(32, 32, 3),
        batch_size=32,
        transforms=["Normalize", "CastToFloat32"],
        extra={"variant_name": "micro_test"},
    )


@pytest.fixture()
def image_data() -> dict[str, np.ndarray]:
    return {"image": np.random.randint(0, 255, (200, 32, 32, 3), dtype=np.uint8)}


class TestMicrobenchmarkResult:
    """Test MicrobenchmarkResult properties."""

    def test_goodput_ratio_computed(self) -> None:
        result = MicrobenchmarkResult(
            scenario_id="CV-1",
            variant_name="test",
            num_batches=10,
            wall_clock_sec=1.0,
            source_sec=0.3,
            transform_sec=0.4,
            transfer_sec=0.1,
            overhead_sec=0.2,
        )
        assert result.goodput_ratio == pytest.approx(0.8, abs=0.01)

    def test_goodput_ratio_zero_wall_clock(self) -> None:
        result = MicrobenchmarkResult(
            scenario_id="CV-1",
            variant_name="test",
            num_batches=0,
            wall_clock_sec=0.0,
            source_sec=0.0,
            transform_sec=0.0,
            transfer_sec=0.0,
            overhead_sec=0.0,
        )
        assert result.goodput_ratio == 0.0

    def test_summary_dict_keys(self) -> None:
        result = MicrobenchmarkResult(
            scenario_id="CV-1",
            variant_name="test",
            num_batches=5,
            wall_clock_sec=0.5,
            source_sec=0.1,
            transform_sec=0.2,
            transfer_sec=0.05,
            overhead_sec=0.15,
        )
        summary = result.summary_dict()
        assert "goodput_ratio" in summary
        assert "source_pct" in summary
        assert "transform_pct" in summary
        assert "overhead_pct" in summary
        assert summary["num_batches"] == 5


class TestRunMicrobenchmark:
    """Test the run_microbenchmark function with real adapter."""

    def test_returns_valid_result(
        self, cv1_config: ScenarioConfig, image_data: dict[str, np.ndarray]
    ) -> None:
        adapter = DataraxAdapter()
        result = run_microbenchmark(
            adapter, cv1_config, image_data, num_batches=3, warmup_batches=1
        )

        assert isinstance(result, MicrobenchmarkResult)
        assert result.scenario_id == "CV-1"
        assert result.variant_name == "micro_test"
        assert result.num_batches == 3
        assert result.wall_clock_sec > 0
        assert result.source_sec >= 0
        assert result.transform_sec >= 0
        assert result.transfer_sec >= 0

    def test_per_batch_lists_match_count(
        self, cv1_config: ScenarioConfig, image_data: dict[str, np.ndarray]
    ) -> None:
        adapter = DataraxAdapter()
        result = run_microbenchmark(
            adapter, cv1_config, image_data, num_batches=3, warmup_batches=1
        )

        assert len(result.per_batch_source) == result.num_batches
        assert len(result.per_batch_transform) == result.num_batches
        assert len(result.per_batch_transfer) == result.num_batches

    def test_components_sum_to_wall_clock(
        self, cv1_config: ScenarioConfig, image_data: dict[str, np.ndarray]
    ) -> None:
        adapter = DataraxAdapter()
        result = run_microbenchmark(
            adapter, cv1_config, image_data, num_batches=5, warmup_batches=1
        )

        component_sum = (
            result.source_sec + result.transform_sec + result.transfer_sec + result.overhead_sec
        )
        assert component_sum == pytest.approx(result.wall_clock_sec, abs=0.001)
