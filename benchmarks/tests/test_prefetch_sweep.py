"""Tests for the prefetch sensitivity sweep utility."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.core.prefetch_sweep import (
    _override_prefetch,
    PrefetchSweepResult,
    run_prefetch_sweep,
)
from benchmarks.scenarios.base import ScenarioVariant


@pytest.fixture()
def cv1_variant() -> ScenarioVariant:
    config = ScenarioConfig(
        scenario_id="CV-1",
        dataset_size=200,
        element_shape=(32, 32, 3),
        batch_size=32,
        transforms=["Normalize", "CastToFloat32"],
        extra={"variant_name": "sweep_test"},
    )
    return ScenarioVariant(
        config=config,
        data_generator=lambda: {
            "image": np.random.randint(0, 255, (200, 32, 32, 3), dtype=np.uint8)
        },
    )


class TestOverridePrefetch:
    """Test the prefetch config override helper."""

    def test_creates_new_config(self) -> None:
        original = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=100,
            element_shape=(32, 32, 3),
            batch_size=16,
            transforms=["Normalize"],
            extra={"variant_name": "test"},
        )
        modified = _override_prefetch(original, 8)
        assert modified.extra["prefetch_size"] == 8
        assert modified.scenario_id == "CV-1"
        assert modified.batch_size == 16

    def test_does_not_mutate_original(self) -> None:
        original = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=100,
            element_shape=(32, 32, 3),
            batch_size=16,
            transforms=["Normalize"],
            extra={"variant_name": "test"},
        )
        _override_prefetch(original, 4)
        assert "prefetch_size" not in original.extra


class TestPrefetchSweepResult:
    """Test PrefetchSweepResult dataclass."""

    def test_default_values(self) -> None:
        result = PrefetchSweepResult(
            scenario_id="CV-1",
            variant_name="test",
            adapter_name="Datarax",
        )
        assert result.optimal_prefetch == 0
        assert result.speedup_vs_disabled == 1.0
        assert result.points == []


class TestRunPrefetchSweep:
    """Integration test for the full prefetch sweep."""

    def test_sweep_returns_points_for_each_size(self, cv1_variant: ScenarioVariant) -> None:
        adapter = DataraxAdapter()
        result = run_prefetch_sweep(
            adapter,
            cv1_variant,
            prefetch_sizes=(0, 2),
            num_batches=3,
            warmup_batches=1,
            num_repetitions=1,
        )

        assert isinstance(result, PrefetchSweepResult)
        assert len(result.points) == 2
        assert result.points[0].prefetch_size == 0
        assert result.points[1].prefetch_size == 2
        assert result.scenario_id == "CV-1"
        assert result.adapter_name == "Datarax"

    def test_sweep_identifies_optimal(self, cv1_variant: ScenarioVariant) -> None:
        adapter = DataraxAdapter()
        result = run_prefetch_sweep(
            adapter,
            cv1_variant,
            prefetch_sizes=(0, 2),
            num_batches=3,
            warmup_batches=1,
            num_repetitions=1,
        )

        assert result.optimal_prefetch in (0, 2)
        assert result.speedup_vs_disabled >= 0.5  # Sanity check

    def test_all_points_have_positive_throughput(self, cv1_variant: ScenarioVariant) -> None:
        adapter = DataraxAdapter()
        result = run_prefetch_sweep(
            adapter,
            cv1_variant,
            prefetch_sizes=(0, 2),
            num_batches=3,
            warmup_batches=1,
            num_repetitions=1,
        )

        for point in result.points:
            assert point.throughput_elem_s > 0
            assert point.wall_clock_sec > 0
