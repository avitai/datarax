"""Tests for pipeline complexity benchmark scenarios PC-1 through PC-5.

RED phase: these tests define the expected API and behavior for each
pipeline complexity scenario module before implementation exists.

All data-generation tests use small inline variants to avoid allocating
the full production datasets (which can be tens of GB).
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import ScenarioVariant
from benchmarks.tests.test_scenarios.conftest import assert_valid_variant, run_quick_scenario
from datarax.benchmarking.results import BenchmarkResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TINY_N = 100
"""Small dataset size for all data-shape tests."""

_TINY_BATCH = 10
"""Small batch size for all adapter integration tests."""


# ===========================================================================
# PC-1: Deep Transform Chain Scaling
# ===========================================================================


class TestPC1Scenario:
    """Tests for pc1_chain_depth scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.pipeline_complexity import pc1_chain_depth as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        assert self.mod.SCENARIO_ID == "PC-1"
        assert isinstance(self.mod.VARIANTS, dict)

        expected_depths = [1, 2, 5, 10, 20, 50, 100]
        assert len(self.mod.VARIANTS) == len(expected_depths)

        for depth in expected_depths:
            name = f"depth_{depth}"
            assert name in self.mod.VARIANTS, f"Missing variant {name}"
            variant = self.mod.VARIANTS[name]
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PC-1"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.extra["chain_depth"] == depth
            assert len(variant.config.transforms) == depth

    def test_data_generation_shapes(self):
        """Generated tabular data must have correct (N, 64, 64) shape.

        Uses a small inline generator to avoid allocating the full 50K
        production dataset in CI.
        """
        variant = self.mod.get_variant("depth_1")
        cfg = variant.config
        # Verify config declares the expected element shape
        assert cfg.element_shape == (64, 64)
        assert cfg.dataset_size == 50_000

        # Generate a small sample to validate structure and dtype
        rng = np.random.default_rng(cfg.seed)
        data = {"data": rng.standard_normal((_TINY_N, *cfg.element_shape)).astype(np.float32)}
        assert data["data"].shape == (_TINY_N, 64, 64)
        assert data["data"].dtype == np.float32

    def test_tier1_variant_exists(self):
        """PC-1 must define TIER1_VARIANT pointing to 'depth_1'."""
        assert self.mod.TIER1_VARIANT == "depth_1"
        tier1 = self.mod.get_variant(self.mod.TIER1_VARIANT)
        assert tier1.config.extra["variant_name"] == "depth_1"

    def test_chain_depth_transform_cycle(self):
        """Transforms must cycle through the canonical 5-transform set."""
        cycle = ["Normalize", "Scale", "Clip", "Add", "Multiply"]
        for depth in [1, 5, 10]:
            variant = self.mod.get_variant(f"depth_{depth}")
            transforms = variant.config.transforms
            assert len(transforms) == depth
            for i, t in enumerate(transforms):
                assert t == cycle[i % len(cycle)]

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running PC-1 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="PC-1",
                dataset_size=_TINY_N,
                element_shape=(64, 64),
                batch_size=_TINY_BATCH,
                transforms=["Normalize"],
                extra={"variant_name": "test_tiny", "chain_depth": 1},
            ),
            data_generator=lambda: {
                "data": np.random.default_rng(42)
                .standard_normal((_TINY_N, 64, 64))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "PC-1"
        assert result.timing.num_batches > 0


# ===========================================================================
# PC-2: Branching/Parallel Pipeline (DAG)
# ===========================================================================


class TestPC2Scenario:
    """Tests for pc2_branching_dag scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.pipeline_complexity import pc2_branching_dag as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Single variant must have valid config with DAG transforms."""
        assert self.mod.SCENARIO_ID == "PC-2"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PC-2"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.extra["topology"] == "parallel_dag"
            assert variant.config.transforms == ["BranchA", "BranchB", "Merge"]

    def test_data_generation_shapes(self):
        """Config declares NHWC images; small sample validates structure."""
        variant = self.mod.get_variant("default")
        cfg = variant.config
        assert cfg.element_shape == (64, 64, 3)
        assert cfg.dataset_size == 50_000

        gen = SyntheticDataGenerator(seed=cfg.seed)
        data = {"image": gen.images(_TINY_N, 64, 64, 3, dtype="float32")}
        assert data["image"].shape == (_TINY_N, 64, 64, 3)
        assert data["image"].dtype == np.float32

    def test_tier1_variant_is_none(self):
        """PC-2 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running PC-2 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="PC-2",
                dataset_size=_TINY_N,
                element_shape=(32, 32, 3),
                batch_size=_TINY_BATCH,
                transforms=["BranchA", "BranchB", "Merge"],
                extra={"variant_name": "test_tiny", "topology": "parallel_dag"},
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42)
                .standard_normal((_TINY_N, 32, 32, 3))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "PC-2"
        assert result.timing.num_batches > 0


# ===========================================================================
# PC-3: Differentiable Rebatching
# ===========================================================================


class TestPC3Scenario:
    """Tests for pc3_rebatching scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.pipeline_complexity import pc3_rebatching as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Single variant must have valid config with rebatching params."""
        assert self.mod.SCENARIO_ID == "PC-3"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PC-3"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.extra["target_batch_size"] == 32
            assert variant.config.transforms == ["Normalize"]

    def test_data_generation_shapes(self):
        """Config declares large NHWC images; small sample validates structure."""
        variant = self.mod.get_variant("default")
        cfg = variant.config
        assert cfg.element_shape == (64, 64, 3)
        assert cfg.dataset_size == 5_000
        assert cfg.batch_size == 256

        gen = SyntheticDataGenerator(seed=cfg.seed)
        data = {"image": gen.images(_TINY_N, 64, 64, 3, dtype="float32")}
        assert data["image"].shape == (_TINY_N, 64, 64, 3)
        assert data["image"].dtype == np.float32

    def test_tier1_variant_is_none(self):
        """PC-3 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running PC-3 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="PC-3",
                dataset_size=_TINY_N,
                element_shape=(32, 32, 3),
                batch_size=_TINY_BATCH,
                transforms=["Normalize"],
                extra={
                    "variant_name": "test_tiny",
                    "target_batch_size": 32,
                },
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42)
                .standard_normal((_TINY_N, 32, 32, 3))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "PC-3"
        assert result.timing.num_batches > 0


# ===========================================================================
# PC-4: Probabilistic & Conditional Pipeline
# ===========================================================================


class TestPC4Scenario:
    """Tests for pc4_probabilistic scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.pipeline_complexity import pc4_probabilistic as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Single variant must have valid config with probabilistic params."""
        assert self.mod.SCENARIO_ID == "PC-4"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PC-4"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.extra["probability"] == 0.5
            assert variant.config.transforms == [
                "ProbabilisticAugment",
                "ConditionalSelect",
            ]

    def test_data_generation_shapes(self):
        """Config declares NHWC images; small sample validates structure."""
        variant = self.mod.get_variant("default")
        cfg = variant.config
        assert cfg.element_shape == (64, 64, 3)
        assert cfg.dataset_size == 50_000

        gen = SyntheticDataGenerator(seed=cfg.seed)
        data = {"image": gen.images(_TINY_N, 64, 64, 3, dtype="float32")}
        assert data["image"].shape == (_TINY_N, 64, 64, 3)
        assert data["image"].dtype == np.float32

    def test_tier1_variant_is_none(self):
        """PC-4 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running PC-4 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="PC-4",
                dataset_size=_TINY_N,
                element_shape=(32, 32, 3),
                batch_size=_TINY_BATCH,
                transforms=["ProbabilisticAugment", "ConditionalSelect"],
                extra={
                    "variant_name": "test_tiny",
                    "probability": 0.5,
                },
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42)
                .standard_normal((_TINY_N, 32, 32, 3))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "PC-4"
        assert result.timing.num_batches > 0


# ===========================================================================
# PC-5: End-to-End Differentiable Pipeline
# ===========================================================================


class TestPC5Scenario:
    """Tests for pc5_differentiable scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.pipeline_complexity import pc5_differentiable as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Single variant must have valid config with differentiable flag."""
        assert self.mod.SCENARIO_ID == "PC-5"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PC-5"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.extra["differentiable"] is True
            assert variant.config.transforms == ["LearnableAugment", "Normalize"]

    def test_data_generation_shapes(self):
        """Config declares small NHWC images; sample validates structure."""
        variant = self.mod.get_variant("default")
        cfg = variant.config
        assert cfg.element_shape == (32, 32, 3)
        assert cfg.dataset_size == 10_000

        gen = SyntheticDataGenerator(seed=cfg.seed)
        data = {"image": gen.images(_TINY_N, 32, 32, 3, dtype="float32")}
        assert data["image"].shape == (_TINY_N, 32, 32, 3)
        assert data["image"].dtype == np.float32

    def test_tier1_variant_is_none(self):
        """PC-5 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running PC-5 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="PC-5",
                dataset_size=_TINY_N,
                element_shape=(32, 32, 3),
                batch_size=_TINY_BATCH,
                transforms=["LearnableAugment", "Normalize"],
                extra={
                    "variant_name": "test_tiny",
                    "differentiable": True,
                },
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42)
                .standard_normal((_TINY_N, 32, 32, 3))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "PC-5"
        assert result.timing.num_batches > 0
