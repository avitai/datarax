"""Tests for DataraxAdapter.

TDD: Write tests first, then implement.
Design ref: Section 7.2 of the benchmark report.

Uses shared fixtures from benchmarks/conftest.py (cv1_small_config,
small_image_data, nlp1_small_config, small_token_data) and
benchmarks/tests/test_adapters/conftest.py (mm1_small_config, etc.).
"""

import numpy as np
from substrax.testing.compiles import expect_compiles

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.tests.test_adapters.conftest import (
    assert_supported_scenarios,
    assert_valid_iteration_result,
)
from datarax import HostResources


class TestDataraxAdapterProperties:
    """Test adapter properties."""

    def test_name(self):
        adapter = DataraxAdapter()
        assert adapter.name == "Datarax"

    def test_version(self):
        adapter = DataraxAdapter()
        assert isinstance(adapter.version, str)
        assert len(adapter.version) > 0

    def test_is_available(self):
        adapter = DataraxAdapter()
        assert adapter.is_available() is True

    def test_supported_scenarios(self):
        assert_supported_scenarios(
            DataraxAdapter(),
            must_include={"CV-1", "NLP-1", "TAB-1", "HCV-1", "HPC-1", "AUG-1"},
        )


class TestDataraxAdapterLifecycle:
    """Test the setup -> warmup -> iterate -> teardown lifecycle."""

    def test_setup_creates_pipeline(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup_runs_batches(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)

        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)

        assert result.num_batches <= 3

        adapter.teardown()

    def test_warmup_then_iterate_delivers_exact_batches(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)

        adapter.warmup(num_batches=2)
        result = adapter.iterate(num_batches=3)

        # Warmup does not corrupt subsequent iteration; exactly 3 batches delivered.
        assert result.num_batches == 3
        adapter.teardown()

    def test_iterate_consumes_exact_num_batches(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)

        result = adapter.iterate(num_batches=3)

        assert result.num_batches == 3
        adapter.teardown()

    def test_iterate_total_bytes_positive(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)

        assert result.total_bytes > 0

        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()  # Should not raise


class TestDataraxAdapterNLP:
    """Test adapter with NLP-style data."""

    def test_token_sequence_scenario(self, nlp1_small_config, small_token_data):
        adapter = DataraxAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)

        assert result.num_batches > 0
        assert result.num_elements > 0

        adapter.teardown()


class TestDataraxAdapterHostResources:
    """The scenario's CPU workers and RAM budget are the host stage's ``HostResources``."""

    def test_a_scenario_without_a_budget_reads_as_datarax_does_without_one(
        self, cv1_small_config, small_image_data
    ):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        plan = adapter._pipeline.host_plan
        assert adapter._pipeline.host_resources is None
        assert (plan.threads, plan.read_buffer) == (1, 2)
        adapter.teardown()

    def test_prefetch_size_does_not_reach_the_host_stage(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(_with(cv1_small_config, extra={"prefetch_size": 4}), small_image_data)
        assert adapter._pipeline.host_plan.read_buffer == 2
        adapter.teardown()

    def test_num_workers_and_a_ram_budget_become_host_resources(
        self, cv1_small_config, small_image_data
    ):
        adapter = DataraxAdapter()
        config = _with(cv1_small_config, num_workers=1, extra={"ram_budget_bytes": 1 << 30})
        adapter.setup(config, small_image_data)
        assert adapter._pipeline.host_resources == HostResources(
            ram_budget_bytes=1 << 30, max_workers=1
        )
        adapter.teardown()


def _with(config, *, num_workers=None, extra=None):
    return ScenarioConfig(
        scenario_id=config.scenario_id,
        dataset_size=config.dataset_size,
        element_shape=config.element_shape,
        batch_size=config.batch_size,
        transforms=config.transforms,
        num_workers=config.num_workers if num_workers is None else num_workers,
        seed=config.seed,
        extra={**config.extra, **(extra or {})},
    )


class TestDataraxAdapterSteadyState:
    """At the end of an epoch the adapter stops with full batches and compiles nothing new."""

    def test_iterate_reaching_the_epoch_end_serves_only_full_batches(self, synth_gen):
        # 25 records in batches of 10: the epoch ends mid-batch. A short final batch would be a
        # second shape, compiled inside the timed region.
        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=25,
            element_shape=(32, 32, 3),
            batch_size=10,
            transforms=["Normalize", "CastToFloat32"],
        )
        adapter = DataraxAdapter()
        adapter.setup(config, {"image": synth_gen.images(25, 32, 32, 3, dtype="uint8")})
        adapter.warmup(num_batches=1)
        with expect_compiles(0):
            result = adapter.iterate(num_batches=6)
        assert result.num_batches == 1
        assert result.num_elements == config.batch_size
        adapter.teardown()


class TestDataraxAdapterFieldRanks:
    """Spatial transforms act on image-like fields; a token field passes through (HMM-1)."""

    def test_random_resized_crop_passes_a_token_field_through(self, synth_gen):
        images = synth_gen.images(8, 16, 16, 3, dtype="uint8")
        tokens = synth_gen.token_sequences(8, 7)
        config = ScenarioConfig(
            scenario_id="HMM-1",
            dataset_size=8,
            element_shape=(16, 16, 3),
            batch_size=4,
            transforms=["RandomResizedCrop"],
        )
        adapter = DataraxAdapter()
        adapter.setup(config, {"image": images, "tokens": tokens})
        batch = next(iter(adapter._iterate_batches()))
        assert batch["image"].shape == (4, 16, 16, 3)
        np.testing.assert_array_equal(np.asarray(batch["tokens"]), tokens[:4])
        adapter.teardown()
