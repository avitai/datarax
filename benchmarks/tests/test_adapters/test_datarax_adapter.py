"""Tests for DataraxAdapter.

TDD: Write tests first, then implement.
Design ref: Section 7.2 of the benchmark report.

Uses shared fixtures from benchmarks/conftest.py (cv1_small_config,
small_image_data, nlp1_small_config, small_token_data) and
benchmarks/tests/test_adapters/conftest.py (mm1_small_config, etc.).
"""

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.tests.test_adapters.conftest import (
    assert_supported_scenarios,
    assert_valid_iteration_result,
)


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

    def test_warmup_consumes_exact_num_batches(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        assert adapter._pipeline._iteration_total == 0  # type: ignore[reportAttributeAccessIssue]

        adapter.warmup(num_batches=2)

        assert adapter._pipeline._iteration_total == 2  # type: ignore[reportAttributeAccessIssue]
        adapter.teardown()

    def test_iterate_consumes_exact_num_batches(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        assert adapter._pipeline._iteration_total == 0  # type: ignore[reportAttributeAccessIssue]

        result = adapter.iterate(num_batches=3)

        assert result.num_batches == 3
        assert adapter._pipeline._iteration_total == 3  # type: ignore[reportAttributeAccessIssue]
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


class TestDataraxAdapterPrefetchPolicy:
    """Benchmark adapter should control prefetch policy explicitly."""

    def test_setup_defaults_prefetch_to_two(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        assert adapter._pipeline.buffer_depth == 2  # type: ignore[reportAttributeAccessIssue]
        adapter.teardown()

    def test_setup_prefetch_override_from_extra(self, cv1_small_config, small_image_data):
        adapter = DataraxAdapter()
        override_config = ScenarioConfig(
            scenario_id=cv1_small_config.scenario_id,
            dataset_size=cv1_small_config.dataset_size,
            element_shape=cv1_small_config.element_shape,
            batch_size=cv1_small_config.batch_size,
            transforms=cv1_small_config.transforms,
            num_workers=cv1_small_config.num_workers,
            seed=cv1_small_config.seed,
            extra={**cv1_small_config.extra, "prefetch_size": 4},
        )
        adapter.setup(override_config, small_image_data)
        assert adapter._pipeline.buffer_depth == 4  # type: ignore[reportAttributeAccessIssue]
        adapter.teardown()
