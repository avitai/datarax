"""Tests for DataraxAdapter.

TDD: Write tests first, then implement.
Design ref: Section 7.2 of the benchmark report.

Uses shared fixtures from benchmarks/conftest.py (cv1_small_config,
small_image_data, nlp1_small_config, small_token_data) and
benchmarks/tests/test_adapters/conftest.py (mm1_small_config, etc.).
"""

from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result


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
        adapter = DataraxAdapter()
        scenarios = adapter.supported_scenarios()
        assert isinstance(scenarios, set)
        assert "CV-1" in scenarios
        assert "NLP-1" in scenarios
        # Datarax supports all 28 scenarios (25 original + 3 AUG)
        assert len(scenarios) == 28


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
