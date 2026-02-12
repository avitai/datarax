"""Tests for TfDataAdapter.

TDD: Tests written first. tf.data is a Tier 1 adapter — supports 12 scenarios.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

tf = pytest.importorskip("tensorflow")


class TestTfDataAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        assert TfDataAdapter().name == "tf.data"

    def test_version(self):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        v = TfDataAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        assert TfDataAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        expected = {
            "CV-1",
            "NLP-1",
            "TAB-1",
            "DIST-1",
            "PR-1",
            "AUG-1",
            "AUG-2",
            "AUG-3",
        }
        assert TfDataAdapter().supported_scenarios() == expected


class TestTfDataAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        adapter = TfDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        adapter = TfDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        adapter = TfDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        adapter = TfDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        adapter = TfDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()


class TestTfDataAdapterNLP:
    def test_token_scenario(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.tfdata_adapter import TfDataAdapter

        adapter = TfDataAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches > 0
        adapter.teardown()
