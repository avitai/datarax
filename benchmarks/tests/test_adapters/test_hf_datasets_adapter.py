"""Tests for HfDatasetsAdapter.

TDD: Tests written first. HuggingFace Datasets — Tier 3 specialized.
Supports 5 scenarios (NLP/tabular/multimodal focus).
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

datasets = pytest.importorskip("datasets")


class TestHfDatasetsAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        assert HfDatasetsAdapter().name == "HuggingFace Datasets"

    def test_version(self):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        v = HfDatasetsAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        assert HfDatasetsAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        expected = {"NLP-1", "TAB-1"}
        assert HfDatasetsAdapter().supported_scenarios() == expected


class TestHfDatasetsAdapterLifecycle:
    def test_setup_teardown(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        adapter = HfDatasetsAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.teardown()

    def test_warmup(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        adapter = HfDatasetsAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        adapter = HfDatasetsAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        adapter = HfDatasetsAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=2)
        assert result.num_batches <= 2
        adapter.teardown()

    def test_teardown_is_idempotent(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        adapter = HfDatasetsAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.teardown()
        adapter.teardown()


class TestHfDatasetsAdapterTabular:
    def test_tabular_scenario(self, tab1_small_config, small_tabular_data):
        from benchmarks.adapters.hf_datasets_adapter import HfDatasetsAdapter

        adapter = HfDatasetsAdapter()
        adapter.setup(tab1_small_config, small_tabular_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches > 0
        adapter.teardown()
