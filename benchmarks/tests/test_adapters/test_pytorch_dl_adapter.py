"""Tests for PyTorchDataLoaderAdapter.

TDD: Tests written first. PyTorch DataLoader is the Tier 2 universal baseline.
Supports 11 scenarios.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

torch = pytest.importorskip("torch")


class TestPyTorchDLAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        assert PyTorchDataLoaderAdapter().name == "PyTorch DataLoader"

    def test_version(self):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        v = PyTorchDataLoaderAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        assert PyTorchDataLoaderAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

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
        assert PyTorchDataLoaderAdapter().supported_scenarios() == expected


class TestPyTorchDLAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        adapter = PyTorchDataLoaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        adapter = PyTorchDataLoaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        adapter = PyTorchDataLoaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        adapter = PyTorchDataLoaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        adapter = PyTorchDataLoaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()


class TestPyTorchDLAdapterNLP:
    def test_token_scenario(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.pytorch_dl_adapter import PyTorchDataLoaderAdapter

        adapter = PyTorchDataLoaderAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches > 0
        adapter.teardown()
