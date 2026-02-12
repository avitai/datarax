"""Tests for WebDatasetAdapter.

TDD: Tests written first. WebDataset is a Tier 2 TAR-based streaming loader.
Supports 3 scenarios. Requires TAR shard conversion in setup().
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

wds = pytest.importorskip("webdataset")


class TestWebDatasetAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        assert WebDatasetAdapter().name == "WebDataset"

    def test_version(self):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        v = WebDatasetAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        assert WebDatasetAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        expected = {"CV-1", "NLP-1"}
        assert WebDatasetAdapter().supported_scenarios() == expected


class TestWebDatasetAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = WebDatasetAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = WebDatasetAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(
        self,
        cv1_small_config,
        small_image_data,
        tmp_path,
    ):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = WebDatasetAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.webdataset_adapter import WebDatasetAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = WebDatasetAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
