"""Tests for LitDataAdapter.

TDD: Tests written first. LitData is Lightning's Tier 3 cloud streaming library.
Supports 2 scenarios. Requires format conversion in setup().
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

litdata = pytest.importorskip("litdata")


class TestLitDataAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        assert LitDataAdapter().name == "LitData"

    def test_version(self):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        v = LitDataAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        assert LitDataAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        expected = {"CV-1"}
        assert LitDataAdapter().supported_scenarios() == expected


class TestLitDataAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = LitDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = LitDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(
        self,
        cv1_small_config,
        small_image_data,
        tmp_path,
    ):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = LitDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.litdata_adapter import LitDataAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = LitDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
