"""Tests for FfcvAdapter.

TDD: Tests written first. FFCV is a Tier 2 vision-only .beton format loader.
Only supports CV-1.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

ffcv = pytest.importorskip("ffcv")


class TestFfcvAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        assert FfcvAdapter().name == "FFCV"

    def test_version(self):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        v = FfcvAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        assert FfcvAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        assert FfcvAdapter().supported_scenarios() == {"CV-1"}


class TestFfcvAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        adapter = FfcvAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        adapter = FfcvAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        adapter = FfcvAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        adapter = FfcvAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.ffcv_adapter import FfcvAdapter

        adapter = FfcvAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
