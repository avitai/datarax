"""Tests for SpdlAdapter.

TDD: Tests written first. Meta SPDL is a Tier 2 thread-based loader.
Supports 6 scenarios.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result


spdl = pytest.importorskip("spdl")


class TestSpdlAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        assert SpdlAdapter().name == "SPDL"

    def test_version(self):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        v = SpdlAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        assert SpdlAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter
        from benchmarks.tests.test_adapters.conftest import assert_supported_scenarios

        assert_supported_scenarios(
            SpdlAdapter(),
            must_include={"CV-1", "NLP-1", "HCV-1", "HPC-1", "AUG-1"},
        )


class TestSpdlAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        adapter = SpdlAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        adapter = SpdlAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        adapter = SpdlAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        adapter = SpdlAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.spdl_adapter import SpdlAdapter

        adapter = SpdlAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
