"""Tests for MosaicAdapter.

TDD: Tests written first. MosaicML StreamingDataset — Tier 2 cloud/streaming.
Supports 5 scenarios. Requires MDS format conversion in setup().
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

streaming = pytest.importorskip("streaming")


class TestMosaicAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        assert MosaicAdapter().name == "MosaicML Streaming"

    def test_version(self):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        v = MosaicAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        assert MosaicAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        expected = {"CV-1", "NLP-1"}
        assert MosaicAdapter().supported_scenarios() == expected


class TestMosaicAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = MosaicAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = MosaicAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(
        self,
        cv1_small_config,
        small_image_data,
        tmp_path,
    ):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = MosaicAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.mosaic_adapter import MosaicAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = MosaicAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
