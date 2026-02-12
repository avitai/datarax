"""Tests for DeepLakeAdapter.

TDD: Tests written first. Deep Lake is a Tier 3 AI database.
Only supports CV-1.

Deep Lake v4 requires filesystem storage (no in-memory mode),
so tests pass ``tmp_path`` via ``config.extra["tmp_dir"]``.

.. warning:: Import order matters

   Deep Lake's native extension (aws-c-cal) crashes with a fatal OpenSSL
   assertion if TensorFlow/BoringSSL is loaded first.  The ``importorskip``
   below must execute **before** any TensorFlow import in the same process.
   When running via ``--all-suites``, ``tests/conftest.py`` pre-imports
   deeplake to ensure correct ordering.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

deeplake = pytest.importorskip("deeplake")


class TestDeepLakeAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        assert DeepLakeAdapter().name == "Deep Lake"

    def test_version(self):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        v = DeepLakeAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        assert DeepLakeAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        assert DeepLakeAdapter().supported_scenarios() == {"CV-1"}


class TestDeepLakeAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = DeepLakeAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data, tmp_path):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = DeepLakeAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(
        self,
        cv1_small_config,
        small_image_data,
        tmp_path,
    ):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = DeepLakeAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(
        self,
        cv1_small_config,
        small_image_data,
        tmp_path,
    ):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = DeepLakeAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(
        self,
        cv1_small_config,
        small_image_data,
        tmp_path,
    ):
        from benchmarks.adapters.deep_lake_adapter import DeepLakeAdapter

        cv1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = DeepLakeAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
