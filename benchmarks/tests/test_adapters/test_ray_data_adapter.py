"""Tests for RayDataAdapter.

TDD: Tests written first. Ray Data is a Tier 3 distributed actor-based loader.
Supports 3 scenarios.

Follows Ray's own ``shutdown_only`` testing pattern: each test calls
``ray.init()`` itself; the ``ray_shutdown`` fixture ensures ``ray.shutdown()``
runs after every test.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

ray = pytest.importorskip("ray")


class TestRayDataAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        assert RayDataAdapter().name == "Ray Data"

    def test_version(self):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        v = RayDataAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        # Ray must be importable for this to be True
        assert RayDataAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        expected = {"NLP-1", "TAB-1"}
        assert RayDataAdapter().supported_scenarios() == expected


class TestRayDataAdapterLifecycle:
    """Lifecycle tests — each test inits its own Ray cluster.

    The ``ray_shutdown`` fixture (from conftest) guarantees cleanup.
    """

    def test_setup_teardown(self, ray_shutdown, cv1_small_config, small_image_data):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        adapter = RayDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, ray_shutdown, cv1_small_config, small_image_data):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        adapter = RayDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(
        self,
        ray_shutdown,
        cv1_small_config,
        small_image_data,
    ):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        adapter = RayDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(
        self,
        ray_shutdown,
        cv1_small_config,
        small_image_data,
    ):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        adapter = RayDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(
        self,
        ray_shutdown,
        cv1_small_config,
        small_image_data,
    ):
        from benchmarks.adapters.ray_data_adapter import RayDataAdapter

        adapter = RayDataAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
