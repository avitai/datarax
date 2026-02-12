"""Tests for JaxDataloaderAdapter.

TDD: Tests written first. jax-dataloader is a Tier 1 minimal JAX-native loader.
Only supports CV-1.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

jdl = pytest.importorskip("jax_dataloader")


class TestJaxDLAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        assert JaxDataloaderAdapter().name == "jax-dataloader"

    def test_version(self):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        v = JaxDataloaderAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        assert JaxDataloaderAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        assert JaxDataloaderAdapter().supported_scenarios() == {"CV-1"}


class TestJaxDLAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        adapter = JaxDataloaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        adapter = JaxDataloaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        adapter = JaxDataloaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        adapter = JaxDataloaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter

        adapter = JaxDataloaderAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
