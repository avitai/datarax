"""Tests for DaliAdapter.

TDD: Tests written first. NVIDIA DALI is a Tier 2 GPU-accelerated loader.
Requires CUDA. Supports 7 scenarios.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

nvidia_dali = pytest.importorskip("nvidia.dali")

# DALI requires CUDA — skip all tests if no GPU available
pytestmark = pytest.mark.skipif(
    nvidia_dali.types.NO_GPU is not False if hasattr(nvidia_dali.types, "NO_GPU") else True,
    reason="DALI requires CUDA GPU",
)

try:
    import nvidia.dali.fn as fn  # noqa: F401

    _has_cuda = True
except Exception:
    _has_cuda = False

if not _has_cuda:
    pytest.skip("DALI CUDA not available", allow_module_level=True)


class TestDaliAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        assert DaliAdapter().name == "NVIDIA DALI"

    def test_version(self):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        v = DaliAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        assert DaliAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        expected = {
            "CV-1",
            "CV-2",
            "CV-3",
            "CV-4",
            "MM-2",
            "PC-1",
            "DIST-1",
            "AUG-1",
            "AUG-2",
            "AUG-3",
        }
        assert DaliAdapter().supported_scenarios() == expected


class TestDaliAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        adapter = DaliAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        adapter = DaliAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        adapter = DaliAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        adapter = DaliAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()
