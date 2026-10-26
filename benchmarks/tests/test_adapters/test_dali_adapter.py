"""Tests for DaliAdapter.

NVIDIA DALI is a Tier 2 GPU-accelerated loader. ``is_available`` reports DALI usable only when
DALI's own device count finds a CUDA device, which the availability tests check wherever DALI is
installed; the property and lifecycle tests need that device and skip without it.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result


# ``nvidia.dali`` may exist only as an empty namespace stub created by the
# nvidia-*-cu12 CUDA packages, so import a real DALI submodule to skip cleanly
# when DALI itself is not installed.
pytest.importorskip("nvidia.dali")
dali_backend = pytest.importorskip("nvidia.dali.backend")


def _dali_sees_a_cuda_device() -> bool:
    from benchmarks.adapters.dali_adapter import DaliAdapter

    return DaliAdapter().is_available()


requires_dali_gpu = pytest.mark.skipif(
    not _dali_sees_a_cuda_device(), reason="DALI sees no CUDA device"
)


class TestDaliAvailability:
    """``is_available`` follows DALI's device count, which needs no GPU to test."""

    def test_no_cuda_device_is_unavailable(self, monkeypatch):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        monkeypatch.setattr(dali_backend, "GetCUDADeviceCount", lambda: 0)

        assert DaliAdapter().is_available() is False

    def test_a_cuda_device_is_available(self, monkeypatch):
        from benchmarks.adapters.dali_adapter import DaliAdapter

        monkeypatch.setattr(dali_backend, "GetCUDADeviceCount", lambda: 1)

        assert DaliAdapter().is_available() is True


@requires_dali_gpu
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
        from benchmarks.tests.test_adapters.conftest import assert_supported_scenarios

        assert_supported_scenarios(
            DaliAdapter(),
            must_include={"CV-1", "NLP-1", "HCV-1", "HPC-1"},
        )


@requires_dali_gpu
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
