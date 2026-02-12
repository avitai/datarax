"""Tests for EnergonAdapter.

TDD: Tests written first. NVIDIA Energon (Megatron-LM) is a Tier 3 adapter.
Only supports MM-1. Requires separate Megatron-LM install.
"""

import pytest

energon = pytest.importorskip("megatron.energon")


class TestEnergonAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        assert EnergonAdapter().name == "Energon"

    def test_version(self):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        v = EnergonAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        assert EnergonAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        assert EnergonAdapter().supported_scenarios() == {"MM-1"}


class TestEnergonAdapterLifecycle:
    def test_setup_teardown(self, mm1_small_config, small_multimodal_data, tmp_path):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        mm1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = EnergonAdapter()
        adapter.setup(mm1_small_config, small_multimodal_data)
        adapter.teardown()

    def test_warmup(self, mm1_small_config, small_multimodal_data, tmp_path):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        mm1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = EnergonAdapter()
        adapter.setup(mm1_small_config, small_multimodal_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(
        self,
        mm1_small_config,
        small_multimodal_data,
        tmp_path,
    ):
        from benchmarks.adapters.energon_adapter import EnergonAdapter
        from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

        mm1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = EnergonAdapter()
        adapter.setup(mm1_small_config, small_multimodal_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_teardown_is_idempotent(
        self,
        mm1_small_config,
        small_multimodal_data,
        tmp_path,
    ):
        from benchmarks.adapters.energon_adapter import EnergonAdapter

        mm1_small_config.extra["tmp_dir"] = str(tmp_path)
        adapter = EnergonAdapter()
        adapter.setup(mm1_small_config, small_multimodal_data)
        adapter.teardown()
        adapter.teardown()
