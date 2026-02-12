"""Tests for GrainAdapter.

TDD: Tests written first. Design ref: Section 7.3 of the benchmark report.
Grain is a Tier 1 (JAX ecosystem) alternative — already a base dependency.
"""

import pytest

from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result

grain = pytest.importorskip("grain")


class TestGrainAdapterProperties:
    def test_name(self):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        assert GrainAdapter().name == "Google Grain"

    def test_version(self):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        v = GrainAdapter().version
        assert isinstance(v, str) and len(v) > 0

    def test_is_available(self):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        assert GrainAdapter().is_available() is True

    def test_supported_scenarios(self):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        expected = {
            "CV-1",
            "NLP-1",
            "TAB-1",
            "DIST-1",
            "PR-1",
            "AUG-1",
            "AUG-2",
            "AUG-3",
        }
        assert GrainAdapter().supported_scenarios() == expected


class TestGrainAdapterLifecycle:
    def test_setup_teardown(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        adapter = GrainAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()

    def test_warmup(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        adapter = GrainAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=2)
        adapter.teardown()

    def test_iterate_returns_iteration_result(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        adapter = GrainAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=5)

        assert_valid_iteration_result(result)
        adapter.teardown()

    def test_iterate_respects_num_batches(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        adapter = GrainAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches <= 3
        adapter.teardown()

    def test_teardown_is_idempotent(self, cv1_small_config, small_image_data):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        adapter = GrainAdapter()
        adapter.setup(cv1_small_config, small_image_data)
        adapter.teardown()
        adapter.teardown()


class TestGrainAdapterNLP:
    def test_token_scenario(self, nlp1_small_config, small_token_data):
        from benchmarks.adapters.grain_adapter import GrainAdapter

        adapter = GrainAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=1)
        result = adapter.iterate(num_batches=3)
        assert result.num_batches > 0
        adapter.teardown()
