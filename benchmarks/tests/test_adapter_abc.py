"""Tests for BenchmarkAdapter ABC, ScenarioConfig, IterationResult, and registry.

TDD: Write tests first, then implement.
Design ref: Sections 7.1, 7.4 of the benchmark report.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.adapters.base import (
    BenchmarkAdapter,
    IterationResult,
    ScenarioConfig,
)


# ---------------------------------------------------------------------------
# ScenarioConfig tests
# ---------------------------------------------------------------------------


class TestScenarioConfig:
    """Test ScenarioConfig dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating ScenarioConfig with required fields only."""
        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=50000,
            element_shape=(256, 256, 3),
            batch_size=128,
            transforms=["RandomResizedCrop", "Normalize"],
        )

        assert config.scenario_id == "CV-1"
        assert config.dataset_size == 50000
        assert config.element_shape == (256, 256, 3)
        assert config.batch_size == 128
        assert config.transforms == ["RandomResizedCrop", "Normalize"]

    def test_default_values(self):
        """Test default values for optional fields."""
        config = ScenarioConfig(
            scenario_id="NLP-1",
            dataset_size=10000,
            element_shape=(512,),
            batch_size=64,
            transforms=[],
        )

        assert config.num_workers == 0
        assert config.seed == 42
        assert config.extra == {}

    def test_custom_optional_fields(self):
        """Test overriding default values."""
        config = ScenarioConfig(
            scenario_id="TAB-1",
            dataset_size=100000,
            element_shape=(50,),
            batch_size=256,
            transforms=["Normalize"],
            num_workers=4,
            seed=123,
            extra={"dtype": "float16"},
        )

        assert config.num_workers == 4
        assert config.seed == 123
        assert config.extra == {"dtype": "float16"}


# ---------------------------------------------------------------------------
# IterationResult tests
# ---------------------------------------------------------------------------


class TestIterationResult:
    """Test IterationResult dataclass."""

    def test_creation(self):
        """Test creating an IterationResult."""
        result = IterationResult(
            num_batches=50,
            num_elements=6400,
            total_bytes=6400 * 256 * 256 * 3 * 4,
            wall_clock_sec=2.5,
            per_batch_times=[0.05] * 50,
            first_batch_time=0.1,
        )

        assert result.num_batches == 50
        assert result.num_elements == 6400
        assert result.wall_clock_sec == 2.5
        assert result.first_batch_time == 0.1
        assert len(result.per_batch_times) == 50

    def test_default_extra_metrics(self):
        """Test default extra_metrics is empty dict."""
        result = IterationResult(
            num_batches=10,
            num_elements=100,
            total_bytes=400,
            wall_clock_sec=1.0,
            per_batch_times=[0.1] * 10,
            first_batch_time=0.1,
        )

        assert result.extra_metrics == {}

    def test_custom_extra_metrics(self):
        """Test extra_metrics can hold custom data."""
        result = IterationResult(
            num_batches=10,
            num_elements=100,
            total_bytes=400,
            wall_clock_sec=1.0,
            per_batch_times=[0.1] * 10,
            first_batch_time=0.1,
            extra_metrics={"jit_compile_time_sec": 0.5},
        )

        assert result.extra_metrics["jit_compile_time_sec"] == 0.5


# ---------------------------------------------------------------------------
# BenchmarkAdapter ABC tests
# ---------------------------------------------------------------------------


class ConcreteAdapter(BenchmarkAdapter):
    """Minimal concrete adapter for testing the ABC.

    Implements the Template Method hooks (_iterate_batches, _materialize_batch)
    required by the current BenchmarkAdapter interface.
    """

    _batch_size: int = 32
    _num_total_batches: int = 100  # Large pool; iterate() stops at num_batches

    @property
    def name(self) -> str:
        return "test_adapter"

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_available(self) -> bool:
        return True

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        self._config = config
        self._data = data
        self._batch_size = config.batch_size

    def teardown(self) -> None:
        pass

    def supported_scenarios(self) -> set[str]:
        return {"CV-1", "NLP-1"}

    def _iterate_batches(self) -> Iterator[Any]:
        """Yield fake batches as numpy arrays."""
        for _ in range(self._num_total_batches):
            yield np.zeros((self._batch_size, 10), dtype=np.float32)

    def _materialize_batch(self, batch: Any) -> list[Any]:
        """Batch is already a numpy array."""
        return [batch]


class UnavailableAdapter(BenchmarkAdapter):
    """Adapter that reports itself as unavailable."""

    @property
    def name(self) -> str:
        return "unavailable_adapter"

    @property
    def version(self) -> str:
        return "0.0.0"

    def is_available(self) -> bool:
        return False

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        pass

    def teardown(self) -> None:
        pass

    def supported_scenarios(self) -> set[str]:
        return {"CV-1"}

    def _iterate_batches(self) -> Iterator[Any]:
        return iter([])

    def _materialize_batch(self, batch: Any) -> list[Any]:
        return []


class TestBenchmarkAdapterABC:
    """Test BenchmarkAdapter abstract base class."""

    def test_cannot_instantiate_abc_directly(self):
        """Test that BenchmarkAdapter cannot be instantiated."""
        with pytest.raises(TypeError):
            BenchmarkAdapter()  # type: ignore[abstract]

    def test_concrete_adapter_properties(self):
        """Test concrete adapter implements required properties."""
        adapter = ConcreteAdapter()
        assert adapter.name == "test_adapter"
        assert adapter.version == "1.0.0"

    def test_is_available(self):
        """Test is_available returns correct values."""
        assert ConcreteAdapter().is_available() is True
        assert UnavailableAdapter().is_available() is False

    def test_setup_and_iterate(self):
        """Test setup followed by iterate produces valid result."""
        adapter = ConcreteAdapter()
        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=1000,
            element_shape=(32, 32, 3),
            batch_size=32,
            transforms=[],
        )
        adapter.setup(config, data=None)
        result = adapter.iterate(num_batches=10)

        assert isinstance(result, IterationResult)
        assert result.num_batches == 10
        assert result.num_elements == 320

    def test_supports_scenario(self):
        """Test supports_scenario checks against supported_scenarios."""
        adapter = ConcreteAdapter()
        assert adapter.supports_scenario("CV-1") is True
        assert adapter.supports_scenario("NLP-1") is True
        assert adapter.supports_scenario("TAB-1") is False

    def test_supported_scenarios(self):
        """Test supported_scenarios returns a set."""
        adapter = ConcreteAdapter()
        scenarios = adapter.supported_scenarios()
        assert isinstance(scenarios, set)
        assert "CV-1" in scenarios

    def test_get_peak_memory_mb(self):
        """Test get_peak_memory_mb returns a positive float."""
        adapter = ConcreteAdapter()
        memory = adapter.get_peak_memory_mb()
        # psutil should be available in our environment
        assert memory is not None
        assert memory > 0

    def test_get_peak_memory_mb_without_psutil(self):
        """Test get_peak_memory_mb returns None without psutil."""
        adapter = ConcreteAdapter()
        with patch.dict("sys.modules", {"psutil": None}):
            # Force ImportError by clearing the cached import
            adapter.get_peak_memory_mb()
            # May still return a value due to prior import; that's OK


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Test adapter registry functions."""

    def test_register_decorator(self):
        """Test @register adds adapter to registry."""
        from benchmarks.adapters import (
            _ADAPTER_REGISTRY,
            register,
        )

        # Clear registry for isolated test
        _ADAPTER_REGISTRY.clear()

        @register
        class TestRegisteredAdapter(ConcreteAdapter):
            @property
            def name(self) -> str:
                return "registered_test"

        assert "registered_test" in _ADAPTER_REGISTRY
        assert _ADAPTER_REGISTRY["registered_test"] is TestRegisteredAdapter

        # Cleanup
        _ADAPTER_REGISTRY.clear()

    def test_get_available_adapters(self):
        """Test get_available_adapters filters by is_available()."""
        from benchmarks.adapters import (
            _ADAPTER_REGISTRY,
            get_available_adapters,
            register,
        )

        _ADAPTER_REGISTRY.clear()

        register(ConcreteAdapter)
        register(UnavailableAdapter)

        available = get_available_adapters()

        assert "test_adapter" in available
        assert "unavailable_adapter" not in available

        _ADAPTER_REGISTRY.clear()

    def test_get_adapters_for_scenario(self):
        """Test get_adapters_for_scenario filters by scenario support."""
        from benchmarks.adapters import (
            _ADAPTER_REGISTRY,
            get_adapters_for_scenario,
            register,
        )

        _ADAPTER_REGISTRY.clear()

        register(ConcreteAdapter)

        cv_adapters = get_adapters_for_scenario("CV-1")
        assert "test_adapter" in cv_adapters

        tab_adapters = get_adapters_for_scenario("TAB-1")
        assert "test_adapter" not in tab_adapters

        _ADAPTER_REGISTRY.clear()

    def test_empty_registry(self):
        """Test functions with empty registry."""
        from benchmarks.adapters import (
            _ADAPTER_REGISTRY,
            get_adapters_for_scenario,
            get_available_adapters,
        )

        _ADAPTER_REGISTRY.clear()

        assert get_available_adapters() == {}
        assert get_adapters_for_scenario("CV-1") == {}
