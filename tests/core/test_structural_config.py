"""Tests for StructuralConfig configuration class.

This test suite validates StructuralConfig used by StructuralModule for non-parametric,
structural data organization operations. Tests frozen dataclass behavior, validation,
and inheritance patterns.

Test Categories:
- Valid config construction
- Frozen dataclass behavior (immutability)
- Validation rules (__post_init__)
- Child config inheritance
"""

import pytest


# NOTE: Import will fail initially (RED phase) - this is expected!
try:
    from datarax.core.config import DataraxModuleConfig, StructuralConfig
except ImportError:
    DataraxModuleConfig = None
    StructuralConfig = None


pytestmark = pytest.mark.skipif(
    StructuralConfig is None,
    reason="StructuralConfig not implemented yet (RED phase)",
)


class TestStructuralConfigConstruction:
    """Test valid structural config construction."""

    def test_default_construction(self):
        """Test config with all defaults."""
        config = StructuralConfig()

        assert config.stochastic is False
        assert config.stream_name is None
        assert config.cacheable is False  # Inherited

    def test_deterministic_structural(self):
        """Test deterministic structural config."""
        config = StructuralConfig(stochastic=False, cacheable=False)

        assert config.stochastic is False
        assert config.stream_name is None

    def test_stochastic_structural(self):
        """Test stochastic structural config (e.g., random sampler)."""
        config = StructuralConfig(stochastic=True, stream_name="sampling")

        assert config.stochastic is True
        assert config.stream_name == "sampling"

    def test_with_precomputed_stats(self):
        """Test structural config with precomputed statistics."""
        stats = {"count": 1000, "shape": (32, 224, 224, 3)}
        config = StructuralConfig(stochastic=False, precomputed_stats=stats)

        assert config.precomputed_stats == stats


class TestStructuralConfigFrozenBehavior:
    """Test frozen dataclass immutability."""

    def test_is_frozen(self):
        """Test that StructuralConfig is frozen (immutable)."""
        config = StructuralConfig()

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.cacheable = True

    def test_frozen_enforces_compile_time_constants(self):
        """Test that frozen config represents compile-time constants."""
        config = StructuralConfig(stochastic=False, cacheable=True)

        # Values should be fixed
        assert config.cacheable is True

        # Should not be modifiable
        with pytest.raises(Exception):
            config.cacheable = False

    def test_child_frozen_config(self):
        """Test that child configs inherit runtime freezing behavior."""
        from dataclasses import dataclass
        from datarax.core.config import FrozenInstanceError

        @dataclass
        class BatcherConfig(StructuralConfig):
            """Child config for batcher (inherits runtime freezing)."""

            batch_size: int = 32

        config = BatcherConfig(batch_size=64)

        assert config.batch_size == 64

        # Should be frozen (runtime immutability inherited from parent)
        with pytest.raises(FrozenInstanceError) as exc_info:
            config.batch_size = 128

        # Error message should be informative
        assert "frozen" in str(exc_info.value).lower()

    def test_frozen_with_mutable_precomputed_stats(self):
        """Test that precomputed_stats dict itself is not frozen."""
        stats = {"count": 1000}
        config = StructuralConfig(precomputed_stats=stats)

        # Config is frozen
        with pytest.raises(Exception):
            config.cacheable = True

        # But the dict inside can be modified (not ideal but acceptable)
        stats["count"] = 2000  # Modifying original dict
        assert stats["count"] == 2000


class TestStructuralConfigValidation:
    """Test __post_init__ validation rules."""

    def test_stochastic_requires_stream_name(self):
        """Test that stochastic=True requires stream_name."""
        with pytest.raises(ValueError) as exc_info:
            StructuralConfig(stochastic=True)  # Missing stream_name

        error_msg = str(exc_info.value).lower()
        assert "stochastic" in error_msg
        assert "stream_name" in error_msg

    def test_deterministic_forbids_stream_name(self):
        """Test that stochastic=False forbids stream_name."""
        with pytest.raises(ValueError) as exc_info:
            StructuralConfig(
                stochastic=False,
                stream_name="sampling",  # Should not be specified
            )

        error_msg = str(exc_info.value).lower()
        assert "deterministic" in error_msg or "should not" in error_msg
        assert "stream_name" in error_msg

    def test_inherits_base_validation(self):
        """Test that StructuralConfig inherits DataraxModuleConfig validation."""
        # Note: With frozen=True, parent __post_init__ validation needs
        # special handling, but mutual exclusivity should still be enforced

        # Should enforce mutual exclusivity of statistics
        with pytest.raises(ValueError) as exc_info:
            StructuralConfig(batch_stats_fn=lambda x: {}, precomputed_stats={})

        error_msg = str(exc_info.value).lower()
        assert "both" in error_msg

    def test_validation_runs_before_freeze(self):
        """Test that __post_init__ validation runs before freezing."""
        # Invalid config should raise error, not create frozen instance
        with pytest.raises(ValueError):
            StructuralConfig(stochastic=True)  # Missing stream_name

        # Should not create a frozen instance with invalid state


class TestStructuralConfigInheritance:
    """Test StructuralConfig inheritance from DataraxModuleConfig."""

    def test_inherits_all_base_fields(self):
        """Test that StructuralConfig has all DataraxModuleConfig fields."""
        config = StructuralConfig()

        # Base fields should be accessible
        assert hasattr(config, "cacheable")
        assert hasattr(config, "batch_stats_fn")
        assert hasattr(config, "precomputed_stats")

        # Structural fields should be accessible
        assert hasattr(config, "stochastic")
        assert hasattr(config, "stream_name")

    def test_base_field_defaults_preserved(self):
        """Test that base field defaults are preserved."""
        config = StructuralConfig()

        # Base defaults
        assert config.cacheable is False
        assert config.batch_stats_fn is None
        assert config.precomputed_stats is None

        # Structural defaults
        assert config.stochastic is False
        assert config.stream_name is None

    def test_is_instance_of_base(self):
        """Test that StructuralConfig is instance of DataraxModuleConfig."""
        config = StructuralConfig()
        assert isinstance(config, DataraxModuleConfig)


class TestStructuralConfigChildInheritance:
    """Test child configs inheriting from StructuralConfig."""

    def test_child_config_adds_fields(self):
        """Test child config can add structural-specific fields."""
        from dataclasses import dataclass
        from datarax.core.config import FrozenInstanceError

        @dataclass
        class BatcherConfig(StructuralConfig):
            """Batcher config with batch_size parameter (inherits runtime freezing)."""

            batch_size: int = 32

            def __post_init__(self):
                # Validate before freezing (order matters!)
                if self.batch_size <= 0:
                    raise ValueError("batch_size must be positive")

                # Call parent __post_init__ which does parent validation + freezing
                super().__post_init__()

        # Valid construction
        config = BatcherConfig(batch_size=64)

        assert config.stochastic is False
        assert config.batch_size == 64

        # Frozen (runtime immutability inherited from parent)
        with pytest.raises(FrozenInstanceError):
            config.batch_size = 128

    def test_child_config_inherits_validation(self):
        """Test child config inherits all parent validation rules."""
        from dataclasses import dataclass

        @dataclass
        class ChildConfig(StructuralConfig):
            """Simple child config (inherits runtime freezing)."""

            extra_param: int = 10

        # Should enforce stochastic requires stream_name
        with pytest.raises(ValueError):
            ChildConfig(stochastic=True)  # Missing stream_name

        # Should enforce deterministic forbids stream_name
        with pytest.raises(ValueError):
            ChildConfig(stochastic=False, stream_name="sample")

    def test_child_config_with_convenience_method(self):
        """Test child config with convenience method for updates."""
        from dataclasses import dataclass, replace

        @dataclass
        class BatcherConfig(StructuralConfig):
            """Batcher config with convenience method (inherits runtime freezing)."""

            batch_size: int = 32

            def with_batch_size(self, batch_size: int) -> "BatcherConfig":
                """Create new config with different batch_size."""
                return replace(self, batch_size=batch_size)

        config = BatcherConfig(batch_size=32)
        new_config = config.with_batch_size(64)

        # Original unchanged (frozen)
        assert config.batch_size == 32

        # New config has updated value
        assert new_config.batch_size == 64

    def test_child_validation_with_frozen(self):
        """Test child validation works correctly with runtime freezing."""
        from dataclasses import dataclass

        @dataclass
        class SamplerConfig(StructuralConfig):
            """Sampler config with validation (inherits runtime freezing)."""

            num_samples: int = 100

            def __post_init__(self):
                # Child validation before parent __post_init__ (which freezes)
                if self.num_samples <= 0:
                    raise ValueError("num_samples must be positive")

                # Parent validation + freezing
                super().__post_init__()

        # Parent validation works
        with pytest.raises(ValueError):
            SamplerConfig(stochastic=True)  # Missing stream_name

        # Child validation works
        with pytest.raises(ValueError) as exc_info:
            SamplerConfig(num_samples=-10)

        assert "num_samples" in str(exc_info.value).lower()


class TestStructuralConfigDataclass:
    """Test StructuralConfig dataclass properties."""

    def test_is_dataclass(self):
        """Test that StructuralConfig is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(StructuralConfig)

    def test_is_frozen(self):
        """Test that StructuralConfig is frozen."""

        # Check if frozen in dataclass definition
        config = StructuralConfig()

        # Attempting modification should fail
        with pytest.raises(Exception):
            config.cacheable = True

    def test_repr_shows_all_fields(self):
        """Test that __repr__ includes all field values."""
        config = StructuralConfig(stochastic=False, cacheable=True)
        repr_str = repr(config)

        assert "StructuralConfig" in repr_str
        assert "stochastic" in repr_str
        assert "cacheable" in repr_str


class TestStructuralConfigDefaults:
    """Test StructuralConfig default values."""

    def test_stochastic_defaults_to_false(self):
        """Test stochastic defaults to False."""
        config = StructuralConfig()
        assert config.stochastic is False

    def test_stream_name_defaults_to_none(self):
        """Test stream_name defaults to None."""
        config = StructuralConfig()
        assert config.stream_name is None

    def test_inherited_defaults(self):
        """Test inherited defaults from DataraxModuleConfig."""
        config = StructuralConfig()
        assert config.cacheable is False
        assert config.batch_stats_fn is None
        assert config.precomputed_stats is None


# Test Count Summary
# ------------------
# TestStructuralConfigConstruction: 4 tests
# TestStructuralConfigFrozenBehavior: 5 tests
# TestStructuralConfigValidation: 4 tests
# TestStructuralConfigInheritance: 3 tests
# TestStructuralConfigChildInheritance: 4 tests
# TestStructuralConfigDataclass: 3 tests
# TestStructuralConfigDefaults: 3 tests
# ------------------
# Total: 26 tests
