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

from dataclasses import fields

import pytest

from datarax.core.config import DataraxModuleConfig, FrozenInstanceError, StructuralConfig


class TestStructuralConfigConstruction:
    """Test valid structural config construction."""

    def test_default_construction(self):
        """Test config with all defaults."""
        config = StructuralConfig()

        assert config.stochastic is False
        assert config.stream_name is None

    def test_deterministic_structural(self):
        """Test deterministic structural config."""
        config = StructuralConfig(stochastic=False)

        assert config.stochastic is False
        assert config.stream_name is None

    def test_stochastic_structural(self):
        """Test stochastic structural config (e.g., random sampler)."""
        config = StructuralConfig(stochastic=True, stream_name="sampling")

        assert config.stochastic is True
        assert config.stream_name == "sampling"


class TestStructuralConfigFrozenBehavior:
    """Test frozen dataclass immutability."""

    def test_is_frozen(self):
        """Test that StructuralConfig is frozen (immutable)."""
        config = StructuralConfig()

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            config.stochastic = True  # type: ignore[reportAttributeAccessIssue]

    def test_frozen_enforces_compile_time_constants(self):
        """Test that frozen config represents compile-time constants."""
        config = StructuralConfig(stochastic=False, stream_name=None)

        # Values should be fixed
        assert config.stochastic is False

        # Should not be modifiable
        with pytest.raises(FrozenInstanceError):
            config.stochastic = True  # type: ignore[reportAttributeAccessIssue]

    def test_child_frozen_config(self):
        """Test that child configs inherit runtime freezing behavior."""
        from dataclasses import dataclass

        from datarax.core.config import FrozenInstanceError

        @dataclass(frozen=True)
        class BatcherConfig(StructuralConfig):  # type: ignore[reportGeneralTypeIssues]
            """Child config for batcher (inherits runtime freezing)."""

            batch_size: int = 32

        config = BatcherConfig(batch_size=64)

        assert config.batch_size == 64

        # Should be frozen (runtime immutability inherited from parent)
        with pytest.raises(FrozenInstanceError) as exc_info:
            config.batch_size = 128  # type: ignore[reportAttributeAccessIssue]

        # Error message should mention the field
        assert "batch_size" in str(exc_info.value)

    def test_frozen_config_rejects_assignment(self):
        """Test that a constructed config refuses every field assignment."""
        config = StructuralConfig(stochastic=True, stream_name="sampling")

        with pytest.raises(FrozenInstanceError):
            config.stochastic = False  # type: ignore[reportAttributeAccessIssue]

        with pytest.raises(FrozenInstanceError):
            config.stream_name = "other"  # type: ignore[reportAttributeAccessIssue]


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
        """Test that StructuralConfig runs its parent's __post_init__ under frozen=True."""
        with pytest.raises(ValueError) as exc_info:
            StructuralConfig(stochastic=False, stream_name="sampling")

        error_msg = str(exc_info.value).lower()
        assert "stream_name" in error_msg

    def test_validation_runs_before_freeze(self):
        """Test that __post_init__ validation runs before freezing."""
        # Invalid config should raise error, not create frozen instance
        with pytest.raises(ValueError):
            StructuralConfig(stochastic=True)  # Missing stream_name

        # Should not create a frozen instance with invalid state


class TestStructuralConfigInheritance:
    """Test StructuralConfig inheritance from DataraxModuleConfig."""

    def test_declares_only_structural_fields(self):
        """A structural config says how to run the module, never what it fitted to data.

        Statistics live on the operator that applies them, so a config stays static metadata
        that every transform can compare.
        """
        assert {field.name for field in fields(StructuralConfig)} == {
            "stochastic",
            "stream_name",
        }

    def test_structural_field_defaults(self):
        """Test that structural field defaults are preserved."""
        config = StructuralConfig()

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

        @dataclass(frozen=True)
        class BatcherConfig(StructuralConfig):  # type: ignore[reportGeneralTypeIssues]
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
            config.batch_size = 128  # type: ignore[reportAttributeAccessIssue]

    def test_child_config_inherits_validation(self):
        """Test child config inherits all parent validation rules."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ChildConfig(StructuralConfig):  # type: ignore[reportGeneralTypeIssues]
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

        @dataclass(frozen=True)
        class BatcherConfig(StructuralConfig):  # type: ignore[reportGeneralTypeIssues]
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

        @dataclass(frozen=True)
        class SamplerConfig(StructuralConfig):  # type: ignore[reportGeneralTypeIssues]
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
        with pytest.raises(FrozenInstanceError):
            config.stochastic = True  # type: ignore[reportAttributeAccessIssue]

    def test_repr_shows_all_fields(self):
        """Test that __repr__ includes all field values."""
        config = StructuralConfig(stochastic=False)
        repr_str = repr(config)

        assert "StructuralConfig" in repr_str
        assert "stochastic" in repr_str
        assert "stream_name" in repr_str


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

    def test_cacheable_is_not_a_structural_field(self):
        """Caching belongs to SamplerConfig, the only module kind that memoizes its result."""
        assert not hasattr(StructuralConfig(), "cacheable")
