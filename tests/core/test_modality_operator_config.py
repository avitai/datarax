"""Test suite for ModalityOperatorConfig.

Tests configuration validation for modality-specific operators.
Follows TDD approach - tests written first (RED phase).
"""

import pytest

from datarax.core.modality import ModalityOperatorConfig


class TestModalityOperatorConfigBasic:
    """Test basic configuration creation and validation."""

    def test_minimal_config(self):
        """Minimal config with just field_key should succeed."""
        config = ModalityOperatorConfig(field_key="image")
        assert config.field_key == "image"
        assert config.target_key is None
        assert config.auxiliary_fields is None
        assert config.clip_range is None
        assert config.preserve_auxiliary is True
        assert config.validate_domain_constraints is True

    def test_full_config(self):
        """Full config with all parameters should succeed."""
        config = ModalityOperatorConfig(
            field_key="image",
            target_key="processed_image",
            auxiliary_fields=["mask", "bounding_boxes"],
            clip_range=(0.0, 1.0),
            preserve_auxiliary=False,
            validate_domain_constraints=False,
        )
        assert config.field_key == "image"
        assert config.target_key == "processed_image"
        assert config.auxiliary_fields == ["mask", "bounding_boxes"]
        assert config.clip_range == (0.0, 1.0)
        assert config.preserve_auxiliary is False
        assert config.validate_domain_constraints is False

    def test_inherits_from_operator_config(self):
        """Config should inherit OperatorConfig attributes."""
        config = ModalityOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
            cacheable=True,
        )
        assert config.stochastic is True
        assert config.stream_name == "augment"
        assert config.cacheable is True


class TestModalityOperatorConfigFieldKeyValidation:
    """Test field_key validation rules."""

    def test_empty_field_key_raises_error(self):
        """Empty field_key should raise ValueError."""
        with pytest.raises(ValueError, match="field_key must be a non-empty string"):
            ModalityOperatorConfig(field_key="")

    def test_none_field_key_raises_error(self):
        """None field_key should raise error (caught by type system or validation)."""
        # This might raise ValueError or TypeError depending on implementation
        with pytest.raises((ValueError, TypeError)):
            ModalityOperatorConfig(field_key=None)  # type: ignore

    def test_whitespace_field_key_is_valid(self):
        """Field key with whitespace should be valid (user's responsibility)."""
        # We only check for empty string, not whitespace
        config = ModalityOperatorConfig(field_key="  ")
        assert config.field_key == "  "


class TestModalityOperatorConfigClipRangeValidation:
    """Test clip_range validation rules."""

    def test_valid_clip_range(self):
        """Valid clip_range should succeed."""
        config = ModalityOperatorConfig(
            field_key="image",
            clip_range=(0.0, 1.0),
        )
        assert config.clip_range == (0.0, 1.0)

    def test_negative_clip_range(self):
        """Negative clip range should be valid."""
        config = ModalityOperatorConfig(
            field_key="audio",
            clip_range=(-1.0, 1.0),
        )
        assert config.clip_range == (-1.0, 1.0)

    def test_none_clip_range(self):
        """None clip_range should be valid (no clipping)."""
        config = ModalityOperatorConfig(field_key="image", clip_range=None)
        assert config.clip_range is None

    def test_clip_range_min_equals_max_raises_error(self):
        """clip_range with min == max should raise ValueError."""
        with pytest.raises(ValueError, match="clip_range min .* must be < max"):
            ModalityOperatorConfig(
                field_key="image",
                clip_range=(0.5, 0.5),
            )

    def test_clip_range_min_greater_than_max_raises_error(self):
        """clip_range with min > max should raise ValueError."""
        with pytest.raises(ValueError, match="clip_range min .* must be < max"):
            ModalityOperatorConfig(
                field_key="image",
                clip_range=(1.0, 0.0),
            )

    def test_clip_range_wrong_length_raises_error(self):
        """clip_range with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="clip_range must be tuple of \\(min, max\\)"):
            ModalityOperatorConfig(
                field_key="image",
                clip_range=(0.0, 0.5, 1.0),  # type: ignore
            )

    def test_clip_range_single_value_raises_error(self):
        """clip_range with single value should raise ValueError."""
        with pytest.raises(ValueError, match="clip_range must be tuple of \\(min, max\\)"):
            ModalityOperatorConfig(
                field_key="image",
                clip_range=(0.5,),  # type: ignore
            )

    def test_clip_range_empty_tuple_raises_error(self):
        """Empty clip_range tuple should raise ValueError."""
        with pytest.raises(ValueError, match="clip_range must be tuple of \\(min, max\\)"):
            ModalityOperatorConfig(
                field_key="image",
                clip_range=(),  # type: ignore
            )


class TestModalityOperatorConfigAuxiliaryFields:
    """Test auxiliary_fields configuration."""

    def test_single_auxiliary_field(self):
        """Single auxiliary field should work."""
        config = ModalityOperatorConfig(
            field_key="image",
            auxiliary_fields=["mask"],
        )
        assert config.auxiliary_fields == ["mask"]

    def test_multiple_auxiliary_fields(self):
        """Multiple auxiliary fields should work."""
        config = ModalityOperatorConfig(
            field_key="image",
            auxiliary_fields=["mask", "bounding_boxes", "keypoints"],
        )
        assert config.auxiliary_fields == ["mask", "bounding_boxes", "keypoints"]

    def test_empty_auxiliary_fields_list(self):
        """Empty auxiliary fields list should be valid."""
        config = ModalityOperatorConfig(
            field_key="image",
            auxiliary_fields=[],
        )
        assert config.auxiliary_fields == []

    def test_none_auxiliary_fields(self):
        """None auxiliary_fields should be valid (default)."""
        config = ModalityOperatorConfig(field_key="image")
        assert config.auxiliary_fields is None


class TestModalityOperatorConfigInheritedValidation:
    """Test validation inherited from OperatorConfig."""

    def test_stochastic_requires_stream_name(self):
        """Stochastic mode requires stream_name."""
        with pytest.raises(ValueError, match="Stochastic modules require stream_name"):
            ModalityOperatorConfig(
                field_key="image",
                stochastic=True,
                stream_name=None,
            )

    def test_deterministic_forbids_stream_name(self):
        """Deterministic mode forbids stream_name."""
        with pytest.raises(
            ValueError, match="Deterministic modules should not specify stream_name"
        ):
            ModalityOperatorConfig(
                field_key="image",
                stochastic=False,
                stream_name="augment",
            )

    def test_batch_stats_fn_and_precomputed_stats_mutually_exclusive(self):
        """Cannot specify both batch_stats_fn and precomputed_stats."""
        with pytest.raises(
            ValueError, match="Cannot specify both batch_stats_fn and precomputed_stats"
        ):
            ModalityOperatorConfig(
                field_key="image",
                batch_stats_fn=lambda x: {"mean": 0.5},
                precomputed_stats={"mean": 0.5},
            )


class TestModalityOperatorConfigTargetKey:
    """Test target_key configuration."""

    def test_target_key_none_overwrites_source(self):
        """None target_key means overwrite source field."""
        config = ModalityOperatorConfig(
            field_key="image",
            target_key=None,
        )
        assert config.target_key is None

    def test_target_key_different_from_source(self):
        """Target key can be different from source."""
        config = ModalityOperatorConfig(
            field_key="image",
            target_key="processed_image",
        )
        assert config.field_key == "image"
        assert config.target_key == "processed_image"

    def test_target_key_same_as_source(self):
        """Target key can be same as source (explicit overwrite)."""
        config = ModalityOperatorConfig(
            field_key="image",
            target_key="image",
        )
        assert config.field_key == "image"
        assert config.target_key == "image"
