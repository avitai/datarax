"""Test suite for CrossModalOperatorConfig.

Tests configuration validation for cross-modal operators.
Follows TDD approach - tests written first (RED phase).
"""

import pytest

from datarax.core.cross_modal import CrossModalOperatorConfig


class TestCrossModalOperatorConfigBasic:
    """Test basic configuration creation and validation."""

    def test_minimal_config(self):
        """Minimal config with input_fields and output_fields should succeed."""
        config = CrossModalOperatorConfig(
            input_fields=["image_embedding"],
            output_fields=["processed_embedding"],
        )
        assert config.input_fields == ["image_embedding"]
        assert config.output_fields == ["processed_embedding"]
        assert config.operation == "fusion"
        assert config.validate_alignment is True

    def test_full_config(self):
        """Full config with all parameters should succeed."""
        config = CrossModalOperatorConfig(
            input_fields=["image_embedding", "text_embedding"],
            output_fields=["fused_embedding", "similarity_score"],
            operation="cross_attention",
            validate_alignment=False,
        )
        assert config.input_fields == ["image_embedding", "text_embedding"]
        assert config.output_fields == ["fused_embedding", "similarity_score"]
        assert config.operation == "cross_attention"
        assert config.validate_alignment is False

    def test_inherits_from_operator_config(self):
        """Config should inherit OperatorConfig attributes."""
        config = CrossModalOperatorConfig(
            input_fields=["image_emb"],
            output_fields=["output"],
            stochastic=True,
            stream_name="fusion",
            cacheable=True,
        )
        assert config.stochastic is True
        assert config.stream_name == "fusion"
        assert config.cacheable is True


class TestCrossModalOperatorConfigInputFieldsValidation:
    """Test input_fields validation rules."""

    def test_single_input_field(self):
        """Single input field should be valid."""
        config = CrossModalOperatorConfig(
            input_fields=["embedding"],
            output_fields=["output"],
        )
        assert config.input_fields == ["embedding"]

    def test_multiple_input_fields(self):
        """Multiple input fields should be valid."""
        config = CrossModalOperatorConfig(
            input_fields=["image_emb", "text_emb", "audio_emb"],
            output_fields=["fused"],
        )
        assert config.input_fields == ["image_emb", "text_emb", "audio_emb"]

    def test_empty_input_fields_raises_error(self):
        """Empty input_fields list should raise ValueError."""
        with pytest.raises(ValueError, match="input_fields must be a non-empty list"):
            CrossModalOperatorConfig(
                input_fields=[],
                output_fields=["output"],
            )

    def test_none_input_fields_raises_error(self):
        """None input_fields should raise error."""
        # This will be caught by type system or validation
        with pytest.raises((ValueError, TypeError)):
            CrossModalOperatorConfig(
                input_fields=None,  # type: ignore
                output_fields=["output"],
            )

    def test_input_field_empty_string_raises_error(self):
        """Empty string in input_fields should raise ValueError."""
        with pytest.raises(ValueError, match="All input_fields must be non-empty strings"):
            CrossModalOperatorConfig(
                input_fields=["valid_field", ""],
                output_fields=["output"],
            )

    def test_input_field_non_string_raises_error(self):
        """Non-string in input_fields should raise ValueError."""
        with pytest.raises(ValueError, match="All input_fields must be non-empty strings"):
            CrossModalOperatorConfig(
                input_fields=["valid_field", 123],  # type: ignore
                output_fields=["output"],
            )


class TestCrossModalOperatorConfigOutputFieldsValidation:
    """Test output_fields validation rules."""

    def test_single_output_field(self):
        """Single output field should be valid."""
        config = CrossModalOperatorConfig(
            input_fields=["input"],
            output_fields=["output"],
        )
        assert config.output_fields == ["output"]

    def test_multiple_output_fields(self):
        """Multiple output fields should be valid."""
        config = CrossModalOperatorConfig(
            input_fields=["image_emb", "text_emb"],
            output_fields=["fused_emb", "similarity", "alignment_score"],
        )
        assert config.output_fields == ["fused_emb", "similarity", "alignment_score"]

    def test_empty_output_fields_raises_error(self):
        """Empty output_fields list should raise ValueError."""
        with pytest.raises(ValueError, match="output_fields must be a non-empty list"):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=[],
            )

    def test_none_output_fields_raises_error(self):
        """None output_fields should raise error."""
        with pytest.raises((ValueError, TypeError)):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=None,  # type: ignore
            )

    def test_output_field_empty_string_raises_error(self):
        """Empty string in output_fields should raise ValueError."""
        with pytest.raises(ValueError, match="All output_fields must be non-empty strings"):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=["valid_field", ""],
            )

    def test_output_field_non_string_raises_error(self):
        """Non-string in output_fields should raise ValueError."""
        with pytest.raises(ValueError, match="All output_fields must be non-empty strings"):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=["valid_field", None],  # type: ignore
            )


class TestCrossModalOperatorConfigOperationType:
    """Test operation type configuration."""

    def test_default_operation_is_fusion(self):
        """Default operation should be 'fusion'."""
        config = CrossModalOperatorConfig(
            input_fields=["input"],
            output_fields=["output"],
        )
        assert config.operation == "fusion"

    def test_custom_operation_types(self):
        """Various operation types should be accepted."""
        operations = ["fusion", "cross_attention", "contrastive", "alignment", "custom_op"]

        for op in operations:
            config = CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=["output"],
                operation=op,
            )
            assert config.operation == op


class TestCrossModalOperatorConfigValidateAlignment:
    """Test validate_alignment configuration."""

    def test_default_validate_alignment_is_true(self):
        """Default validate_alignment should be True."""
        config = CrossModalOperatorConfig(
            input_fields=["input"],
            output_fields=["output"],
        )
        assert config.validate_alignment is True

    def test_validate_alignment_can_be_disabled(self):
        """validate_alignment can be set to False."""
        config = CrossModalOperatorConfig(
            input_fields=["input"],
            output_fields=["output"],
            validate_alignment=False,
        )
        assert config.validate_alignment is False


class TestCrossModalOperatorConfigInheritedValidation:
    """Test validation inherited from OperatorConfig."""

    def test_stochastic_requires_stream_name(self):
        """Stochastic mode requires stream_name."""
        with pytest.raises(ValueError, match="Stochastic modules require stream_name"):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=["output"],
                stochastic=True,
                stream_name=None,
            )

    def test_deterministic_forbids_stream_name(self):
        """Deterministic mode forbids stream_name."""
        with pytest.raises(
            ValueError, match="Deterministic modules should not specify stream_name"
        ):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=["output"],
                stochastic=False,
                stream_name="fusion",
            )

    def test_batch_stats_fn_and_precomputed_stats_mutually_exclusive(self):
        """Cannot specify both batch_stats_fn and precomputed_stats."""
        with pytest.raises(
            ValueError, match="Cannot specify both batch_stats_fn and precomputed_stats"
        ):
            CrossModalOperatorConfig(
                input_fields=["input"],
                output_fields=["output"],
                batch_stats_fn=lambda x: {"mean": 0.5},
                precomputed_stats={"mean": 0.5},
            )


class TestCrossModalOperatorConfigRealWorldExamples:
    """Test real-world configuration examples."""

    def test_fusion_config(self):
        """Fusion operator configuration."""
        config = CrossModalOperatorConfig(
            input_fields=["image_embedding", "text_embedding"],
            output_fields=["fused_embedding"],
            operation="fusion",
        )
        assert len(config.input_fields) == 2
        assert len(config.output_fields) == 1
        assert config.operation == "fusion"

    def test_cross_attention_config(self):
        """Cross-attention operator configuration."""
        config = CrossModalOperatorConfig(
            input_fields=["query_features", "key_features", "value_features"],
            output_fields=["attended_features"],
            operation="cross_attention",
        )
        assert len(config.input_fields) == 3
        assert len(config.output_fields) == 1
        assert config.operation == "cross_attention"

    def test_contrastive_config(self):
        """Contrastive learning operator configuration."""
        config = CrossModalOperatorConfig(
            input_fields=["anchor_emb", "positive_emb", "negative_emb"],
            output_fields=["similarity_score"],
            operation="contrastive",
        )
        assert len(config.input_fields) == 3
        assert len(config.output_fields) == 1
        assert config.operation == "contrastive"

    def test_multi_output_config(self):
        """Operator with multiple outputs."""
        config = CrossModalOperatorConfig(
            input_fields=["image_emb", "text_emb"],
            output_fields=["fused_emb", "similarity", "alignment_loss"],
            operation="fusion",
        )
        assert len(config.input_fields) == 2
        assert len(config.output_fields) == 3
