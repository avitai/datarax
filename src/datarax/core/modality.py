"""Modality-specific operator base classes.

This module provides base classes for operators that work on single modalities
(specific fields within elements). Each ModalityOperator handles ONE field
(e.g., image, text, audio) and can have learnable parameters via Flax NNX.

Key Features:

- Single-field transformations with coordinate auxiliary fields
- Value domain constraints and clipping
- Learnable parameters support via Flax NNX
- Compatible with JAX transformations (jit, vmap, grad)
- End-to-end differentiable data pipelines

Examples:
Examples:
    Deterministic image operator:

    ```python
    config = ModalityOperatorConfig(field_key="image", clip_range=(0.0, 1.0))
    # Note: Use specific operators like BrightnessOperator, ContrastOperator, etc.
    ```

    Stochastic audio operator with learnable parameters:

    ```python
    config = ModalityOperatorConfig(
        field_key="waveform",
        stochastic=True,
        stream_name="augment"
    )
    operator = LearnedAudioOperator(config, rngs=nnx.Rngs(0, augment=1))
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


@dataclass
class ModalityOperatorConfig(OperatorConfig):
    """Configuration for modality-specific operators.

    Captures common patterns across modalities:

        - Field identification (primary + auxiliary)
        - Value domain constraints
        - Transformation coordination settings
        - Learnable parameter support via Flax NNX

    Each ModalityOperator handles a SINGLE modality (specific field in Element).
    Multi-modal data handled by composing multiple ModalityOperators via
    CompositeOperatorModule or MultiModalOperator.

    IMPORTANT: Operators can have learnable parameters and must be compatible
    with JAX transformations (jit, vmap, grad).

    Attributes:
        field_key: Primary data field to transform (e.g., "image", "caption", "waveform")
        target_key: Optional target field (if None, overwrites source field)
        auxiliary_fields: Fields that should be transformed coordinately
                         (e.g., ["mask", "bounding_boxes"] for image)
        clip_range: Value domain constraints as (min, max) tuple (None = no clipping)
        preserve_auxiliary: Whether to preserve auxiliary data structure during transformation
        validate_domain_constraints: Enable domain-specific validation rules
                                    (e.g., biological validity, clinical plausibility)

    Validation Rules:

        - field_key must be non-empty string
        - clip_range must be tuple of (min, max) where min < max
        - Inherits stochastic validation from OperatorConfig
        - Inherits statistics validation from DataraxModuleConfig

    Examples:
        Minimal image operator:

        ```python
        config = ModalityOperatorConfig(field_key="image")
        ```

        Image with clipping and auxiliary fields:

        ```python
        config = ModalityOperatorConfig(
            field_key="image",
            auxiliary_fields=["mask", "bounding_boxes"],
            clip_range=(0.0, 1.0)
        )
        ```

        Stochastic text operator:

        ```python
        config = ModalityOperatorConfig(
            field_key="caption",
            stochastic=True,
            stream_name="text_augment"
        )
        ```
    """

    # Primary data field (single modality, single field)
    # Use kw_only to allow required field after parent's optional fields
    field_key: str = field(kw_only=True)

    # Optional target field (if None, overwrites source)
    target_key: str | None = field(default=None, kw_only=True)

    # Auxiliary fields that should be transformed coordinately
    auxiliary_fields: list[str] | None = field(default=None, kw_only=True)

    # Value domain constraints (None = no clipping)
    clip_range: tuple[float, float] | None = field(default=None, kw_only=True)

    # Whether to preserve auxiliary data structure during transformation
    preserve_auxiliary: bool = field(default=True, kw_only=True)

    # Domain-specific validation rules (handled by subclasses)
    validate_domain_constraints: bool = field(default=True, kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters.

        Validates:
        1. Parent configuration (OperatorConfig stochastic/statistics rules)
        2. field_key is non-empty string
        3. clip_range is valid tuple of (min, max) where min < max

        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation first (statistics, stochastic rules)
        super().__post_init__()

        # Validate field_key is non-empty
        if not self.field_key:
            raise ValueError("field_key must be a non-empty string")

        # Validate clip_range if provided
        if self.clip_range is not None:
            if len(self.clip_range) != 2:
                raise ValueError(f"clip_range must be tuple of (min, max), got {self.clip_range}")
            min_val, max_val = self.clip_range
            if min_val >= max_val:
                raise ValueError(f"clip_range min ({min_val}) must be < max ({max_val})")


class ModalityOperator(OperatorModule):
    """Base class for modality-specific operators with learnable parameter support.

    Provides common functionality:

        - Field extraction and remapping
        - Value range validation and clipping
        - Auxiliary field coordination
        - Integration with MapOperator for transformations
        - Learnable parameter support via Flax NNX

    Subclasses (ImageOperator, AudioOperator, etc.) provide:

        - Modality-specific configurations
        - Domain-specific transformation functions
        - Validation logic for modality data
        - Learnable parameters (e.g., augmentation strategies, normalization params)

    Key Features:

        - Compatible with nnx.jit, jax.vmap, jax.grad
        - Supports learnable parameters via nnx.Param
        - End-to-end differentiable data pipelines
        - Can be optimized jointly with model
        - Operates on Batch[Element] (inherited from OperatorModule)

    Inherited Features from OperatorModule:

        - **apply_batch()**: Automatically handles batched operations by calling apply()
          on each element. Override only if you need custom batch-level logic (e.g.,
          batch normalization, cross-element operations). Default is sufficient for
          most element-wise transformations.

        - **Statistics system**: Optionally collect and use batch statistics via stats
          parameter in apply(). Useful for adaptive operations (e.g., batch-aware
          normalization). Statistics are computed externally and passed in.

        - **Caching system**: Results can be cached based on operator configuration
          and input characteristics. Inherited from base OperatorModule, helps avoid
          redundant computation for deterministic operators.

    Subclass Implementation Pattern:
        ```python
        class ImageOperator(ModalityOperator):
            def __init__(self, config: ModalityOperatorConfig, *, rngs: nnx.Rngs | None = None):
                super().__init__(config, rngs=rngs)
                # Add learnable parameters if needed
                # self.augment_strength = nnx.Param(jnp.array(0.5))

            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Extract field
                image = self._extract_field(data, self.config.field_key)

                # Transform (can use learnable parameters)
                transformed = self._transform_image(image)

                # Apply clipping
                transformed = self._apply_clip_range(transformed)

                # Remap to target field
                result = self._remap_field(data, transformed)

                return result, state, metadata

            def generate_random_params(self, rng, data_shapes):
                # For stochastic operators only
                batch_size = data_shapes[self.config.field_key][0]
                return jax.random.uniform(rng, (batch_size,))
        ```

    Examples:
        Deterministic operator (no learnable params):

        ```python
        image_op = ImageOperator(config, rngs=nnx.Rngs(0))
        ```

        Learnable operator (learned augmentation strategy):

        ```python
        class LearnedImageOperator(ImageOperator):
            def __init__(self, config, *, rngs):
                super().__init__(config, rngs=rngs)
                # Learnable augmentation parameters
                self.crop_scale = nnx.Param(jnp.array(0.8))
                self.rotation_angle = nnx.Param(jnp.array(0.1))
        ```

        Custom batch-level operator (rare, only when needed):

        ```python
        class BatchNormOperator(ImageOperator):
            def apply_batch(self, batch, stats=None):
                # Override for batch-level normalization
                # Compute batch statistics here
                # Call apply() for each element with shared stats
                pass
        ```
    """

    def __init__(
        self,
        config: ModalityOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize ModalityOperator.

        Args:
            config: Modality operator configuration (already validated)
            rngs: Random number generators (required if stochastic=True)
            name: Optional operator name

        Raises:
            ValueError: If stochastic=True but rngs is None
        """
        super().__init__(config, rngs=rngs, name=name)
        self.config: ModalityOperatorConfig = config
        # Subclasses can add learnable parameters here via nnx.Param

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply modality-specific transformation to element.

        MUST be implemented by subclasses to provide modality-specific behavior.

        This is a PURE FUNCTION that transforms a single data element.
        It should not access self.rngs or generate random numbers.
        All randomness comes through random_params argument.

        Args:
            data: Element data PyTree (contains field specified by config.field_key)
                 Typically dict[str, Array] with no batch dimension
            state: Element state PyTree (typically dict[str, Any])
            metadata: Element metadata dict
            random_params: Random parameters for this element (from generate_random_params)
            stats: Optional batch statistics (from get_statistics() or passed explicitly)

        Returns:
            Tuple of (transformed_data, new_state, new_metadata)
            - transformed_data: PyTree with same structure as data, containing transformed field
            - new_state: Updated state PyTree
            - new_metadata: Updated metadata dict

        Implementation Pattern:
            ```python
            def apply(self, data, state, metadata, random_params=None, stats=None):
                # 1. Extract field
                field_value = self._extract_field(data, self.config.field_key)

                # 2. Transform (modality-specific logic)
                transformed = self._transform(field_value, random_params, stats)

                # 3. Apply clipping if configured
                transformed = self._apply_clip_range(transformed)

                # 4. Remap to target field
                result = self._remap_field(data, transformed)

                return result, state, metadata
            ```

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply()")

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> PyTree:
        """Generate random parameters for stochastic transformations.

        MUST be implemented by stochastic operators (config.stochastic=True).
        Deterministic operators can use default implementation (returns None).

        Generates PyTree of random parameters matching batch structure. For example,
        image rotation might generate per-element rotation angles.

        This method is impure (uses RNG) and called once per batch. The generated
        parameters are then passed to apply() for each element via vmap.

        Args:
            rng: JAX random key for this batch
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image": (batch_size, H, W, C)}

        Returns:
            PyTree of random parameters for this batch.
            Structure depends on operator needs.
            For deterministic operators, returns None.

        Examples:
            ```python
            def generate_random_params(self, rng, data_shapes):
                # Stochastic rotation: generate per-element angles
                batch_size = data_shapes[self.config.field_key][0]
                return jax.random.uniform(rng, (batch_size,), minval=0, maxval=2*jnp.pi)
            ```

        Raises:
            NotImplementedError: If stochastic=True but not implemented
        """
        # Default implementation for deterministic operators
        return super().generate_random_params(rng, data_shapes)

    def _extract_field(self, data: dict, field_key: str) -> Any:
        """Extract field from data dict with validation.

        Helper method for subclasses to safely extract fields from data.
        Supports nested field access using dot notation (e.g., "data.image").

        Args:
            data: Data dictionary containing fields
            field_key: Key of field to extract (supports dot notation for nested access)

        Returns:
            Field value from data

        Raises:
            KeyError: If field_key not found in data

        Examples:
            ```python
            image = self._extract_field(data, self.config.field_key)
            mask = self._extract_field(data, "mask")
            nested = self._extract_field(data, "data.image")  # Access data["data"]["image"]
            ```
        """
        # Support nested field access with dot notation
        if "." in field_key:
            keys = field_key.split(".")
            current = data
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    raise KeyError(f"Field '{field_key}' not found in data")
                current = current[key]
            return current
        else:
            # Simple field access
            if field_key not in data:
                raise KeyError(f"Field '{field_key}' not found in data")
            return data[field_key]

    def _apply_clip_range(self, value: jax.Array) -> jax.Array:
        """Apply value range clipping if configured.

        Helper method to apply clip_range constraints from config.
        If clip_range is None, returns value unchanged.

        Args:
            value: Array to clip

        Returns:
            Clipped array (or original if clip_range is None)

        Examples:
            ```python
            transformed = self._apply_clip_range(transformed_image)
            ```
        """
        if self.config.clip_range is not None:
            min_val, max_val = self.config.clip_range
            return jnp.clip(value, min_val, max_val)
        return value

    def _remap_field(
        self,
        data: dict,
        transformed_value: Any,
    ) -> dict:
        """Store transformed value in target field.

        Helper method to store transformation result in correct field.
        If target_key is None, overwrites source field.
        Otherwise, creates/updates target field and preserves source.
        Supports nested field paths using dot notation (e.g., "data.image").

        Args:
            data: Original data dictionary
            transformed_value: Transformed field value to store

        Returns:
            New data dictionary with transformed value stored

        Examples:
            ```python
            result = self._remap_field(data, transformed_image)  # Overwrite source
            result = self._remap_field(data, transformed_image)  # New field
            result = self._remap_field(data, transformed_image)  # Nested field
            ```
        """
        target_key = self.config.target_key or self.config.field_key

        # Support nested field paths with dot notation
        if "." in target_key:
            keys = target_key.split(".")

            # Recursively rebuild nested dict structure
            def set_nested(current_data: dict, key_path: list[str], value: Any) -> dict:
                if len(key_path) == 1:
                    # Base case: set the value
                    return {**current_data, key_path[0]: value}
                else:
                    # Recursive case: navigate deeper
                    key = key_path[0]
                    rest = key_path[1:]

                    # Get existing nested dict or create empty one
                    nested = current_data.get(key, {})
                    if not isinstance(nested, dict):
                        raise ValueError(
                            f"Cannot create nested path '{target_key}': '{key}' is not a dict"
                        )

                    # Recursively update nested structure
                    updated_nested = set_nested(nested, rest, value)
                    return {**current_data, key: updated_nested}

            return set_nested(data, keys, transformed_value)
        else:
            # Simple field assignment (shallow copy + update)
            return {**data, target_key: transformed_value}
