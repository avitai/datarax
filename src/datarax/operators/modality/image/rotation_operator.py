"""RotationOperator - Image rotation augmentation for modality operators.

This module provides rotation augmentation with:
- Configurable angle ranges (deterministic and stochastic)
- Bilinear interpolation for smooth rotation
- Fill value for empty areas after rotation
- Support for 2D (grayscale) and 3D (RGB) images
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.operators.modality.image import functional


@dataclass
class RotationOperatorConfig(ModalityOperatorConfig):
    """Configuration for rotation augmentation operator.

    Attributes:
        field_key: Field to rotate (e.g., "image" or "data.image" for nested).
        angle_range: Tuple of (min_angle, max_angle) in degrees for rotation range.
                     Positive angles rotate counter-clockwise.
        fill_value: Value to fill empty areas after rotation (default: 0.0).
        interpolation: Interpolation mode (currently only "bilinear" supported).
        clip_range: Range to clip output values (default: (0.0, 1.0)).
        stochastic: Whether to use random angle sampling.
        stream_name: RNG stream name (required if stochastic=True).
    """

    # Rotation parameters
    angle_range: tuple[float, float] = (-15.0, 15.0)
    fill_value: float = 0.0
    interpolation: str = "bilinear"

    # Clip range (inherited from ModalityOperatorConfig, default (0.0, 1.0))
    clip_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate angle_range order
        if self.angle_range[0] > self.angle_range[1]:
            raise ValueError(f"angle_range min must be <= max, got {self.angle_range}")


class RotationOperator(ModalityOperator):
    """Operator for rotating images with bilinear interpolation.

    This operator applies rotation transformations to image data using bilinear
    interpolation. It supports both deterministic (fixed angle) and stochastic
    (random angle sampling) modes.

    Features:
        - Bilinear interpolation for smooth rotation
        - Configurable angle ranges
        - Fill value for empty areas
        - Support for 2D and 3D images
        - JAX-compatible (jit, vmap, grad)

    Examples:
        Deterministic rotation (fixed angle):

        ```python
        config = RotationOperatorConfig(
            field_key="image",
            angle_range=(15.0, 15.0),  # Fixed 15-degree rotation
            fill_value=0.0,
        )
        operator = RotationOperator(config)
        ```

        Stochastic rotation (random angles):

        ```python
        config = RotationOperatorConfig(
            field_key="image",
            angle_range=(-30.0, 30.0),
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(augment=42)
        operator = RotationOperator(config, rngs=rngs)
        ```

        Apply rotation:

        ```python
        data = {"image": jnp.ones((32, 32, 3))}
        result, state, metadata = operator.apply(data, {}, {})
        ```
    """

    def __init__(
        self,
        config: RotationOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the rotation operator.

        Args:
            config: Configuration for the rotation operator.
            rngs: Optional RNGs object (required if config.stochastic=True).

        Raises:
            ValueError: If stochastic=True but rngs is None (raised by base class).
        """
        super().__init__(config, rngs=rngs)

    def generate_random_params(
        self,
        key: jax.Array,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate random rotation angle for stochastic mode.

        Args:
            key: JAX random key.
            data: Input data dict.
            state: State dict.
            metadata: Metadata dict.

        Returns:
            Dictionary with "angle" key containing random angle in degrees.
        """
        min_angle, max_angle = self.config.angle_range

        # Sample random angle in degrees
        angle = jax.random.uniform(
            key,
            minval=min_angle,
            maxval=max_angle,
        )

        return {"angle": angle}

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply rotation to image data.

        Args:
            data: Input data dict containing image to rotate.
            state: State dict (passed through unchanged).
            metadata: Metadata dict (passed through unchanged).
            random_params: Optional random parameters (contains "angle" if stochastic).
            stats: Optional statistics dict (unused).

        Returns:
            Tuple of (transformed_data, state, metadata).
        """
        # Extract image from data
        value = self._extract_field(data, self.config.field_key)

        # If field is missing, return data unchanged
        if value is None:
            return data, state, metadata

        # Determine rotation angle
        if self.config.stochastic and random_params is not None:
            angle_deg = random_params["angle"]
        else:
            # Deterministic: use midpoint of angle_range
            min_angle, max_angle = self.config.angle_range
            angle_deg = (min_angle + max_angle) / 2.0

        # Convert angle to radians
        angle_rad = angle_deg * jnp.pi / 180.0

        # Apply rotation transformation
        rotated_value = functional.rotate(value, angle_rad, fill_value=self.config.fill_value)

        # Apply clip range
        rotated_value = self._apply_clip_range(rotated_value)

        # Remap field in data
        result = self._remap_field(data, rotated_value)

        return result, state, metadata
