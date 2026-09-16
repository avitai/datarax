"""RotationOperator - Image rotation augmentation for modality operators.

This module provides rotation augmentation with:

- Configurable angle ranges (deterministic and stochastic)
- Bilinear interpolation for smooth rotation
- Fill value for empty areas after rotation
- Support for 2D (grayscale) and 3D (RGB) images
"""

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.core.operator import require_key
from datarax.operators.modality.image import functional
from datarax.operators.modality.image._validation import resolve_mode_parameters


logger = logging.getLogger(__name__)

DEFAULT_ANGLE_RANGE = (-15.0, 15.0)
DEFAULT_ANGLE = 0.0


@dataclass(frozen=True)
class RotationOperatorConfig(ModalityOperatorConfig):
    """Configuration for rotation augmentation operator.

    Attributes:
        field_key: Field to rotate (e.g., "image" or "data.image" for nested).
        angle_range: ``(min_angle, max_angle)`` in degrees a stochastic operator draws each
                     record's angle from. Default: (-15.0, 15.0). Refused when
                     stochastic=False. Positive angles rotate counter-clockwise.
        angle: The angle in degrees a deterministic operator applies. Default: 0.0.
               Refused when stochastic=True.
        fill_value: Value to fill empty areas after rotation (default: 0.0).
        interpolation: Interpolation mode (currently only "bilinear" supported).
        clip_range: Range to clip output values (default: (0.0, 1.0)).
        stochastic: Whether to use random angle sampling.
        stream_name: RNG stream name (required if stochastic=True).
    """

    # Each mode uses one of these; __post_init__ fills in its default and refuses the other.
    angle_range: tuple[float, float] | None = field(default=None, kw_only=True)
    angle: float | None = field(default=None, kw_only=True)
    fill_value: float = 0.0
    interpolation: str = "bilinear"

    # Clip range (inherited from ModalityOperatorConfig, default (0.0, 1.0))
    clip_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        """Validate configuration parameters and resolve the parameter the mode uses."""
        super().__post_init__()
        resolve_mode_parameters(
            self,
            range_field="angle_range",
            fixed_field="angle",
            default_range=DEFAULT_ANGLE_RANGE,
            default_fixed=DEFAULT_ANGLE,
        )


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
            angle=15.0,  # Fixed 15-degree rotation
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

    def __init__(  # noqa: DOC502
        self,
        config: RotationOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize the rotation operator.

        Args:
            config: Configuration for the rotation operator.
            rngs: Optional RNGs object (required if config.stochastic=True).

        Raises:
            ValueError: If stochastic=True but rngs is None (raised by base class).
        """
        super().__init__(config, rngs=rngs)
        # Type narrowing for pyright — config is RotationOperatorConfig
        self.config: RotationOperatorConfig = config

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply rotation to image data.

        Args:
            data: Input data dict containing image to rotate.
            state: State dict (passed through unchanged).
            metadata: Metadata dict (passed through unchanged).
            key: This record's PRNG key, required in stochastic mode.
            stats: Optional statistics dict (unused).

        Returns:
            Tuple of (transformed_data, state, metadata).
        """
        del stats
        # Extract image from data
        value = self._extract_field(data, self.config.field_key)

        # If field is missing, return data unchanged
        if value is None:
            return data, state, metadata

        # Determine rotation angle
        if self.config.stochastic:
            # This record's own angle, drawn from its own key
            min_angle, max_angle = cast(tuple[float, float], self.config.angle_range)
            angle_deg = jax.random.uniform(
                require_key(key, self), shape=(), minval=min_angle, maxval=max_angle
            )
        else:
            angle_deg = cast(float, self.config.angle)

        # Convert angle to radians
        angle_rad = angle_deg * jnp.pi / 180.0

        # Apply rotation transformation
        rotated_value = functional.rotate(value, angle_rad, fill_value=self.config.fill_value)

        # Apply clip range
        rotated_value = self._apply_clip_range(rotated_value)

        # Remap field in data
        result = self._remap_field(data, rotated_value)

        return result, state, metadata
