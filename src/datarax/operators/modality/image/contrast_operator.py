"""ContrastOperator - Operator for image contrast adjustment.

This operator extends ModalityOperator to provide contrast-only transformations.

Key Features:

- Single-purpose: Only contrast adjustment
- Simplified config with contrast_range parameter
- Uses functional API for implementation

Examples:
    Basic usage:

    ```python
    config = ContrastOperatorConfig(
        field_key="image",
        contrast_range=(0.8, 1.2)
    )
    op = ContrastOperator(config, rngs=rngs)
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.operators.modality.image import functional


@dataclass
class ContrastOperatorConfig(ModalityOperatorConfig):
    """Configuration for ContrastOperator.

    Extends ModalityOperatorConfig with contrast-specific parameters.

    Attributes:
        clip_range: Range for clipping output values. Default: (0.0, 1.0) for
                   normalized images. Overrides parent default of None.
        contrast_range: Range for random contrast adjustment in stochastic mode.
                       Format: (min_factor, max_factor). Default: (0.8, 1.2)
        contrast_factor: Fixed contrast factor for deterministic mode.
                        Only used when stochastic=False. Default: 1.0
    """

    # Override parent's clip_range default to (0.0, 1.0) for normalized images
    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    contrast_range: tuple[float, float] = field(default=(0.8, 1.2), kw_only=True)
    contrast_factor: float = field(default=1.0, kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate contrast_range
        if not isinstance(self.contrast_range, tuple) or len(self.contrast_range) != 2:
            raise ValueError(
                f"contrast_range must be a tuple of length 2, got {self.contrast_range}"
            )
        min_contrast, max_contrast = self.contrast_range
        if min_contrast > max_contrast:
            raise ValueError(
                f"contrast_range must be (min, max) with min <= max, got {self.contrast_range}"
            )


class ContrastOperator(ModalityOperator):
    """Image contrast transformation operator.

    Applies contrast adjustment to images using:
        output = (input - mean) * factor + mean

    Supports three modes:

    - Deterministic: Fixed contrast_factor from config
    - Stochastic: Random factor generated per batch item
    - Learnable: Trainable contrast parameters (via subclass)

    Examples:
        Deterministic mode:

        ```python
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_factor=1.2,
            stochastic=False
        )
        operator = ContrastOperator(config, rngs=nnx.Rngs(0))
        ```
    """

    def __init__(self, config: ContrastOperatorConfig, *, rngs: nnx.Rngs):
        """Initialize ContrastOperator with configuration.

        Args:
            config: ContrastOperatorConfig specifying transformation parameters
            rngs: Flax NNX random number generator state
        """
        super().__init__(config, rngs=rngs)
        self.config: ContrastOperatorConfig = config  # Type narrowing for pyright

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply contrast transformation to data.

        Args:
            data: Input data dictionary containing the image field
            state: Operator state
            metadata: Metadata dictionary
            random_params: Optional random parameters for stochastic mode.
            stats: Optional statistics dictionary

        Returns:
            Tuple of (transformed_data, state, metadata)
        """
        # 1. Extract field using base class helper
        image = self._extract_field(data, self.config.field_key)

        # 2. Determine contrast factor
        if self.config.stochastic and random_params is not None:
            # Stochastic mode
            contrast_factor = random_params.get("contrast", 1.0)
        else:
            # Deterministic mode
            contrast_factor = self.config.contrast_factor

        # 3. Apply contrast adjustment via functional API
        transformed = functional.adjust_contrast(image, contrast_factor)

        # 4. Apply clipping using base class helper
        transformed = self._apply_clip_range(transformed)

        # 5. Remap field using base class helper
        result = self._remap_field(data, transformed)

        return result, state, metadata

    def generate_random_params(
        self, rng: jax.Array, data_shapes: dict[str, tuple[int, ...]]
    ) -> dict[str, Any]:
        """Generate random parameters for stochastic mode.

        Args:
            rng: JAX random number generator key
            data_shapes: Dictionary mapping field keys to their shapes.

        Returns:
            Dictionary containing 'contrast' array of shape (batch_size,)
        """
        # Get batch size from data shapes
        image_shape = data_shapes[self.config.field_key]
        batch_size = image_shape[0]

        # Generate random contrast factors
        min_contrast, max_contrast = self.config.contrast_range
        contrast = jax.random.uniform(
            rng, shape=(batch_size,), minval=min_contrast, maxval=max_contrast
        )

        return {"contrast": contrast}
