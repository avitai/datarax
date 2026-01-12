"""BrightnessOperator - Operator for image brightness adjustment.

This operator extends ModalityOperator to provide brightness-only transformations.

Key Features:
- Single-purpose: Only brightness adjustment (no contrast)
- Simplified config with brightness_range parameter
- Cleaner for composition in pipelines

Examples:
    Basic usage:

    ```python
    config = BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.2, 0.2)
    )
    op = BrightnessOperator(config, rngs=rngs)
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.operators.modality.image import functional


@dataclass
class BrightnessOperatorConfig(ModalityOperatorConfig):
    """Configuration for BrightnessOperator.

    Extends ModalityOperatorConfig with brightness-specific parameters.

    Attributes:
        clip_range: Range for clipping output values. Default: (0.0, 1.0) for
                   normalized images. Overrides parent default of None.
        brightness_range: Range for random brightness adjustment in stochastic mode.
                         Format: (min_delta, max_delta). Default: (-0.2, 0.2)
        brightness_delta: Fixed brightness adjustment for deterministic mode.
                         Only used when stochastic=False. Default: 0.0

    Note:
        Use brightness_range=(-max_delta, max_delta) for symmetric adjustments,
        e.g., brightness_range=(-0.2, 0.2) for ±0.2 brightness changes.
    """

    # Override parent's clip_range default to (0.0, 1.0) for normalized images
    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    brightness_range: tuple[float, float] = field(default=(-0.2, 0.2), kw_only=True)
    brightness_delta: float = field(default=0.0, kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate brightness_range
        if not isinstance(self.brightness_range, tuple) or len(self.brightness_range) != 2:
            raise ValueError(
                f"brightness_range must be a tuple of length 2, got {self.brightness_range}"
            )
        min_bright, max_bright = self.brightness_range
        if min_bright > max_bright:
            raise ValueError(
                f"brightness_range must be (min, max) with min <= max, got {self.brightness_range}"
            )


class BrightnessOperator(ModalityOperator):
    """Image brightness transformation operator.

    Applies brightness adjustment to images using additive delta:
        output = input + brightness_delta

    Supports three modes:
    - Deterministic: Fixed brightness_delta from config
    - Stochastic: Random delta generated per batch item
    - Learnable: Trainable brightness parameters (via subclass)

    The operator uses element-level apply() design:
    - apply(): Operates on single element (H,W,C) without batch dimension
    - apply_batch(): Handles batches via vmap

    Examples:
        Deterministic mode:

        ```python
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = operator.apply(data, {}, {})
        ```

        Stochastic mode:

        ```python
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.2, 0.2),  # Matches max_delta=0.2
            stochastic=True,
            stream_name="augment"
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0, augment=1))
        random_params = operator.generate_random_params(rng, data_shapes)
        result, _, _ = operator.apply(data, {}, {}, random_params=random_params)
        ```
    """

    def __init__(self, config: BrightnessOperatorConfig, *, rngs: nnx.Rngs):
        """Initialize BrightnessOperator with configuration.

        Args:
            config: BrightnessOperatorConfig specifying transformation parameters
            rngs: Flax NNX random number generator state

        Note:
            For learnable transformations, create a subclass that:
            1. Adds nnx.Param fields in its __init__
            2. Overrides apply() to use those parameters
        """
        super().__init__(config, rngs=rngs)
        self.config: BrightnessOperatorConfig = config  # Type narrowing for pyright

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply brightness transformation to data.

        This method demonstrates the standard pattern for using base class helpers:
        1. Extract field using _extract_field (handles KeyError gracefully)
        2. Apply transformation
        3. Apply clip_range using _apply_clip_range
        4. Remap field using _remap_field (handles target_key logic)

        Args:
            data: Input data dictionary containing the image field
            state: Operator state (unused for stateless transformations)
            metadata: Metadata dictionary (passed through unchanged)
            random_params: Optional random parameters for stochastic mode.
                          Expected keys: 'brightness'
            stats: Optional statistics dictionary (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
        """
        # 1. Extract field using base class helper (handles validation)
        image = self._extract_field(data, self.config.field_key)

        # 2. Determine brightness adjustment
        # Note: apply_batch() always passes random_params (dummy in deterministic mode)
        # so we check config.stochastic instead of checking if random_params is None
        if self.config.stochastic and random_params is not None:
            # Stochastic mode: use random parameters
            brightness_delta = random_params.get("brightness", 0.0)
        else:
            # Deterministic mode: use config value
            brightness_delta = self.config.brightness_delta

        # 3. Apply brightness adjustment via functional API
        # Note: apply() operates on single elements (no batch dimension)
        image = functional.adjust_brightness_delta(image, brightness_delta)

        # 4. Apply clipping using base class helper
        transformed = self._apply_clip_range(image)

        # 5. Remap field using base class helper (handles target_key logic)
        result = self._remap_field(data, transformed)

        return result, state, metadata

    def generate_random_params(
        self, rng: jax.Array, data_shapes: dict[str, tuple[int, ...]]
    ) -> dict[str, Any]:
        """Generate random parameters for stochastic mode.

        Creates random brightness values within the configured range,
        one value per batch item. These arrays are later distributed
        by apply_batch() via vmap, so each apply() call receives a scalar value.

        Args:
            rng: JAX random number generator key
            data_shapes: Dictionary mapping field keys to their shapes.
                        Used to determine batch size.

        Returns:
            Dictionary containing:
                - 'brightness': Array of shape (batch_size,) with per-element brightness deltas
        """
        # Get batch size from data shapes
        image_shape = data_shapes[self.config.field_key]
        batch_size = image_shape[0]

        # Generate random brightness deltas
        min_bright, max_bright = self.config.brightness_range
        brightness = jax.random.uniform(
            rng, shape=(batch_size,), minval=min_bright, maxval=max_bright
        )

        return {"brightness": brightness}
