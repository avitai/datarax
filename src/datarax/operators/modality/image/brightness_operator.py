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
        brightness_range=(-0.2, 0.2),
        stochastic=True,
        stream_name="augment",
    )
    op = BrightnessOperator(config, rngs=rngs)
    ```
"""

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import jax
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.core.operator import require_key
from datarax.operators.modality.image import functional
from datarax.operators.modality.image._validation import resolve_mode_parameters


logger = logging.getLogger(__name__)

DEFAULT_BRIGHTNESS_RANGE = (-0.2, 0.2)
DEFAULT_BRIGHTNESS_DELTA = 0.0


@dataclass(frozen=True)
class BrightnessOperatorConfig(ModalityOperatorConfig):
    """Configuration for BrightnessOperator.

    Extends ModalityOperatorConfig with brightness-specific parameters.

    Attributes:
        clip_range: Range for clipping output values. Default: (0.0, 1.0) for
                   normalized images. Overrides parent default of None.
        brightness_range: ``(min_delta, max_delta)`` a stochastic operator draws each
                         record's delta from. Default: (-0.2, 0.2). Refused when
                         stochastic=False.
        brightness_delta: The delta a deterministic operator adds. Default: 0.0.
                         Refused when stochastic=True.

    Note:
        Use brightness_range=(-max_delta, max_delta) for symmetric adjustments,
        e.g., brightness_range=(-0.2, 0.2) for ±0.2 brightness changes.
    """

    # Override parent's clip_range default to (0.0, 1.0) for normalized images
    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    # Each mode uses one of these; __post_init__ fills in its default and refuses the other.
    brightness_range: tuple[float, float] | None = field(default=None, kw_only=True)
    brightness_delta: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate configuration parameters and resolve the parameter the mode uses."""
        super().__post_init__()
        resolve_mode_parameters(
            self,
            range_field="brightness_range",
            fixed_field="brightness_delta",
            default_range=DEFAULT_BRIGHTNESS_RANGE,
            default_fixed=DEFAULT_BRIGHTNESS_DELTA,
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
        result, _, _ = operator.apply(data, {}, {}, key=jax.random.key(0))
        ```
    """

    def __init__(self, config: BrightnessOperatorConfig, *, rngs: nnx.Rngs) -> None:
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
        key: jax.Array | None = None,
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
            key: This record's PRNG key, required in stochastic mode
            stats: Optional statistics dictionary (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
        """
        del stats
        # 1. Extract field using base class helper (handles validation)
        image = self._extract_field(data, self.config.field_key)

        # 2. Determine brightness adjustment
        if self.config.stochastic:
            # This record's own delta, drawn from its own key
            min_bright, max_bright = cast(tuple[float, float], self.config.brightness_range)
            brightness_delta = jax.random.uniform(
                require_key(key, self), shape=(), minval=min_bright, maxval=max_bright
            )
        else:
            # Deterministic mode: use config value
            brightness_delta = cast(float, self.config.brightness_delta)

        # 3. Apply brightness adjustment via functional API
        # Note: apply() operates on single elements (no batch dimension)
        image = functional.adjust_brightness_delta(image, brightness_delta)

        # 4. Apply clipping using base class helper
        transformed = self._apply_clip_range(image)

        # 5. Remap field using base class helper (handles target_key logic)
        result = self._remap_field(data, transformed)

        return result, state, metadata
