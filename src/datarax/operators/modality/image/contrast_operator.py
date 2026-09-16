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
        contrast_range=(0.8, 1.2),
        stochastic=True,
        stream_name="augment",
    )
    op = ContrastOperator(config, rngs=rngs)
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

DEFAULT_CONTRAST_RANGE = (0.8, 1.2)
DEFAULT_CONTRAST_FACTOR = 1.0


@dataclass(frozen=True)
class ContrastOperatorConfig(ModalityOperatorConfig):
    """Configuration for ContrastOperator.

    Extends ModalityOperatorConfig with contrast-specific parameters.

    Attributes:
        clip_range: Range for clipping output values. Default: (0.0, 1.0) for
                   normalized images. Overrides parent default of None.
        contrast_range: ``(min_factor, max_factor)`` a stochastic operator draws each
                       record's factor from. Default: (0.8, 1.2). Refused when
                       stochastic=False.
        contrast_factor: The factor a deterministic operator applies. Default: 1.0.
                        Refused when stochastic=True.
    """

    # Override parent's clip_range default to (0.0, 1.0) for normalized images
    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    # Each mode uses one of these; __post_init__ fills in its default and refuses the other.
    contrast_range: tuple[float, float] | None = field(default=None, kw_only=True)
    contrast_factor: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate configuration parameters and resolve the parameter the mode uses."""
        super().__post_init__()
        resolve_mode_parameters(
            self,
            range_field="contrast_range",
            fixed_field="contrast_factor",
            default_range=DEFAULT_CONTRAST_RANGE,
            default_fixed=DEFAULT_CONTRAST_FACTOR,
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

    def __init__(self, config: ContrastOperatorConfig, *, rngs: nnx.Rngs) -> None:
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
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply contrast transformation to data.

        Args:
            data: Input data dictionary containing the image field
            state: Operator state
            metadata: Metadata dictionary
            key: This record's PRNG key, required in stochastic mode
            stats: Optional statistics dictionary

        Returns:
            Tuple of (transformed_data, state, metadata)
        """
        del stats
        # 1. Extract field using base class helper
        image = self._extract_field(data, self.config.field_key)

        # 2. Determine contrast factor
        if self.config.stochastic:
            # This record's own factor, drawn from its own key
            min_contrast, max_contrast = cast(tuple[float, float], self.config.contrast_range)
            contrast_factor = jax.random.uniform(
                require_key(key, self), shape=(), minval=min_contrast, maxval=max_contrast
            )
        else:
            # Deterministic mode
            contrast_factor = cast(float, self.config.contrast_factor)

        # 3. Apply contrast adjustment via functional API
        transformed = functional.adjust_contrast(image, contrast_factor)

        # 4. Apply clipping using base class helper
        transformed = self._apply_clip_range(transformed)

        # 5. Remap field using base class helper
        result = self._remap_field(data, transformed)

        return result, state, metadata
