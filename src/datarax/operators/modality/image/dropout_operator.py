"""DropoutOperator - Operator for image dropout augmentation.

This operator extends ModalityOperator to provide pixel-wise and channel-wise dropout.

Key Features:

- Two dropout modes: 'pixel' (element-wise) and 'channel' (entire channels)
- Per-record dropout masks drawn from each record's own key; ``eval()`` turns them off
- Full JAX compatibility with JIT compilation

Examples:
    Basic usage:

    ```python
    config = DropoutOperatorConfig(
        field_key="image",
        dropout_rate=0.2,
        mode="pixel"
    )
    op = DropoutOperator(config, rngs=rngs)
    ```
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import require_stochastic
from datarax.core.element_batch import Element
from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.core.operator import require_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DropoutOperatorConfig(ModalityOperatorConfig):
    """Configuration for DropoutOperator.

    Extends ModalityOperatorConfig with dropout-specific parameters.

    Attributes:
        dropout_rate: Probability of dropping pixels/channels (0.0 to 1.0).
                     Default: 0.1
        mode: Dropout mode. Either "pixel" for pixel-wise dropout or
              "channel" for channel-wise dropout. Default: "pixel"
        clip_range: Range for clipping output values. None means no clipping.
                   Default: None (dropout naturally produces values in valid range)

    Note:
        Use dropout_rate and mode parameters to configure the dropout behavior.
    """

    # The operator only draws, so it is stochastic by default (turned off with ``eval()``).
    stochastic: bool = True
    stream_name: str | None = "augment"

    dropout_rate: float = field(default=0.1, kw_only=True)
    mode: Literal["pixel", "channel"] = field(default="pixel", kw_only=True)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        require_stochastic(self)
        super().__post_init__()

        # Validate dropout_rate
        if not isinstance(self.dropout_rate, int | float):
            raise TypeError(f"dropout_rate must be a number, got {type(self.dropout_rate)}")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0.0, 1.0], got {self.dropout_rate}")

        # Validate mode
        if self.mode not in ("pixel", "channel"):
            raise ValueError(f"mode must be 'pixel' or 'channel', got '{self.mode}'")


class DropoutOperator(ModalityOperator):
    """Image dropout transformation operator.

    Applies dropout to images by randomly setting pixels or channels to zero:

        - Pixel mode: Each pixel independently dropped with probability dropout_rate
        - Channel mode: Entire channels dropped with probability dropout_rate

    Draws per-record dropout masks from the record's own key. ``eval()``, or
    ``nnx.view(operator, deterministic=True)``, returns the record unchanged.

    The operator works on single elements (H, W, C images) and is composed into
    batch processing via apply_batch() from the base class.

    Examples:
        Stochastic dropout with random masks:

        ```python
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="channel",
            stochastic=True
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))
        # Call the operator on a batch: it draws a mask per record
        result_batch = operator(batch)
        ```

    """

    def __init__(
        self,
        config: DropoutOperatorConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the dropout operator.

        Args:
            config: Configuration for dropout operation
            rngs: RNG streams for stochastic operations
        """
        super().__init__(config, rngs=rngs)
        # Type narrowing for better IDE support
        self.config: DropoutOperatorConfig = config

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Apply dropout transformation to a single element.

        This operates on single elements (e.g., one image of shape [H, W, C]).

        Args:
            element: The record, without a batch axis.
            key: This record's PRNG key, required in stochastic mode
            stats: Optional statistics dictionary (unused)

        Returns:
            The transformed record.

        Raises:
            ValueError: If ``config.mode`` is not a known dropout mode.
        """
        data = element.data
        del stats
        # Extract the field to transform using base class helper
        value = self._extract_field(data, self.config.field_key)

        # Short-circuit if dropout rate is zero
        if self.config.dropout_rate == 0.0:
            return element

        # This record's key: the mask depends on the record alone.
        rng_key = require_key(key, self)

        if self.config.mode == "pixel":
            # Pixel-wise dropout: each pixel independently dropped
            keep_mask = jax.random.bernoulli(
                rng_key, 1.0 - self.config.dropout_rate, shape=value.shape
            )
            transformed = value * keep_mask

        elif self.config.mode == "channel":
            # Channel-wise dropout: entire channels dropped
            if value.ndim == 3:
                h, w, c = value.shape
                # Generate channel mask
                channel_mask = jax.random.bernoulli(
                    rng_key, 1.0 - self.config.dropout_rate, shape=(c,)
                )
                # Broadcast to full image shape (H, W, C)
                keep_mask = jnp.ones((h, w, 1)) * channel_mask[None, None, :]
                transformed = value * keep_mask
            else:
                # Fallback to pixel-wise for non-3D images
                keep_mask = jax.random.bernoulli(
                    rng_key, 1.0 - self.config.dropout_rate, shape=value.shape
                )
                transformed = value * keep_mask
        else:
            # Should never reach here due to config validation
            raise ValueError(f"Unknown dropout mode: {self.config.mode}")

        # Apply clipping if configured (though typically not needed for dropout)
        if self.config.clip_range is not None:
            transformed = self._apply_clip_range(transformed)

        # Remap the transformed value back into the data dictionary
        result = self._remap_field(data, transformed)

        return element.replace(data=result)
