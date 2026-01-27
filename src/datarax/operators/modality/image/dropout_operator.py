"""DropoutOperator - Operator for image dropout augmentation.

This operator extends ModalityOperator to provide pixel-wise and channel-wise dropout.

Key Features:

- Two dropout modes: 'pixel' (element-wise) and 'channel' (entire channels)
- Stochastic mode with per-sample dropout masks
- Deterministic mode for fixed dropout pattern
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

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig


@dataclass
class DropoutOperatorConfig(ModalityOperatorConfig):
    """Configuration for DropoutOperator.

    Extends ModalityOperatorConfig with dropout-specific parameters.

    Attributes:
        dropout_rate: Probability of dropping pixels/channels (0.0 to 1.0).
                     Used in deterministic mode or as default. Default: 0.1
        mode: Dropout mode. Either "pixel" for pixel-wise dropout or
              "channel" for channel-wise dropout. Default: "pixel"
        clip_range: Range for clipping output values. None means no clipping.
                   Default: None (dropout naturally produces values in valid range)

    Note:
        Use dropout_rate and mode parameters to configure the dropout behavior.
    """

    dropout_rate: float = field(default=0.1, kw_only=True)
    mode: Literal["pixel", "channel"] = field(default="pixel", kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters."""
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

    Supports three modes:
    1. **Deterministic**: Fixed dropout pattern using fixed seed
    2. **Stochastic**: Per-sample random dropout masks from generate_random_params()
    3. **External params**: Accept pre-generated random parameters

    The operator works on single elements (H, W, C images) and is composed into
    batch processing via apply_batch() from the base class.

    Examples:
        Deterministic dropout:

        ```python
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="pixel",
            stochastic=False
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))
        result, state, metadata = operator.apply(data, state, metadata)
        ```

        Stochastic dropout with random masks:

        ```python
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="channel",
            stochastic=True
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))
        # Use apply_batch() for automatic random param generation
        result, state, metadata = operator.apply_batch(batch_data, state, metadata)
        ```

    """

    def __init__(
        self,
        config: DropoutOperatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the dropout operator.

        Args:
            config: Configuration for dropout operation
            rngs: RNG streams for stochastic operations
        """
        super().__init__(config, rngs=rngs)
        # Type narrowing for better IDE support
        self.config: DropoutOperatorConfig = config

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, jax.Array]:
        """Generate random dropout masks for stochastic mode.

        In stochastic mode, this pre-generates the dropout masks for the entire batch.
        This approach avoids RNG state mutations inside vmapped apply().

        Args:
            rng: JAX random key
            data_shapes: Dictionary mapping field keys to their shapes.
                        Used to determine batch size and element shapes.

        Returns:
            Dictionary with:

                - "keep_mask": Boolean array indicating which values to keep
                              Shape: (batch_size, H, W, C) for pixel mode
                              Shape: (batch_size, C) for channel mode (will be broadcast)

        Raises:
            KeyError: If field_key not in data_shapes
        """
        if self.config.field_key not in data_shapes:
            raise KeyError(
                f"Field key '{self.config.field_key}' not found in data_shapes. "
                f"Available keys: {list(data_shapes.keys())}"
            )

        # Get full shape including batch dimension
        full_shape = data_shapes[self.config.field_key]  # e.g., (batch_size, H, W, C)

        # Generate keep masks based on mode
        if self.config.mode == "pixel":
            # Pixel-wise dropout: generate mask for all pixels
            keep_mask = jax.random.bernoulli(rng, 1.0 - self.config.dropout_rate, shape=full_shape)
        elif self.config.mode == "channel":
            # Channel-wise dropout: generate mask per channel, will be broadcast in apply()
            batch_size = full_shape[0]
            if len(full_shape) == 4:  # (batch, H, W, C)
                num_channels = full_shape[3]
                # Generate channel masks: (batch_size, num_channels)
                channel_mask = jax.random.bernoulli(
                    rng, 1.0 - self.config.dropout_rate, shape=(batch_size, num_channels)
                )
                # Expand to (batch, 1, 1, C) for broadcasting in apply()
                keep_mask = channel_mask[:, jnp.newaxis, jnp.newaxis, :]
            else:
                # Fallback to pixel-wise for non-4D tensors
                keep_mask = jax.random.bernoulli(
                    rng, 1.0 - self.config.dropout_rate, shape=full_shape
                )
        else:
            raise ValueError(f"Unknown dropout mode: {self.config.mode}")

        return {"keep_mask": keep_mask}

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, jax.Array] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], dict[str, Any]]:
        """Apply dropout transformation to a single element.

        This operates on single elements (e.g., one image of shape [H, W, C]).
        For batch processing, use apply_batch() which handles random param generation.

        Args:
            data: Input data dictionary. Must contain field specified by config.field_key
            state: Operator state (unused for dropout, passed through)
            metadata: Metadata dictionary (passed through unchanged)
            random_params: Optional random parameters from generate_random_params().
                          If config.stochastic=True and this is provided, uses
                          random_params["keep_mask"] for the dropout mask.
            stats: Optional statistics dictionary (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
                - transformed_data: Data dict with dropout applied to target field
                - state: Unchanged state dict
                - metadata: Unchanged metadata dict

        Note:
            CRITICAL: Always check config.stochastic flag, not whether random_params is None.
            apply_batch() always passes random_params even in deterministic mode.
        """
        # Extract the field to transform using base class helper
        value = self._extract_field(data, self.config.field_key)

        # Short-circuit if dropout rate is zero
        if self.config.dropout_rate == 0.0:
            return data, state, metadata

        # Get or generate dropout mask
        # CRITICAL: Check config.stochastic flag, not if random_params is None
        if self.config.stochastic and random_params is not None:
            # Use pre-generated mask from generate_random_params()
            # The mask is generated for the full batch, we get one slice per element
            keep_mask = random_params.get("keep_mask")
            if keep_mask is None:
                raise ValueError(
                    "Stochastic mode requires 'keep_mask' in random_params. "
                    "This should be generated by generate_random_params()."
                )
            # Apply mask (will be broadcast if needed for channel mode)
            transformed = value * keep_mask
        else:
            # Deterministic mode: generate mask with fixed seed
            # CRITICAL: Never call self.rngs() here - it fails inside vmap!
            # Use fixed seed for reproducible dropout pattern
            rng_key = jax.random.key(0)

            # Apply dropout based on mode
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

        return result, state, metadata
