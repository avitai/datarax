"""PatchDropoutOperator - Operator for patch-based occlusion augmentation.

This operator extends ModalityOperator to provide patch-based dropout (occlusion).

Key Features:

- Drops random rectangular patches from images
- Configurable number of patches and patch size
- Deterministic mode with fixed patch positions
- Stochastic mode with random patch positions per sample
- Full JAX compatibility with JIT compilation

Examples:
    Basic usage:

    ```python
    config = PatchDropoutOperatorConfig(
        field_key="image",
        num_patches=4,
        patch_size=(8, 8),
        drop_value=0.0
    )
    op = PatchDropoutOperator(config, rngs=rngs)
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig


@dataclass
class PatchDropoutOperatorConfig(ModalityOperatorConfig):
    """Configuration for PatchDropoutOperator.

    Extends ModalityOperatorConfig with patch dropout-specific parameters.

    Attributes:
        num_patches: Number of rectangular patches to drop from each image.
                    Default: 4
        patch_size: Size of each patch as (height, width) tuple.
                   Default: (8, 8)
        drop_value: Value to fill dropped patches with. Typically 0.0 for black
                   or the mean image value. Default: 0.0
        clip_range: Range for clipping output values. None means no clipping.
                   Default: None (patch dropout preserves valid ranges)

    Note:
        The patch_size can be specified as a single int for square patches,
        which will be converted to (size, size) tuple automatically.
    """

    num_patches: int = field(default=4, kw_only=True)
    patch_size: tuple[int, int] = field(default=(8, 8), kw_only=True)
    drop_value: float = field(default=0.0, kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate num_patches
        if not isinstance(self.num_patches, int):
            raise TypeError(f"num_patches must be an integer, got {type(self.num_patches)}")
        if self.num_patches < 0:
            raise ValueError(f"num_patches must be non-negative, got {self.num_patches}")

        # Validate and normalize patch_size
        if isinstance(self.patch_size, int):
            # Convert int to tuple for backward compatibility
            if self.patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {self.patch_size}")
            object.__setattr__(self, "patch_size", (self.patch_size, self.patch_size))
        elif isinstance(self.patch_size, tuple):
            if len(self.patch_size) != 2:
                raise ValueError(f"patch_size must be a tuple of length 2, got {self.patch_size}")
            patch_h, patch_w = self.patch_size
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(f"patch_size dimensions must be positive, got {self.patch_size}")
        else:
            raise TypeError(
                f"patch_size must be int or tuple[int, int], got {type(self.patch_size)}"
            )

        # Validate drop_value
        if not isinstance(self.drop_value, int | float):
            raise TypeError(f"drop_value must be a number, got {type(self.drop_value)}")


class PatchDropoutOperator(ModalityOperator):
    """Image patch dropout transformation operator.

    Applies patch dropout by randomly dropping rectangular regions from images:

        - Selects num_patches random positions
        - Replaces each patch with drop_value
        - Useful for occlusion robustness training

    Supports three modes:
    1. **Deterministic**: Fixed patch positions using fixed seed
    2. **Stochastic**: Per-sample random patch positions from generate_random_params()
    3. **External params**: Accept pre-generated random parameters

    The operator works on single elements (H, W, C images) and is composed into
    batch processing via apply_batch() from the base class.

    Examples:
        Deterministic patch dropout:

        ```python
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(16, 16),
            drop_value=0.0,
            stochastic=False
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))
        result, state, metadata = operator.apply(data, state, metadata)
        ```

        Stochastic patch dropout with random positions:

        ```python
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=8,
            patch_size=(8, 8),
            drop_value=0.5,
            stochastic=True
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))
        # Use apply_batch() for automatic random param generation
        result, state, metadata = operator.apply_batch(batch_data, state, metadata)
        ```

    """

    def __init__(
        self,
        config: PatchDropoutOperatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the patch dropout operator.

        Args:
            config: Configuration for patch dropout operation
            rngs: RNG streams for stochastic operations
        """
        super().__init__(config, rngs=rngs)
        # Type narrowing for better IDE support
        self.config: PatchDropoutOperatorConfig = config

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, jax.Array]:
        """Generate random patch positions for stochastic mode.

        In stochastic mode, this pre-generates random patch positions for the entire batch.
        This approach avoids RNG state mutations inside vmapped apply().

        Args:
            rng: JAX random key
            data_shapes: Dictionary mapping field keys to their shapes.
                        Used to determine batch size and image dimensions.

        Returns:
            Dictionary with:

                - "patch_positions": Array of patch top-left positions
                                   Shape: (batch_size, num_patches, 2) where last dim is (y, x)

        Raises:
            KeyError: If field_key not in data_shapes
        """
        if self.config.field_key not in data_shapes:
            raise KeyError(
                f"Field key '{self.config.field_key}' not found in data_shapes. "
                f"Available keys: {list(data_shapes.keys())}"
            )

        # Get shape: (batch_size, H, W, C) or (batch_size, H, W)
        full_shape = data_shapes[self.config.field_key]
        batch_size = full_shape[0]
        image_height = full_shape[1]
        image_width = full_shape[2]

        patch_h, patch_w = self.config.patch_size

        # Check if patches can fit (will be checked again in apply for safety)
        if image_height < patch_h or image_width < patch_w:
            # Return zero positions - apply() will skip processing
            return {
                "patch_positions": jnp.zeros(
                    (batch_size, self.config.num_patches, 2), dtype=jnp.int32
                )
            }

        # Generate random patch positions for entire batch
        # Split RNG for x and y coordinates
        rng_x, rng_y = jax.random.split(rng)

        # Maximum valid positions (top-left corner of patch)
        max_x = image_width - patch_w
        max_y = image_height - patch_h

        # Generate positions: (batch_size, num_patches)
        x_positions = jax.random.randint(
            rng_x, shape=(batch_size, self.config.num_patches), minval=0, maxval=max_x + 1
        )
        y_positions = jax.random.randint(
            rng_y, shape=(batch_size, self.config.num_patches), minval=0, maxval=max_y + 1
        )

        # Stack into (batch_size, num_patches, 2) where last dim is (y, x)
        patch_positions = jnp.stack([y_positions, x_positions], axis=-1)

        return {"patch_positions": patch_positions}

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, jax.Array] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], dict[str, Any]]:
        """Apply patch dropout transformation to a single element.

        This operates on single elements (e.g., one image of shape [H, W, C]).
        For batch processing, use apply_batch() which handles random param generation.

        Args:
            data: Input data dictionary. Must contain field specified by config.field_key
            state: Operator state (unused for patch dropout, passed through)
            metadata: Metadata dictionary (passed through unchanged)
            random_params: Optional random parameters from generate_random_params().
                          If config.stochastic=True and this is provided, uses
                          random_params["patch_positions"] for patch locations.
            stats: Optional statistics dictionary (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
                - transformed_data: Data dict with patches dropped from target field
                - state: Unchanged state dict
                - metadata: Unchanged metadata dict

        Note:
            CRITICAL: Always check config.stochastic flag, not whether random_params is None.
            apply_batch() always passes random_params even in deterministic mode.
        """
        # Extract the field to transform using base class helper
        value = self._extract_field(data, self.config.field_key)

        # Short-circuit if num_patches is zero
        if self.config.num_patches == 0:
            return data, state, metadata

        # Handle 2D vs 3D images
        if value.ndim == 2:
            # Add channel dimension for processing
            value = value[:, :, jnp.newaxis]
            is_2d = True
        elif value.ndim == 3:
            is_2d = False
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {value.shape}")

        h, w, c = value.shape
        patch_h, patch_w = self.config.patch_size

        # Check if patches can fit
        if h < patch_h or w < patch_w:
            # Return unchanged if patches don't fit
            return data, state, metadata

        # Get or generate patch positions
        # CRITICAL: Check config.stochastic flag, not if random_params is None
        if self.config.stochastic and random_params is not None:
            # Use pre-generated positions from generate_random_params()
            # Shape: (num_patches, 2) where last dim is (y, x)
            patch_positions = random_params.get("patch_positions")
            if patch_positions is None:
                raise ValueError(
                    "Stochastic mode requires 'patch_positions' in random_params. "
                    "This should be generated by generate_random_params()."
                )
            # Extract y and x positions
            y_positions = patch_positions[:, 0]
            x_positions = patch_positions[:, 1]
        else:
            # Deterministic mode: generate positions with fixed seed
            # CRITICAL: Never call self.rngs() here - it fails inside vmap!
            rng_key = jax.random.key(0)
            rng_x, rng_y = jax.random.split(rng_key)

            max_x = w - patch_w
            max_y = h - patch_h

            x_positions = jax.random.randint(
                rng_x, shape=(self.config.num_patches,), minval=0, maxval=max_x + 1
            )
            y_positions = jax.random.randint(
                rng_y, shape=(self.config.num_patches,), minval=0, maxval=max_y + 1
            )

        # Apply patches using JAX-compatible loop
        def apply_single_patch(i, img):
            """Apply a single patch to the image."""
            x = x_positions[i]
            y = y_positions[i]
            # Create patch filled with drop_value
            patch_shape = (patch_h, patch_w, c)
            patch = jnp.full(patch_shape, self.config.drop_value)
            # Use dynamic_update_slice to insert patch
            return jax.lax.dynamic_update_slice(img, patch, (y, x, 0))

        # Apply all patches sequentially
        transformed = jax.lax.fori_loop(0, self.config.num_patches, apply_single_patch, value)

        # Remove channel dimension if originally 2D
        if is_2d:
            transformed = transformed[:, :, 0]

        # Apply clipping if configured
        if self.config.clip_range is not None:
            transformed = self._apply_clip_range(transformed)

        # Remap the transformed value back into the data dictionary
        result = self._remap_field(data, transformed)

        return result, state, metadata
