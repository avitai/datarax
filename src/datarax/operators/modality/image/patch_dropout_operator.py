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

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.core.operator import require_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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

    """

    num_patches: int = field(default=4, kw_only=True)
    patch_size: tuple[int, int] = field(default=(8, 8), kw_only=True)
    drop_value: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate num_patches
        if not isinstance(self.num_patches, int):
            raise TypeError(f"num_patches must be an integer, got {type(self.num_patches)}")
        if self.num_patches < 0:
            raise ValueError(f"num_patches must be non-negative, got {self.num_patches}")

        # Validate patch_size
        if not isinstance(self.patch_size, tuple):
            raise TypeError(f"patch_size must be tuple[int, int], got {type(self.patch_size)}")
        if len(self.patch_size) != 2:
            raise ValueError(f"patch_size must be a tuple of length 2, got {self.patch_size}")
        patch_h, patch_w = self.patch_size
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_size dimensions must be positive, got {self.patch_size}")

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
    2. **Stochastic**: Per-record patch positions drawn from the record's own key
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
        # Call the operator on a batch: it draws patch positions per record
        result_batch = operator(batch)
        ```

    """

    def __init__(
        self,
        config: PatchDropoutOperatorConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the patch dropout operator.

        Args:
            config: Configuration for patch dropout operation
            rngs: RNG streams for stochastic operations
        """
        super().__init__(config, rngs=rngs)
        # Type narrowing for better IDE support
        self.config: PatchDropoutOperatorConfig = config

    @staticmethod
    def _normalize_to_3d(value: jax.Array) -> tuple[jax.Array, bool]:
        """Ensure the image is 3-D ``(H, W, C)``, adding a channel axis for 2-D input.

        Args:
            value: Image array, either ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Tuple of (3-D image, ``is_2d`` flag indicating a channel axis was added).

        Raises:
            ValueError: If the image is neither 2-D nor 3-D.
        """
        if value.ndim == 2:
            return value[:, :, jnp.newaxis], True
        if value.ndim == 3:
            return value, False
        raise ValueError(f"Expected 2D or 3D image, got shape {value.shape}")

    def _resolve_patch_positions(
        self,
        h: int,
        w: int,
        patch_h: int,
        patch_w: int,
        key: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(y_positions, x_positions)`` for the patches to drop.

        Positions come from this record's key when stochastic, and from a fixed key
        otherwise, which is what makes the deterministic pattern reproducible. One draw
        serves both: only where the key comes from differs.

        Args:
            h: Image height.
            w: Image width.
            patch_h: Patch height.
            patch_w: Patch width.
            key: This record's PRNG key, required in stochastic mode.

        Returns:
            Tuple of ``(y_positions, x_positions)`` arrays of shape ``(num_patches,)``.
        """
        rng_key = require_key(key, self) if self.config.stochastic else jax.random.key(0)
        rng_x, rng_y = jax.random.split(rng_key)
        x_positions = jax.random.randint(
            rng_x, shape=(self.config.num_patches,), minval=0, maxval=(w - patch_w) + 1
        )
        y_positions = jax.random.randint(
            rng_y, shape=(self.config.num_patches,), minval=0, maxval=(h - patch_h) + 1
        )
        return y_positions, x_positions

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: dict[str, Any],
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], dict[str, Any]]:
        """Apply patch dropout transformation to a single element.

        This operates on single elements (e.g., one image of shape [H, W, C]).

        Args:
            data: Input data dictionary. Must contain field specified by config.field_key
            state: Operator state (unused for patch dropout, passed through)
            metadata: Metadata dictionary (passed through unchanged)
            key: This record's PRNG key, required in stochastic mode
            stats: Optional statistics dictionary (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
                - transformed_data: Data dict with patches dropped from target field
                - state: Unchanged state dict
                - metadata: Unchanged metadata dict
        """
        del stats
        # Extract the field to transform using base class helper
        value = self._extract_field(data, self.config.field_key)

        # Short-circuit if num_patches is zero
        if self.config.num_patches == 0:
            return data, state, metadata

        value, is_2d = self._normalize_to_3d(value)

        h, w, c = value.shape
        patch_h, patch_w = self.config.patch_size

        # Check if patches can fit
        if h < patch_h or w < patch_w:
            # Return unchanged if patches don't fit
            return data, state, metadata

        y_positions, x_positions = self._resolve_patch_positions(h, w, patch_h, patch_w, key)

        # Apply patches using JAX-compatible loop
        def apply_single_patch(i: int, img: jax.Array) -> jax.Array:
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
