"""BatchMixOperator - MixUp and CutMix batch augmentation.

This module provides BatchMixOperator, which performs batch-level sample mixing
that cannot be decomposed into element-level operations.

Key Difference from Other Operators:
- Standard operators use vmap to process elements independently
- BatchMixOperator overrides apply_batch() to access full batch
- Mixing requires cross-element access (sample A mixed with sample B)

Supported Modes:
- mixup: Linear interpolation between pairs of samples
- cutmix: Cut and paste rectangular patches between images

Key Features:
- Unified API for both MixUp and CutMix
- Beta distribution for mixing ratio (alpha parameter)
- Optional label mixing (proportional to mixed area)
- Full JAX compatibility (JIT, grad)
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import BatchMixOperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule


class BatchMixOperator(OperatorModule):
    """Unified operator for batch-level MixUp and CutMix augmentation.

    Performs batch-level sample mixing that requires access to multiple
    samples simultaneously. This operator overrides apply_batch() to
    work at the batch level instead of using vmap.

    MixUp Mode:
        Creates virtual training examples by linear interpolation:
        x_mixed = λ * x_a + (1 - λ) * x_b
        where λ ~ Beta(α, α)

    CutMix Mode:
        Cuts rectangular patches and pastes between images:
        x_mixed = mask * x_a + (1 - mask) * x_b
        Labels are mixed proportionally to the cut area.

    Examples:
        # MixUp augmentation
        config = BatchMixOperatorConfig(mode="mixup", alpha=0.4)
        op = BatchMixOperator(config, rngs=rngs)
        mixed_batch = op(batch)

        # CutMix augmentation
        config = BatchMixOperatorConfig(mode="cutmix", alpha=1.0)
        op = BatchMixOperator(config, rngs=rngs)
        mixed_batch = op(batch)
    """

    def __init__(
        self,
        config: BatchMixOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize BatchMixOperator.

        Args:
            config: Operator configuration (mode, alpha, field names)
            rngs: Random number generators (required - always stochastic)
            name: Optional name for the operator
        """
        super().__init__(config, rngs=rngs, name=name)

        # Type narrowing for pyright
        self.config: BatchMixOperatorConfig = config

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> jax.Array:
        """Generate random parameters - not used for batch-level ops.

        BatchMixOperator overrides apply_batch() completely, so this method
        is not called. Implemented to satisfy the interface.

        Args:
            rng: JAX random key
            data_shapes: PyTree with shapes

        Returns:
            The input rng unchanged (not used)
        """
        return rng

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply operator to single element - not used for batch-level ops.

        BatchMixOperator overrides apply_batch() completely, so this method
        is not called. Batch mixing cannot be decomposed into element-level
        operations. Implemented to satisfy the interface.

        Args:
            data: Element data PyTree
            state: Element state PyTree
            metadata: Element metadata
            random_params: Unused
            stats: Unused

        Returns:
            Input unchanged (not used in practice)
        """
        return data, state, metadata

    def apply_batch(
        self,
        batch: Batch,
        stats: dict[str, Any] | None = None,
    ) -> Batch:
        """Apply batch-level mixing augmentation.

        This method overrides the base class to work at batch level
        instead of using vmap. Batch mixing requires cross-element
        access that cannot be expressed with vmap.

        Args:
            batch: Input batch to mix
            stats: Optional statistics (unused)

        Returns:
            Mixed batch with same structure
        """
        # Handle edge cases: empty or single-element batch
        if batch.batch_size < 2:
            return batch

        # Get RNG key from configured stream
        assert self.rngs is not None, "BatchMixOperator requires rngs"
        assert self.stream_name is not None, "BatchMixOperator requires stream_name"
        key = self.rngs[self.stream_name]()

        # Dispatch to appropriate mixing method
        if self.config.mode == "mixup":
            return self._apply_mixup(batch, key)
        else:  # cutmix
            return self._apply_cutmix(batch, key)

    def _apply_mixup(self, batch: Batch, key: jax.Array) -> Batch:
        """Apply MixUp augmentation to batch.

        MixUp creates virtual training examples by linear interpolation
        between pairs of samples.

        Args:
            batch: Input batch
            key: JAX random key

        Returns:
            Mixed batch
        """
        # Extract batch data
        batch_data = batch.data.get_value()  # PyTree with batch dim

        # Sample mixing ratio from Beta distribution
        key1, key2 = jax.random.split(key)
        lam = jax.random.beta(key1, self.config.alpha, self.config.alpha)

        # Create random permutation for pairing
        batch_size = batch.batch_size
        perm = jax.random.permutation(key2, jnp.arange(batch_size, dtype=jnp.int32))

        # Mix all data fields using numerically stable formula:
        # arr_perm + lam * (arr - arr_perm) is equivalent to lam * arr + (1-lam) * arr_perm
        # but avoids the issue where lam + (1-lam) != 1.0 in floating point
        def mix_array(arr: jax.Array) -> jax.Array:
            arr_perm = arr[perm]
            return arr_perm + lam * (arr - arr_perm)

        mixed_data = jax.tree.map(mix_array, batch_data)

        # Reconstruct batch with mixed data
        return Batch.from_parts(
            data=mixed_data,
            states=batch.states.get_value(),
            metadata_list=batch._metadata_list,
            batch_metadata=batch._batch_metadata,
            batch_state=batch.batch_state.get_value(),
            validate=False,
        )

    def _apply_cutmix(self, batch: Batch, key: jax.Array) -> Batch:
        """Apply CutMix augmentation to batch.

        CutMix cuts rectangular patches and pastes them between images.
        Labels are mixed proportionally to the cut area.

        Args:
            batch: Input batch
            key: JAX random key

        Returns:
            Mixed batch
        """
        # Extract batch data
        batch_data = batch.data.get_value()
        data_field = self.config.data_field
        label_field = self.config.label_field

        # Check if data field exists
        if data_field not in batch_data:
            return batch  # Return unchanged if no image field

        images = batch_data[data_field]

        # Check image shape: need (B, H, W, C) format
        if len(images.shape) < 4:
            return batch  # Return unchanged for invalid shape

        batch_size, height, width = images.shape[:3]

        # Sample mixing ratio and create random box
        key1, key2, key3, key4 = jax.random.split(key, 4)
        lam = jax.random.beta(key1, self.config.alpha, self.config.alpha)

        # Random permutation for pairing
        perm = jax.random.permutation(key2, batch_size)

        # Calculate cut dimensions
        cut_ratio = jnp.sqrt(1.0 - lam)
        cut_h = height * cut_ratio
        cut_w = width * cut_ratio

        # Random center point for cut box
        cx = jax.random.randint(key3, (), 0, width).astype(jnp.float32)
        cy = jax.random.randint(key4, (), 0, height).astype(jnp.float32)

        # Box coordinates (clipped to image bounds)
        x1 = jnp.clip(cx - cut_w / 2, 0, width)
        x2 = jnp.clip(cx + cut_w / 2, 0, width)
        y1 = jnp.clip(cy - cut_h / 2, 0, height)
        y2 = jnp.clip(cy + cut_h / 2, 0, height)

        # Create binary mask using coordinate grids (JIT-compatible)
        # This avoids dynamic slicing which doesn't work with traced indices
        y_coords = jnp.arange(height, dtype=jnp.float32)
        x_coords = jnp.arange(width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")

        # mask = 1 outside cut region, 0 inside cut region
        inside_box = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)
        mask = jnp.where(inside_box, 0.0, 1.0)
        mask = mask[None, :, :, None]  # Add batch and channel dims

        # Apply CutMix to images
        images_perm = images[perm]
        mixed_images = mask * images + (1 - mask) * images_perm

        # Build result data
        result_data = dict(batch_data)
        result_data[data_field] = mixed_images

        # Mix labels if present using numerically stable formula
        if label_field in batch_data:
            labels = batch_data[label_field]
            labels_perm = labels[perm]

            # Adjusted lambda based on actual cut area
            box_area = (x2 - x1) * (y2 - y1)
            total_area = height * width
            lam_adjusted = 1 - (box_area / total_area)

            # Use stable form: labels_perm + lam * (labels - labels_perm)
            result_data[label_field] = labels_perm + lam_adjusted * (labels - labels_perm)

        # Reconstruct batch with mixed data
        return Batch.from_parts(
            data=result_data,
            states=batch.states.get_value(),
            metadata_list=batch._metadata_list,
            batch_metadata=batch._batch_metadata,
            batch_state=batch.batch_state.get_value(),
            validate=False,
        )
