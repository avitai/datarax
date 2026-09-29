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

import logging
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import BatchMixOperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule
from datarax.core.prng import per_record_keys


logger = logging.getLogger(__name__)


class BatchMixOperator(OperatorModule):
    """Unified operator for batch-level MixUp and CutMix augmentation.

    Performs batch-level sample mixing that requires access to multiple
    samples simultaneously. This operator overrides apply_batch() to
    work at the batch level instead of using vmap.

    Modes:

    MixUp Mode:
        Creates virtual training examples by linear interpolation:
        x_mixed = λ * x_a + (1 - λ) * x_b
        where λ ~ Beta(α, α)

    CutMix Mode:
        Cuts rectangular patches and pastes between images:
        x_mixed = mask * x_a + (1 - mask) * x_b
        Labels are mixed proportionally to the cut area.

    Examples:
        ```python
        config = BatchMixOperatorConfig(mode="mixup", alpha=0.4)  # MixUp augmentation
        op = BatchMixOperator(config, rngs=rngs)
        mixed_batch = op(batch)
        config = BatchMixOperatorConfig(mode="cutmix", alpha=1.0)  # CutMix augmentation
        op = BatchMixOperator(config, rngs=rngs)
        mixed_batch = op(batch)
        ```
    """

    def __init__(
        self,
        config: BatchMixOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize BatchMixOperator.

        Args:
            config: Operator configuration (mode, alpha, field names)
            rngs: Random number generators (required - always stochastic)
            name: Optional name for the operator
        """
        super().__init__(config, rngs=rngs, name=name)

        # Type narrowing for pyright
        self.config: BatchMixOperatorConfig = config

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> NoReturn:
        """Refuse the per-record call: batch mixing has no element-level form.

        The operator mixes each record with another record of the same batch, which a
        single record cannot do, so it overrides ``apply_batch`` instead. This method exists
        to satisfy the element-level interface.

        Args:
            data: Element data PyTree
            state: Element state PyTree
            metadata: Element metadata
            key: Unused
            stats: Unused

        Raises:
            NotImplementedError: Always; mixing needs the whole batch.
        """
        del data, state, metadata, key, stats
        raise NotImplementedError(
            "BatchMixOperator mixes each record with another record of the same batch, "
            "so it has no per-record form; call apply_batch(batch) or the operator itself."
        )

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
        del stats
        # Deterministic mode mixes nothing; nor can a batch of fewer than two records.
        if self.deterministic or batch.batch_size < 2:
            return batch

        # The whole batch mixes with one key: its first record's, so a resumed run mixes the
        # same batch the same way on any number of hosts.
        key = per_record_keys(
            self._base_key[...],
            jnp.asarray(batch.indices)[:1],
            jnp.asarray(batch.epochs)[:1],
            jnp.asarray(batch.draws)[:1],
        )[0]
        if self.config.mode == "mixup":
            return batch.replace(data=self._mixup(batch.data, batch.batch_size, key))
        return batch.replace(data=self._cutmix(batch.data, key))

    def _mixup(self, data: PyTree, batch_size: int, key: jax.Array) -> PyTree:
        """Mix every field with a batch axis with a permutation of the batch (MixUp)."""
        key1, key2 = jax.random.split(key)
        lam = jax.random.beta(key1, self.config.alpha, self.config.alpha)
        perm = jax.random.permutation(key2, jnp.arange(batch_size, dtype=jnp.int32))

        def mix_array(arr: jax.Array) -> jax.Array:
            if not hasattr(arr, "shape") or len(arr.shape) == 0 or arr.shape[0] != batch_size:
                return arr
            arr_perm = arr[perm]
            # arr_perm + lam * (arr - arr_perm) == lam * arr + (1 - lam) * arr_perm, without
            # lam + (1 - lam) != 1 in floating point.
            return arr_perm + lam * (arr - arr_perm)

        return jax.tree.map(mix_array, data)

    def _cutmix(self, data: PyTree, key: jax.Array) -> PyTree:
        """Paste a box of a permuted image into each image and mix labels by area (CutMix)."""
        if not isinstance(data, dict):
            return data

        data_field = self.config.data_field
        label_field = self.config.label_field
        if data_field not in data:
            return data

        images = data[data_field]
        if len(images.shape) < 4:
            return data

        batch_size, height, width = images.shape[:3]

        key1, key2, key3, key4 = jax.random.split(key, 4)
        lam = jax.random.beta(key1, self.config.alpha, self.config.alpha)
        perm = jax.random.permutation(key2, batch_size)

        cut_ratio = jnp.sqrt(1.0 - lam)
        cut_h = height * cut_ratio
        cut_w = width * cut_ratio

        cx = jax.random.randint(key3, (), 0, width).astype(jnp.float32)
        cy = jax.random.randint(key4, (), 0, height).astype(jnp.float32)

        x1 = jnp.clip(cx - cut_w / 2, 0, width)
        x2 = jnp.clip(cx + cut_w / 2, 0, width)
        y1 = jnp.clip(cy - cut_h / 2, 0, height)
        y2 = jnp.clip(cy + cut_h / 2, 0, height)

        # A mask from coordinate grids, not a dynamic slice, so the box may be traced.
        y_coords = jnp.arange(height, dtype=jnp.float32)
        x_coords = jnp.arange(width, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")
        inside_box = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)
        mask = jnp.where(inside_box, 0.0, 1.0)[None, :, :, None]

        result = dict(data)
        images_perm = images[perm]
        result[data_field] = mask * images + (1 - mask) * images_perm

        if label_field in data:
            labels = data[label_field]
            labels_perm = labels[perm]
            box_area = (x2 - x1) * (y2 - y1)
            total_area = height * width
            lam_adjusted = 1 - (box_area / total_area)
            result[label_field] = labels_perm + lam_adjusted * (labels - labels_perm)

        return result
