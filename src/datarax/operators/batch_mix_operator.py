"""BatchMixOperator - MixUp and CutMix batch augmentation.

Mixing pairs each record with another record of the same batch, so the operator has no per-record
form: it overrides ``apply_batch`` alone and draws from its first record's key.

Supported modes:

- mixup: each image becomes ``lam * x + (1 - lam) * x[partner]`` (Zhang et al. 2018)
- cutmix: a box of the partner's image is pasted into each image (Yun et al. 2019)

Labels are left untouched. The operator writes what it did for the loss to read: each record's
partner row in ``states[MIX_PARTNER]`` and the fraction of each image kept in
``batch_state[MIX_LAMBDA]``, so ``lam * loss(y) + (1 - lam) * loss(y[partner])`` is the mixed
loss for integer and one-hot labels alike.
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import BatchMixOperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule
from datarax.core.state_keys import MIX_LAMBDA, MIX_PARTNER


logger = logging.getLogger(__name__)


class BatchMixOperator(OperatorModule):
    """Unified operator for batch-level MixUp and CutMix augmentation.

    Performs batch-level sample mixing that requires access to multiple
    samples simultaneously. It overrides apply_batch() alone and has no
    per-record form.

    Modes:

    MixUp Mode:
        Creates virtual training examples by linear interpolation:
        x_mixed = λ * x_a + (1 - λ) * x_b
        where λ ~ Beta(α, α)

    CutMix Mode:
        Cuts rectangular patches and pastes between images:
        x_mixed = mask * x_a + (1 - mask) * x_b
        λ is the fraction of each image kept.

    Labels are untouched; ``states[MIX_PARTNER]`` and ``batch_state[MIX_LAMBDA]`` tell the loss
    how the records were mixed.

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

    def apply_batch(
        self, batch: Batch, keys: jax.Array | None, stats: dict[str, Any] | None
    ) -> Batch:
        """Mix every record with a partner record of the batch.

        Args:
            batch: The batch.
            keys: One key per record; the whole batch mixes with the first record's, so a
                resumed run mixes the same batch the same way on any number of hosts.
            stats: Unused.

        Returns:
            The batch with ``data_field`` mixed, labels untouched, each record's partner row in
            ``states[MIX_PARTNER]`` and the kept fraction in ``batch_state[MIX_LAMBDA]``.

        Raises:
            ValueError: If the data has no ``data_field``, or CutMix's field is not a batch of
                images ``(B, H, W, C)``.
        """
        del stats
        assert keys is not None  # a stochastic operator is always handed keys
        data = batch.data
        field = self.config.data_field
        if not isinstance(data, dict) or field not in data:
            available = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise ValueError(
                f"BatchMixOperator mixes data[{field!r}], which this batch lacks: {available}"
            )
        if batch.batch_size == 0:
            # No record to mix and no first record's key: every (absent) record keeps all of
            # itself, which the state says as it would for any batch.
            return batch.replace(
                states={**batch.states, MIX_PARTNER: jnp.zeros((0,), jnp.int32)},
                batch_state={**batch.batch_state, MIX_LAMBDA: jnp.ones((), data[field].dtype)},
            )
        if self.config.mode == "mixup":
            k_lambda, k_partner = jax.random.split(keys[0])
            box_keys = None
        else:
            k_lambda, k_partner, k_x, k_y = jax.random.split(keys[0], 4)
            box_keys = (k_x, k_y)
        lam = jax.random.beta(k_lambda, self.config.alpha, self.config.alpha)
        partner = jax.random.permutation(k_partner, jnp.arange(batch.batch_size, dtype=jnp.int32))
        values = data[field]
        if box_keys is None:
            # values[partner] + lam * (values - values[partner]) equals
            # lam * values + (1 - lam) * values[partner] without lam + (1 - lam) != 1 rounding.
            mixed, kept = values[partner] + lam * (values - values[partner]), lam
        else:
            mixed, kept = _cutmix(values, values[partner], lam, box_keys, field)
        return batch.replace(
            data={**data, field: mixed},
            states={**batch.states, MIX_PARTNER: partner},
            batch_state={**batch.batch_state, MIX_LAMBDA: kept},
        )


def _cutmix(
    images: jax.Array,
    partners: jax.Array,
    lam: jax.Array,
    box_keys: tuple[jax.Array, jax.Array],
    field: str,
) -> tuple[jax.Array, jax.Array]:
    """Paste a box of each partner image into each image; return them and the fraction kept.

    The box covers ``1 - lam`` of the image before clipping at the border (Yun et al. 2019); the
    fraction kept is measured after clipping.

    Args:
        images: The batch's images, ``(B, H, W, C)``.
        partners: Each image's partner image, the same shape.
        lam: The drawn mixing ratio.
        box_keys: The keys for the box centre's column and row.
        field: The data field's name, for the error.

    Returns:
        The mixed images and the fraction of each image kept.

    Raises:
        ValueError: If ``images`` is not ``(B, H, W, C)``.
    """
    if images.ndim != 4:
        raise ValueError(
            f"CutMix pastes boxes into images (B, H, W, C); data[{field!r}] has shape "
            f"{images.shape}"
        )
    height, width = images.shape[1:3]
    k_x, k_y = box_keys
    cut_ratio = jnp.sqrt(1.0 - lam)
    cut_h, cut_w = height * cut_ratio, width * cut_ratio
    cx = jax.random.randint(k_x, (), 0, width).astype(jnp.float32)
    cy = jax.random.randint(k_y, (), 0, height).astype(jnp.float32)
    x1, x2 = jnp.clip(cx - cut_w / 2, 0, width), jnp.clip(cx + cut_w / 2, 0, width)
    y1, y2 = jnp.clip(cy - cut_h / 2, 0, height), jnp.clip(cy + cut_h / 2, 0, height)
    # A mask from coordinate grids, not a dynamic slice, so the box may be traced.
    yy, xx = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32), jnp.arange(width, dtype=jnp.float32), indexing="ij"
    )
    inside = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)
    keep = jnp.where(inside, 0.0, 1.0)[None, :, :, None]
    return keep * images + (1 - keep) * partners, jnp.mean(keep)
