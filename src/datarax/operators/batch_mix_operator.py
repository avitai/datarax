"""BatchMixOperator - MixUp and CutMix batch augmentation.

Mixing pairs each record with another record of the same batch, so the operator has no per-record
form: it overrides ``apply_batch`` alone and draws from its first record's key.

Supported modes:

- mixup: each image becomes ``lam * x + (1 - lam) * x[partner]`` (Zhang et al. 2018)
- cutmix: a box of the partner's image is pasted into each image (Yun et al. 2019)

Labels are left untouched. The operator writes what it did for the loss to read in
``batch.mix``: each row's partner row in ``mix.partners`` and the fraction of each image kept in
``mix.lams``, one per row, so ``lams * loss(y) + (1 - lams) * loss(y[partners])`` is the mixed
loss for integer and one-hot labels alike. A batch is mixed once: a second mixing operator is
refused (``MixedTwiceError``), since a loss reads one partner and one coefficient per row.

``lam`` is drawn from Beta(alpha, alpha) with JAX's loop-free sampler (``method="approximate"``),
so a GPU queues the batch program without the host waiting on it; JAX's default rejection sampler
lowers to ``while`` loops that copy their predicate to the host on every iteration. The draw is
within a Kolmogorov-Smirnov distance of 6.4e-5 of Beta(alpha, alpha) for alpha in [0.01, 2], less
above: the sampler's one approximate step, a chi-square quantile series at shape alpha + 4, is
turned into Gamma(alpha) exactly by Stuart's identity, and its scale error cancels in the ratio of
the two Gamma draws. That is the effect of changing alpha by about 0.05%.
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import BatchMixOperatorConfig
from datarax.core.element_batch import Batch, Mix, MixedTwiceError
from datarax.core.field_paths import field_listing, field_paths, refuse_field_nodes
from datarax.core.operator import OperatorModule


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

    Labels are untouched; ``batch.mix`` tells the loss how the records were mixed.

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

    def apply_batch(  # noqa: DOC503 - refuse_field_nodes raises the TypeError
        self, batch: Batch, keys: jax.Array | None, stats: dict[str, Any] | None
    ) -> Batch:
        """Mix every record with a partner record of the batch.

        Args:
            batch: The batch.
            keys: One key per record; the whole batch mixes with the first record's, so a
                resumed run mixes the same batch the same way on any number of hosts.
            stats: Unused.

        Returns:
            The batch with ``data_field`` mixed, labels untouched, and ``mix`` holding each row's
            partner row and kept fraction: MixUp's in the field's dtype, CutMix's float32.

        Raises:
            MixedTwiceError: If the batch is already mixed.
            ValueError: If the data has no ``data_field``, or CutMix's field is not a batch of
                images ``(B, H, W, C)``.
            TypeError: If ``data_field`` is a ``Maybe``: a missing value has nothing to mix.
        """
        del stats
        assert keys is not None  # a stochastic operator is always handed keys
        if batch.mix is not None:
            raise MixedTwiceError(type(self).__name__)
        data = batch.data
        field = self.config.data_field
        if not isinstance(data, dict) or field not in data:
            available = (
                field_listing(field_paths(data)) if isinstance(data, dict) else type(data).__name__
            )
            raise ValueError(
                f"BatchMixOperator mixes the data field {field!r}, which this batch lacks; its "
                f"data fields: {available}"
            )
        refuse_field_nodes(
            data[field], "BatchMixOperator mixes a field's values between records", (field,)
        )
        if batch.batch_size == 0:
            # No record to mix and no first record's key: the batch is mixed, with no rows.
            dtype = data[field].dtype if self.config.mode == "mixup" else jnp.float32
            return batch.replace(mix=Mix(jnp.zeros((0,), jnp.int32), jnp.ones((0,), dtype)))
        if self.config.mode == "mixup":
            k_lambda, k_partner = jax.random.split(keys[0])
            box_keys = None
        else:
            k_lambda, k_partner, k_x, k_y = jax.random.split(keys[0], 4)
            box_keys = (k_x, k_y)
        lam = jax.random.beta(k_lambda, self.config.alpha, self.config.alpha, method="approximate")
        partner = jax.random.permutation(k_partner, jnp.arange(batch.batch_size, dtype=jnp.int32))
        values = data[field]
        lam = lam.astype(values.dtype)
        if box_keys is None:
            # values[partner] + lam * (values - values[partner]) equals
            # lam * values + (1 - lam) * values[partner] without lam + (1 - lam) != 1 rounding.
            mixed, kept = values[partner] + lam * (values - values[partner]), lam
        else:
            mixed, kept = _cutmix(values, values[partner], lam, box_keys, field)
        # The batch's one draw, written to every row.
        lams = jnp.broadcast_to(kept, (batch.batch_size,))
        return batch.replace(data={**data, field: mixed}, mix=Mix(partner, lams))


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
