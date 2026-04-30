"""Verify ``valid_mask`` semantics across the batcher → Batch construction pipeline.

After Phase 1's mask-propagation audit, the Batch constructor and
``Batch.from_parts`` both default to all-True ``valid_mask``. This file locks
the contract: as long as no batcher emits explicit padding, downstream Batches
have all-True masks (no surprise gradient leakage).

Phase 4's ``scan_epoch`` will introduce end-of-epoch padding via explicit
``valid_mask=`` arguments to ``Batch.from_parts``. Until then, the existing
``DefaultBatcher`` returns shape-honest partial batches (no padding, no mask
manipulation) and Batch construction is implicitly safe.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig
from datarax.core.element_batch import Batch, Element


def test_default_batcher_partial_batch_returns_shape_honest_output() -> None:
    """``DefaultBatcher`` does NOT pad; partial last batch is returned at its actual size.

    This is the load-bearing invariant: as long as the batcher does not pad,
    no valid_mask manipulation is needed. The actual leading-dim of the
    output PyTree truthfully reflects the count of valid elements.
    """
    config = DefaultBatcherConfig(stochastic=False)
    batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
    elements = [{"x": np.array([i], dtype=np.float32)} for i in range(5)]

    batched = list(batcher(iter(elements), batch_size=2))

    # 3 batches: sizes 2, 2, 1. The last batch's leading dim is 1 — no padding.
    assert len(batched) == 3
    assert batched[-1]["x"].shape == (1, 1)


def test_batch_constructed_from_elements_has_default_all_true_mask() -> None:
    """``Batch(elements)`` sets ``valid_mask`` to all-True by default.

    The DAG data-source node calls ``Batch(elements, validate=False)`` on the
    raw record list (``data_source.py:234``). With no explicit valid_mask,
    every constructed Batch has all-True mask — correct because no padding
    has occurred at this stage.
    """
    elements = [Element(data={"x": jnp.asarray(i, dtype=jnp.float32)}) for i in range(3)]
    batch = Batch(elements, validate=False)

    np.testing.assert_array_equal(np.asarray(batch.valid_mask[...]), np.array([True, True, True]))


def test_partial_batch_constructed_via_default_path_has_correct_mask() -> None:
    """A partial batch (size 1 < batch_size 4) still has all-True mask of size 1.

    The mask reflects the ACTUAL batch size, not the intended padded size.
    Mask-weighted loss naturally handles this: ``sum(loss * mask) / sum(mask)``
    over a length-1 batch with mask=[True] is just the single loss value.
    """
    elements = [Element(data={"x": jnp.asarray(7.0, dtype=jnp.float32)})]
    batch = Batch(elements, validate=False)

    assert batch.batch_size == 1
    assert batch.valid_mask[...].shape == (1,)
    np.testing.assert_array_equal(np.asarray(batch.valid_mask[...]), np.array([True]))
