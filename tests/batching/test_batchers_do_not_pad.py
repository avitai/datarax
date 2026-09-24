"""Batchers never pad: a partial batch is returned at its real size, and a batch has no mask.

A padded row is not a record, so nothing in datarax produces one: the default batcher returns a
short final batch, the pipeline completes or drops an epoch's last batch, and ``Batch`` carries
records only.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig
from datarax.core.element_batch import Batch, Element


def test_default_batcher_partial_batch_returns_shape_honest_output() -> None:
    """``DefaultBatcher`` does NOT pad; partial last batch is returned at its actual size.

    The leading dim of the output PyTree is the count of records it holds; no row is padding.
    """
    config = DefaultBatcherConfig(stochastic=False)
    batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
    elements = [{"x": np.array([i], dtype=np.float32)} for i in range(5)]

    batched = list(batcher(iter(elements), batch_size=2))

    # 3 batches: sizes 2, 2, 1. The last batch's leading dim is 1 — no padding.
    assert len(batched) == 3
    assert batched[-1]["x"].shape == (1, 1)


def test_a_batch_carries_its_records_and_no_padding_mask() -> None:
    """A ``Batch`` built from elements or parts holds records only: nothing marks padding."""
    elements = [Element(data={"x": jnp.asarray(i, dtype=jnp.float32)}) for i in range(3)]
    batch = Batch(elements, validate=False)
    parts = Batch.from_parts({"x": jnp.arange(3.0)}, {})

    assert not hasattr(batch, "valid_mask")
    assert not hasattr(parts, "valid_mask")
    np.testing.assert_array_equal(np.asarray(batch.get_data()["x"]), [0.0, 1.0, 2.0])
