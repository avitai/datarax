"""A stage that keeps the identity of the batch it last ran on, for tests of what reaches the DAG.

:class:`IdentitySpy` writes the batch's ``indices``, ``epochs`` and ``draws`` into its own
Variables, which ``for batch in pipe`` writes back after each compiled DAG call, so a test reads
what reached the stages inside the call. :func:`check_identity_reaches_the_stages` compares that,
batch by batch, with the host stage's own batches of an equal pipeline (brief T7).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.core.element_batch import Batch
from datarax.pipeline import Pipeline


class IdentitySpy(nnx.Module):
    """A stage keeping the last batch's ``indices``, ``epochs`` and ``draws`` in its state."""

    def __init__(self, batch_size: int) -> None:
        """Hold zeros of a batch's identity shapes.

        Args:
            batch_size: Records per batch.
        """
        self.indices = nnx.Variable(jnp.zeros((batch_size, 2), jnp.uint32))
        self.epochs = nnx.Variable(jnp.zeros((batch_size,), jnp.int32))
        self.draws = nnx.Variable(jnp.zeros((batch_size,), jnp.int32))

    def __call__(self, batch: Batch) -> Batch:
        """Keep the batch's identity and return the batch unchanged."""
        self.indices.set_value(jnp.asarray(batch.indices))
        self.epochs.set_value(jnp.asarray(batch.epochs))
        self.draws.set_value(jnp.asarray(batch.draws))
        return batch


def check_identity_reaches_the_stages(
    make_source: Callable[[], DataSourceModule], *, batch_size: int, num_epochs: int = 2
) -> int:
    """Check that each batch's identity reaches the DAG as the host stage read it.

    Args:
        make_source: Builds the source; called twice, for the pipeline and its reference.
        batch_size: Records per batch.
        num_epochs: Epochs the run serves.

    Returns:
        The batches compared.
    """

    def pipeline(stages: list[nnx.Module]) -> Pipeline:
        return Pipeline(
            source=make_source(),
            stages=stages,
            batch_size=batch_size,
            rngs=nnx.Rngs(1),
            shuffle=True,
            num_epochs=num_epochs,
        )

    expected = list(pipeline([]).raw_batches())
    spy = IdentitySpy(batch_size)
    seen = [
        (np.asarray(spy.indices[...]), np.asarray(spy.epochs[...]), np.asarray(spy.draws[...]))
        for _ in pipeline([spy])
    ]
    assert len(seen) == len(expected) > 0
    for (indices, epochs, draws), batch in zip(seen, expected, strict=True):
        np.testing.assert_array_equal(indices, np.asarray(batch.indices))
        np.testing.assert_array_equal(epochs, np.asarray(batch.epochs))
        np.testing.assert_array_equal(draws, np.asarray(batch.draws))
    return len(seen)
