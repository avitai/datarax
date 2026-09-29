"""Field-selection DAG node.

Routes a subset of a batch's fields downstream — used to feed one modality (or a
chosen set of fields) into a branch of a ``Pipeline.from_dag`` graph. Field
selection is static, so the node is safe under ``jax.jit`` and ``nnx.scan``.
"""

from __future__ import annotations

from collections.abc import Sequence

from flax import nnx

from datarax.core.element_batch import Batch


class SplitField(nnx.Module):
    """DAG node that keeps only a named subset of a batch's data fields.

    Absent fields are skipped, so the node composes with upstream stages that add or drop
    fields. Record identities and state pass through.
    """

    def __init__(self, fields: Sequence[str]) -> None:
        """Store the field names to keep.

        Args:
            fields: Names of the fields to route downstream.
        """
        self._fields = tuple(fields)

    def __call__(self, batch: Batch) -> Batch:
        """Return ``batch`` with only the configured fields of its data.

        Args:
            batch: Incoming batch.

        Returns:
            The batch with its data restricted to the configured fields.
        """
        return batch.replace(data={name: batch[name] for name in self._fields if name in batch})
