"""Field-selection DAG node.

Routes a subset of a batch's fields downstream — used to feed one modality (or a
chosen set of fields) into a branch of a ``Pipeline.from_dag`` graph. Field
selection is static, so the node is safe under ``jax.jit`` and ``nnx.scan``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flax import nnx


class SplitField(nnx.Module):
    """DAG node that keeps only a named subset of a batch dict's fields.

    Absent fields are silently skipped, so the node composes with upstream stages
    that add or drop fields. Consumes and returns the raw batch dict the Pipeline
    threads between nodes.
    """

    def __init__(self, fields: Sequence[str]) -> None:
        """Store the field names to keep.

        Args:
            fields: Names of the fields to route downstream.
        """
        self._fields = tuple(fields)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return a new dict with only the configured fields present in ``batch``.

        Args:
            batch: Incoming batch dict.

        Returns:
            Batch dict restricted to the configured fields.
        """
        return {name: batch[name] for name in self._fields if name in batch}
