"""Field-selection DAG node.

Routes a subset of a batch's fields downstream — used to feed one modality (or a
chosen set of fields) into a branch of a ``Pipeline.from_dag`` graph. Field
selection is static, so the node is safe under ``jax.jit`` and ``nnx.scan``.
"""

from __future__ import annotations

from typing import Any

from flax import nnx

from datarax.core.element_batch import Batch
from datarax.core.field_paths import (
    as_field_names,
    FieldNames,
    FieldNotFoundError,
    get_field,
    resolve,
    set_field,
)


class SplitField(nnx.Module):
    """DAG node that keeps only a named subset of a batch's data fields.

    A name resolves against the batch's data (``core.field_paths``): ``"meta/id"`` keeps
    ``data["meta"]["id"]`` inside its group, and a literal key such as ``"image/id"`` is one field.
    Absent fields are skipped, so the node composes with upstream stages that add or drop
    fields. Record identities and state pass through.
    """

    def __init__(self, fields: FieldNames) -> None:
        """Store the field names to keep.

        Args:
            fields: Names of the fields to route downstream: spellings or key tuples (a list of
                strings is one key tuple, as TOML writes it).
        """
        self._fields = as_field_names(fields, "SplitField's fields")

    def __call__(self, batch: Batch) -> Batch:
        """Return ``batch`` with only the configured fields of its data.

        Args:
            batch: Incoming batch.

        Returns:
            The batch with its data restricted to the configured fields.
        """
        kept: dict[str, Any] = {}
        for name in self._fields:
            try:
                keys = resolve(batch.data, name)
            except FieldNotFoundError:
                continue
            kept = set_field(kept, keys, get_field(batch.data, keys), owner="SplitField")
        return batch.replace(data=kept)
