"""``Batch`` reads are key lookups and ``update_data`` a key merge, with no path syntax (D5).

Spellings resolve only where a configuration names a field (operators, sources, filters), so a
batch read adds no per-call resolution: nested data reads as ``batch["meta"]["id"]``, and a literal
key such as ``image/id`` reads as one key.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from datarax.core import batch_ops
from datarax.core.element_batch import Element


def _batch():
    return batch_ops.from_arrays(
        {
            "meta": {"id": jnp.arange(2, dtype=jnp.int32)},
            "image/id": jnp.arange(2, dtype=jnp.int32) + 10,
        }
    )


class TestBatchReads:
    def test_nested_data_reads_one_key_at_a_time(self) -> None:
        np.testing.assert_array_equal(_batch()["meta"]["id"], [0, 1])

    def test_a_literal_key_holding_a_slash_reads_as_one_key(self) -> None:
        np.testing.assert_array_equal(_batch()["image/id"], [10, 11])

    def test_a_spelling_is_not_a_path_for_a_batch_read(self) -> None:
        batch = _batch()
        assert "meta/id" not in batch
        assert batch.get("meta/id") is None


class TestUpdateData:
    def test_a_key_holding_a_slash_is_one_key_not_a_level(self) -> None:
        element = Element({"meta": {"id": jnp.int32(0)}})
        updated = element.update_data({"meta/score": jnp.float32(1.0)})
        assert set(updated.data) == {"meta", "meta/score"}
        assert set(updated.data["meta"]) == {"id"}

    def test_an_existing_key_is_replaced(self) -> None:
        updated = Element({"x": jnp.int32(0)}).update_data({"x": jnp.int32(5)})
        assert int(updated.data["x"]) == 5
