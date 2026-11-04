"""A record's fields keyed by key tuple, and the record rebuilt from them, for storage and the mix.

``flat_fields`` flattens a record's data with jax's own walk, a field node as one field, into its
fields keyed by key tuple; ``fields_nested`` rebuilds the record from key tuples alone, so a field
whose key holds ``/`` (``("a/b",)``) and a nested one (``("a", "b")``) come back apart.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from datarax.core import field_paths
from datarax.core.field_paths import fields_nested, flat_fields
from datarax.core.maybe import Maybe


_OWNER = "the test reads"


def _record() -> dict:
    return {
        "image": np.zeros(2),
        "a/b": np.ones(1),
        "a": {"b": np.full(1, 2.0)},
        "meta": {"label": np.int32(1)},
        "depth": Maybe(np.zeros(3), np.ones(3, bool)),
    }


class TestFlatFields:
    def test_each_field_is_keyed_by_its_key_tuple(self) -> None:
        fields = flat_fields(_record(), owner=_OWNER)

        assert set(fields) == {("image",), ("a/b",), ("a", "b"), ("meta", "label"), ("depth",)}
        assert fields[("a/b",)].tolist() == [1.0]
        assert fields[("a", "b")].tolist() == [2.0]

    def test_a_field_node_is_one_field(self) -> None:
        fields = flat_fields(_record(), owner=_OWNER)

        assert isinstance(fields[("depth",)], Maybe)

    def test_keys_come_in_name_order_per_level(self) -> None:
        record = {"z": 1, "b": {"y": 2, "a": 3}, "a": 4}

        assert list(flat_fields(record, owner=_OWNER)) == [("a",), ("b", "a"), ("b", "y"), ("z",)]

    def test_none_is_no_field(self) -> None:
        assert flat_fields({"a": None, "b": 1}, owner=_OWNER) == {("b",): 1}

    @pytest.mark.parametrize(
        ("record", "field"),
        [
            ({"box": [np.zeros(4), np.ones(4)]}, r"'box'\[0\]"),
            ({"box": (np.zeros(4),)}, r"'box'\[0\]"),
        ],
        ids=["list", "tuple"],
    )
    def test_a_step_that_is_not_a_dict_key_is_refused_naming_the_field(
        self, record: dict, field: str
    ) -> None:
        with pytest.raises(TypeError, match=rf"{_OWNER}.*{field}"):
            flat_fields(record, owner=_OWNER)


class TestFieldsNested:
    def test_it_inverts_flat_fields(self) -> None:
        record = _record()

        rebuilt = fields_nested(flat_fields(record, owner=_OWNER))

        assert jax.tree.structure(rebuilt) == jax.tree.structure(record)
        assert rebuilt["a/b"].tolist() == [1.0]
        assert rebuilt["a"]["b"].tolist() == [2.0]
        assert isinstance(rebuilt["depth"], Maybe)

    def test_a_literal_key_and_a_nested_one_come_back_apart(self) -> None:
        assert fields_nested({("a/b",): 1, ("a", "b"): 2}) == {"a/b": 1, "a": {"b": 2}}

    @pytest.mark.parametrize("key", ["a/b", ("a", 1), ()], ids=["spelling", "non-string", "empty"])
    def test_a_key_that_is_not_a_tuple_of_strings_is_refused(self, key: object) -> None:
        """A string would nest by its characters under flax's rebuild (a list is no mapping key)."""
        fields = {("x",): 0}
        fields[key] = 1  # type: ignore[index]  # the refused key

        with pytest.raises(TypeError, match="key tuple"):
            fields_nested(fields)


def test_storage_and_the_mix_rebuild_with_the_same_pair() -> None:
    from datarax.sources import mixed_source, streaming_disk_source  # noqa: PLC0415 - under test
    from datarax.sources.decoded_copy import _layout  # noqa: PLC0415 - the module under test

    assert streaming_disk_source.fields_nested is field_paths.fields_nested
    assert _layout.fields_nested is field_paths.fields_nested
    assert _layout.flat_fields is field_paths.flat_fields
    assert mixed_source.fields_nested is field_paths.fields_nested
    assert mixed_source.flat_fields is field_paths.flat_fields
