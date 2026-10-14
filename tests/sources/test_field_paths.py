"""A record's fields named by their path: the one naming every source and its refusals share.

A field is a leaf of a record's pytree, named by its path's steps joined by ``/``: a dictionary
key by the key itself, a list position as JAX prints it (``[0]``). Provenance keys, the field names
a refusal prints and the columns a source holds on disk are all named this way.
"""

from __future__ import annotations

import numpy as np
import pytest

from datarax.sources import _field_paths
from datarax.sources._field_paths import by_field, field_name, field_names, LEVEL, nested


def _record() -> dict:
    return {"image": np.zeros(2), "meta": {"label": np.int32(1), "box": [np.zeros(4), np.ones(4)]}}


class TestFieldNames:
    def test_a_dictionary_key_is_its_key_and_levels_join_with_a_slash(self) -> None:
        assert field_names({"meta": {"label": 1}}) == ["meta/label"]

    def test_a_list_position_prints_as_jax_prints_it(self) -> None:
        """``keystr(simple=True)`` would print ``0``, which a key named ``0`` also prints."""
        assert field_names({"box": [1, 2]}) == ["box/[0]", "box/[1]"]

    def test_every_leaf_is_named_in_jax_s_flattening_order(self) -> None:
        assert field_names(_record()) == ["image", "meta/box/[0]", "meta/box/[1]", "meta/label"]

    def test_a_leaf_at_the_root_has_the_empty_name(self) -> None:
        assert field_names(np.zeros(3)) == [""]

    def test_field_name_names_one_path(self) -> None:
        import jax  # noqa: PLC0415 - the path type is JAX's

        paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(_record())[0]]
        assert [field_name(path) for path in paths] == field_names(_record())


class TestNested:
    def test_names_become_one_mapping_level_per_step(self) -> None:
        assert nested({"meta/label": 1, "meta/box": 2, "image": 3}) == {
            "meta": {"label": 1, "box": 2},
            "image": 3,
        }

    def test_nesting_inverts_naming(self) -> None:
        record = {"image": np.zeros(2), "meta": {"label": np.int32(1)}}
        import jax  # noqa: PLC0415 - the path type is JAX's

        leaves = {
            field_name(p): leaf for p, leaf in jax.tree_util.tree_flatten_with_path(record)[0]
        }
        assert field_names(nested(leaves)) == field_names(record)

    def test_one_separator_names_and_nests(self) -> None:
        assert LEVEL == "/"
        assert nested({f"a{LEVEL}b": 1}) == {"a": {"b": 1}}


class TestByField:
    def test_each_leaf_keyed_by_its_field_name(self) -> None:
        record = {"image": 1, "meta": {"label": 2}}
        assert by_field(record) == {"image": 1, "meta/label": 2}

    def test_nesting_inverts_it(self) -> None:
        record = {"image": 1, "meta": {"label": 2, "box": 3}}
        assert nested(by_field(record)) == record

    def test_a_leaf_rule_keeps_lists_whole(self) -> None:
        """A JSON document's fields: mappings are levels; a list is one value."""
        document = {"source": {"name": "mnist", "splits": ["train", "test"]}, "version": 1}
        leaves = by_field(document, is_leaf=lambda value: not isinstance(value, dict))
        assert leaves == {"source/name": "mnist", "source/splits": ["train", "test"], "version": 1}

    def test_an_empty_mapping_has_no_field(self) -> None:
        """So ``{}`` differs from a mapping holding fields, field by field."""
        assert by_field({"a": {}}, is_leaf=lambda value: not isinstance(value, dict)) == {}

    def test_a_leaf_at_the_root_has_the_empty_name(self) -> None:
        assert by_field(7, is_leaf=lambda value: not isinstance(value, dict)) == {"": 7}


@pytest.mark.parametrize("name", ["field_name", "field_names"])
def test_the_module_is_the_one_home_of_field_naming(name: str) -> None:
    """``eager_source`` reads these names; no second copy of the joiner is kept there."""
    from datarax.sources import eager_source  # noqa: PLC0415 - the module under test

    assert getattr(eager_source, name) is getattr(_field_paths, name)
    assert f"_{name}" not in vars(eager_source)


def test_the_column_sets_nest_with_the_same_function() -> None:
    from datarax.sources import streaming_disk_source  # noqa: PLC0415 - the module under test

    assert streaming_disk_source.nested is _field_paths.nested
    assert "_nested" not in vars(streaming_disk_source)
    assert "_LEVEL" not in vars(streaming_disk_source)
