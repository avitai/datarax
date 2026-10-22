"""One naming of record fields: resolution, creation, display, name order and conversion.

A field is the tuple of dict keys from the record's root. A key is a non-empty string and may
hold "/": ``State/Province`` is one field. A name is a spelling (the keys joined by "/") or a key
tuple. A spelling resolves against the record's structure, never by splitting it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import numpy as np
import pytest

import datarax.core.field_paths as field_paths
from datarax.core.field_paths import (
    as_field_name,
    as_field_names,
    create_path,
    FieldCreateError,
    FieldNotFoundError,
    get_field,
    label,
    name_order,
    path_label,
    resolve,
    set_field,
    sorted_names,
    spelling,
)
from datarax.core.maybe import Maybe


def _coco() -> dict:
    """TFDS COCO's shape: a leaf ``image`` beside the literal key ``image/id``, and a group."""
    return {
        "image": np.zeros((4, 4), np.float32),
        "image/id": np.int32(7),
        "objects": {"bbox": np.zeros((3, 4), np.float32), "label": np.zeros(3, np.int32)},
    }


class TestResolve:
    def test_a_spelling_resolves_through_groups(self) -> None:
        assert resolve(_coco(), "objects/bbox") == ("objects", "bbox")

    def test_a_literal_key_holding_a_slash_is_one_field(self) -> None:
        assert resolve(_coco(), "image/id") == ("image/id",)

    def test_a_key_tuple_names_its_exact_path(self) -> None:
        assert resolve(_coco(), ("objects", "label")) == ("objects", "label")

    def test_a_list_of_keys_is_a_key_tuple(self) -> None:
        """A list of keys, as TOML writes one, names the same field everywhere a name is taken."""
        assert resolve(_coco(), ["objects", "label"]) == ("objects", "label")
        assert get_field(_coco(), ["image/id"]) == 7
        assert set_field({}, ["meta", "score"], 1) == {"meta": {"score": 1}}

    def test_a_key_holding_a_dot_is_reached(self) -> None:
        """pandas names a duplicate column ``a.1``; a dotted split could never reach it."""
        assert resolve({"a": np.zeros(2), "a.1": np.ones(2)}, "a.1") == ("a.1",)

    def test_an_unknown_name_is_refused_listing_the_fields(self) -> None:
        with pytest.raises(FieldNotFoundError) as caught:
            resolve(_coco(), "objects/area")
        message = str(caught.value)
        assert "'objects/area'" in message
        assert "'objects/bbox'" in message and "'image/id' (key tuple ('image/id',))" in message

    def test_a_key_tuple_through_a_leaf_is_unknown(self) -> None:
        with pytest.raises(FieldNotFoundError, match="image"):
            resolve(_coco(), ("image", "id"))

    def test_a_field_node_is_one_field_and_its_members_are_not_names(self) -> None:
        data = {"depth": Maybe(np.zeros(3), np.array(True))}
        assert resolve(data, "depth") == ("depth",)
        with pytest.raises(FieldNotFoundError, match="depth/value"):
            resolve(data, "depth/value")

    def test_two_nodes_sharing_a_spelling_are_refused_by_key_tuple(self) -> None:
        """Uniqueness is checked where records enter; resolution still refuses the ambiguity."""
        data = {"a": {"b": np.zeros(1)}, "a/b": np.ones(1)}
        with pytest.raises(ValueError, match=r"\('a', 'b'\).*\('a/b',\)"):
            resolve(data, "a/b")


class TestCreate:
    def test_a_hit_is_a_write_onto_that_node(self) -> None:
        assert create_path(_coco(), "image/id", owner="Op", param="target_key") == ("image/id",)

    def test_a_miss_descends_into_the_longest_group_and_keeps_the_rest_as_one_key(self) -> None:
        assert create_path(_coco(), "objects/score", owner="Op", param="target_key") == (
            "objects",
            "score",
        )
        assert create_path(_coco(), "objects/score/raw", owner="Op", param="target_key") == (
            "objects",
            "score/raw",
        )

    def test_the_longest_group_wins_over_a_shorter_one(self) -> None:
        data = {"a/b": {"x": np.zeros(1)}, "a": {"c": np.zeros(1)}}
        assert create_path(data, "a/b/y", owner="Op", param="target_key") == ("a/b", "y")

    def test_with_no_group_the_whole_spelling_is_one_top_level_key(self) -> None:
        data = {"Country": np.zeros(1)}
        assert create_path(data, "State/Province", owner="Op", param="target_key") == (
            "State/Province",
        )

    def test_a_spelling_extending_a_field_is_refused_offering_the_key_tuple(self) -> None:
        data = {"meta": np.zeros(2, np.int32)}
        with pytest.raises(FieldCreateError) as caught:
            create_path(data, "meta/size", owner="BrightnessOperator", param="target_key")
        assert str(caught.value) == (
            "BrightnessOperator creates 'meta/size', but 'meta' is a field (int32[2]), not a "
            "group, so 'meta/size' cannot be nested under it. For a field whose name holds '/', "
            "beside 'meta', pass the key tuple target_key=('meta/size',) "
            '(TOML: target_key = ["meta/size"]); otherwise name another output'
        )

    def test_the_refusal_names_the_longest_field_extended(self) -> None:
        with pytest.raises(FieldCreateError, match=r"'image/id' is a field"):
            create_path(_coco(), "image/id/f0", owner="Op", param="target_key")

    def test_a_spelling_extending_a_field_inside_a_group_is_refused(self) -> None:
        with pytest.raises(FieldCreateError, match=r"'objects/bbox' is a field"):
            create_path(_coco(), "objects/bbox/area", owner="Op", param="target_key")

    def test_a_spelling_extending_a_field_node_is_refused(self) -> None:
        data = {"depth": Maybe(np.zeros(3), np.array(True))}
        with pytest.raises(FieldCreateError, match=r"'depth' is a field \(Maybe\)"):
            create_path(data, "depth/value", owner="Op", param="target_key")

    def test_the_offered_key_tuple_is_accepted_and_resolves_back(self) -> None:
        data = {"meta": np.zeros(2, np.int32)}
        keys = create_path(data, ("meta/size",), owner="Op", param="target_key")
        assert keys == ("meta/size",)
        assert resolve(set_field(data, keys, np.int32(3)), "meta/size") == ("meta/size",)

    def test_a_key_tuple_creates_exactly_its_path(self) -> None:
        assert create_path({}, ("meta", "score"), owner="Op", param="target_key") == (
            "meta",
            "score",
        )

    def test_a_key_tuple_through_a_field_is_refused(self) -> None:
        with pytest.raises(FieldCreateError, match=r"passes through 'image'"):
            create_path(_coco(), ("image", "f0"), owner="Op", param="target_key")

    def test_a_key_tuple_whose_spelling_another_node_has_is_refused(self) -> None:
        data = {"a/b": np.zeros(1)}
        with pytest.raises(FieldCreateError, match=r"'a/b'"):
            create_path(data, ("a", "b"), owner="Op", param="target_key")

    @pytest.mark.parametrize("name", [("",), ("a", ""), ("/a",), ("a/",), ("a//b",)])
    def test_a_key_tuple_with_an_empty_or_exotic_key_is_refused(
        self, name: tuple[str, ...]
    ) -> None:
        with pytest.raises(FieldCreateError, match="empty or starts or ends with '/'"):
            create_path({}, name, owner="Op", param="target_key")

    @pytest.mark.parametrize("name", ["", "a/", "/a", "x//y"])
    def test_a_spelling_giving_an_empty_or_exotic_key_is_refused(self, name: str) -> None:
        with pytest.raises(FieldCreateError, match="empty or starts or ends with '/'"):
            create_path({"a": {"x": np.zeros(1)}}, name, owner="Op", param="target_key")


class TestGetAndSet:
    def test_get_reads_by_spelling_and_by_key_tuple(self) -> None:
        assert get_field(_coco(), "image/id") == 7
        assert get_field(_coco(), ("objects", "label")).shape == (3,)

    def test_set_writes_a_copy_and_creates_by_c3(self) -> None:
        record = _coco()
        updated = set_field(record, "objects/score", np.float32(1.0))
        assert "score" in updated["objects"] and "score" not in record["objects"]
        assert set_field(record, "image", np.ones((4, 4)))["image"].sum() == 16

    def test_a_dot_is_part_of_a_key_not_a_level(self) -> None:
        assert set_field({}, "audio.signal", 1) == {"audio.signal": 1}

    def test_set_creates_the_groups_of_a_key_tuple(self) -> None:
        assert set_field({"label": 3}, ("audio", "signal"), 1) == {
            "label": 3,
            "audio": {"signal": 1},
        }

    def test_set_refuses_what_create_refuses(self) -> None:
        with pytest.raises(FieldCreateError, match="'meta' is a field"):
            set_field({"meta": np.zeros(2)}, "meta/size", 1)


class TestDisplay:
    def test_a_node_prints_as_its_spelling_in_quotes(self) -> None:
        assert label(("meta", "id")) == "'meta/id'"
        assert spelling(("meta", "id")) == "meta/id"

    @pytest.mark.parametrize(
        ("keys", "text"),
        [
            (("image/id",), "'image/id' (key tuple ('image/id',))"),
            (("a", "b/c"), "'a/b/c' (key tuple ('a', 'b/c'))"),
            (("",), "'' (key tuple ('',))"),
        ],
    )
    def test_the_key_tuple_is_added_where_the_spelling_hides_structure(
        self, keys: tuple[str, ...], text: str
    ) -> None:
        assert label(keys) == text

    def test_a_field_node_member_prints_as_an_attribute(self) -> None:
        data = {"depth": Maybe(np.zeros(3), np.array(True))}
        paths = [p for p, _ in jax.tree_util.tree_flatten_with_path(data)[0]]
        assert [path_label(p) for p in paths] == ["'depth'.value", "'depth'.present"]

    def test_a_list_position_prints_as_jax_prints_it(self) -> None:
        paths = [p for p, _ in jax.tree_util.tree_flatten_with_path({"c": [1, 2]})[0]]
        assert [path_label(p) for p in paths] == ["'c'[0]", "'c'[1]"]


class TestNameOrder:
    def test_spellings_come_before_key_tuples_each_in_code_point_order(self) -> None:
        names = (("meta", "id"), "meta/score", ("image/id",), "image")
        assert sorted_names(names) == ("image", "meta/score", ("image/id",), ("meta", "id"))

    def test_two_orders_of_one_selection_give_one_stored_order(self) -> None:
        assert sorted_names(("b", ("a", "x"), "a")) == sorted_names((("a", "x"), "a", "b"))

    def test_a_list_of_keys_is_held_as_a_key_tuple(self) -> None:
        assert sorted_names([["a", "x"], "b"]) == ("b", ("a", "x"))

    def test_python_s_default_sort_cannot_order_a_mix(self) -> None:
        """The reason the order is named: ``sorted`` raises on str against tuple."""
        with pytest.raises(TypeError):
            sorted(["a", ("b",)])  # type: ignore[type-var]
        assert name_order("a") < name_order(("a",))


class TestConversion:
    def test_a_one_field_value_is_a_spelling_or_one_key_tuple(self) -> None:
        assert as_field_name("meta/id", "target_key") == "meta/id"
        assert as_field_name(["meta", "id"], "target_key") == ("meta", "id")

    def test_a_several_field_value_is_a_list_of_names(self) -> None:
        assert as_field_names(["image", ["meta", "score"]], "fields") == (
            "image",
            ("meta", "score"),
        )

    def test_a_bare_string_for_several_fields_is_refused(self) -> None:
        with pytest.raises(TypeError, match="fields names several fields"):
            as_field_names("image", "fields")

    @pytest.mark.parametrize("value", [[], [1], [["a"]], 3, None])
    def test_a_value_that_names_no_field_is_refused(self, value: object) -> None:
        with pytest.raises(TypeError, match="target_key names one field"):
            as_field_name(value, "target_key")


def test_the_naming_module_imports_no_record_or_node_type() -> None:
    """``core.maybe`` imports ``core.element_batch``; if naming imported either, every module
    that names fields would sit inside that cycle. It reaches field nodes through
    ``core.field_nodes`` only."""
    tree = ast.parse(Path(field_paths.__file__).read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("datarax")
    } | {
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for name in node.names
        if name.name.startswith("datarax")
    }
    assert imported == {"datarax.core.field_nodes"}
