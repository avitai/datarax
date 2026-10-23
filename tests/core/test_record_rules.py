"""N2's rules over a record tree: which nodes break which rule.

A record is nested dicts of arrays and field nodes. Every key is a non-empty string that neither
starts nor ends with "/" nor holds "//"; every node holding fields is a dict; and no two nodes,
groups included, share a spelling (their keys joined by "/"). The rules read structure only, so
they apply to data and to a spec alike, and a field node is one leaf.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from datarax.core.field_paths import record_problems, RecordProblem
from datarax.core.maybe import Maybe


def _depth() -> Maybe:
    return Maybe(np.zeros(3, np.float32), np.array(True))


class TestAValidRecord:
    def test_nested_groups_and_literal_keys_with_distinct_spellings_pass(self) -> None:
        record = {
            "image": np.zeros((4, 4), np.float32),
            "image/id": np.int32(7),
            "objects": {"bbox": np.zeros((3, 4)), "label": np.zeros(3)},
            "State/Province": np.int32(1),
        }
        assert record_problems(record) == ()

    def test_a_field_node_is_one_leaf(self) -> None:
        assert record_problems({"depth": _depth(), "meta": {"mask": _depth()}}) == ()

    def test_a_spec_is_checked_as_its_record(self) -> None:
        spec = {"x": jax.ShapeDtypeStruct((3,), np.float32), "meta": {"id": _depth()}}
        assert record_problems(spec) == ()

    def test_a_single_array_is_a_record_with_no_keys(self) -> None:
        assert record_problems(np.zeros(3)) == ()


class TestKeys:
    @pytest.mark.parametrize(
        ("key", "rule"),
        [("/a", "starts with '/'"), ("a/", "ends with '/'"), ("a//b", "holds '//'")],
    )
    def test_an_exotic_key_is_named_with_the_rule_it_breaks(self, key: str, rule: str) -> None:
        assert record_problems({"meta": {key: np.zeros(1)}}) == (
            RecordProblem(keys=("meta", key), rule=rule),
        )

    def test_the_empty_key_is_refused(self) -> None:
        assert record_problems({"": np.zeros(1)}) == (RecordProblem(keys=("",), rule="is empty"),)

    def test_a_key_that_is_not_a_string_is_refused_holding_the_key_as_found(self) -> None:
        """The key stays ``0``: as ``"0"`` it would read as a string key that is legal."""
        assert record_problems({0: np.zeros(1)}) == (  # type: ignore[dict-item]
            RecordProblem(keys=(0,), rule="is not a string"),
        )

    def test_every_broken_key_of_the_record_is_named_in_path_order(self) -> None:
        problems = record_problems({"a//b": np.zeros(1), "ok": {"": np.zeros(1)}})
        assert [p.keys for p in problems] == [("a//b",), ("ok", "")]


class TestNodes:
    def test_a_list_node_is_refused_by_its_key_tuple(self) -> None:
        assert record_problems({"c": [np.zeros(1), np.zeros(1)]}) == (
            RecordProblem(keys=("c",), rule="is a list"),
        )

    def test_a_tuple_node_is_refused_as_a_list(self) -> None:
        assert record_problems({"meta": {"box": (np.zeros(1),)}}) == (
            RecordProblem(keys=("meta", "box"), rule="is a list"),
        )


class TestUniqueSpellings:
    def test_a_literal_key_spelled_as_a_nested_field_is_refused_naming_both(self) -> None:
        """Defect 13's record: a stage adds ``meta/id`` beside the group ``meta`` holding ``id``."""
        problems = record_problems({"meta": {"id": np.zeros(1)}, "meta/id": np.zeros(1)})
        assert problems == (
            RecordProblem(keys=("meta", "id"), rule="shares its spelling", other=("meta/id",)),
        )

    def test_groups_are_covered_not_only_leaves(self) -> None:
        """A leaf-only rule, as TFDS checks, misses a group spelled as another group."""
        record = {"a": {"b": {"c": np.zeros(1)}}, "a/b": {"d": np.zeros(1)}}
        assert record_problems(record) == (
            RecordProblem(keys=("a", "b"), rule="shares its spelling", other=("a/b",)),
        )

    def test_every_spelling_named_twice_is_reported_once(self) -> None:
        record = {"a": {"b": {"c": np.zeros(1)}}, "a/b": {"c": np.zeros(1)}}
        assert [(p.keys, p.other) for p in record_problems(record)] == [
            (("a", "b"), ("a/b",)),
            (("a", "b", "c"), ("a/b", "c")),
        ]

    def test_a_field_nodes_members_are_not_spellings(self) -> None:
        """``depth/value`` is no node of the record, so a literal ``depth/value`` key is legal."""
        assert record_problems({"depth": _depth(), "depth/value": np.zeros(3)}) == ()


class TestTheWordsOfAProblem:
    """Every site that refuses a record prints a problem in these words (G9, the display rule)."""

    def test_an_exotic_key_reads_as_g9s_sentence(self) -> None:
        (problem,) = record_problems({"a//b": np.zeros(1)})
        assert str(problem) == (
            "the key 'a//b', which holds '//'; a key may not start or end with '/' or hold '//'"
        )

    def test_a_nested_exotic_key_names_its_group(self) -> None:
        (problem,) = record_problems({"meta": {"/a": np.zeros(1)}})
        assert str(problem) == (
            "the key '/a' in 'meta', which starts with '/'; a key may not start or end with '/' "
            "or hold '//'"
        )

    def test_the_empty_key(self) -> None:
        (problem,) = record_problems({"meta": {"": np.zeros(1)}})
        assert str(problem) == "an empty key in 'meta'; a key is a non-empty string"

    def test_a_key_that_is_not_a_string(self) -> None:
        (problem,) = record_problems({0: np.zeros(1)})  # type: ignore[dict-item]
        assert str(problem) == "the key 0, which is not a string; a key is a non-empty string"

    def test_a_list_node(self) -> None:
        (problem,) = record_problems({"c": [np.zeros(1)]})
        assert str(problem) == (
            "'c' is a list; a record's nodes are dicts and its fields arrays or field nodes"
        )

    def test_two_nodes_with_one_spelling_read_as_the_display_example(self) -> None:
        (problem,) = record_problems({"meta": {"score": np.zeros(1)}, "meta/score": np.zeros(1)})
        assert str(problem) == (
            "'meta/score' names two fields: ('meta', 'score') and ('meta/score',)"
        )
