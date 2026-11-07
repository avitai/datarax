"""A field node is one field held as several arrays; consumers that read arrays refuse it.

``Maybe`` is a field node: naming treats it as one field, and its members are not names. A
consumer that would treat a field as an array refuses any field node through one helper, which
names each node and gives the node's own fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from datarax.core.field_nodes import FieldNode, is_field_node
from datarax.core.field_paths import refuse_field_nodes
from datarax.core.maybe import Maybe


def _depth() -> Maybe:
    return Maybe(np.zeros(3, np.float32), np.array(True))


def test_maybe_is_a_field_node() -> None:
    assert isinstance(_depth(), FieldNode)
    assert is_field_node(_depth())
    assert not is_field_node(np.zeros(3))
    assert not is_field_node({"value": np.zeros(3)})


def test_a_tree_without_field_nodes_passes() -> None:
    refuse_field_nodes({"image": np.zeros(2), "meta": {"id": np.int32(1)}}, "Op reads arrays")


def test_the_refusal_names_each_node_and_gives_its_fix() -> None:
    data = {"depth": _depth(), "meta": {"mask": _depth()}, "image": np.zeros(2)}
    with pytest.raises(TypeError) as caught:
        refuse_field_nodes(data, "MapOperator maps fn over the leaves it selects")
    assert str(caught.value) == (
        "MapOperator maps fn over the leaves it selects, and 'depth', 'meta/mask' is a Maybe "
        "(a value that can be missing). Fill it first (value_or) or apply an operator that reads "
        "its presence: read it with .value_or(fill), or with .present in hand."
    )


def test_a_subtree_is_named_from_the_record_root() -> None:
    """A consumer checking one field passes its key tuple, so the label is the field's own."""
    with pytest.raises(TypeError, match=r"Op reads its audio as an array, and 'audio' is a Maybe"):
        refuse_field_nodes(_depth(), "Op reads its audio as an array", where=("audio",))


def test_a_node_inside_a_named_subtree_is_labelled_from_the_root() -> None:
    with pytest.raises(TypeError, match=r"'meta/mask' is a Maybe"):
        refuse_field_nodes({"mask": _depth()}, "Op", where=("meta",))
