"""A record's fields named by their path: the one naming every source and its refusals share.

A field is a leaf of a record's pytree, named by its path's steps joined by ``/``: a dictionary
key by the key itself, a list position as JAX prints it (``[0]``). Provenance keys, the field names
a refusal prints and the columns a source holds on disk are all named this way.
"""

from __future__ import annotations

import numpy as np
import pytest

from datarax.sources import _field_paths
from datarax.sources._field_paths import field_name, field_names


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


@pytest.mark.parametrize("name", ["field_name", "field_names"])
def test_the_module_is_the_one_home_of_field_naming(name: str) -> None:
    """``eager_source`` reads these names; no second copy of the joiner is kept there."""
    from datarax.sources import eager_source  # noqa: PLC0415 - the module under test

    assert getattr(eager_source, name) is getattr(_field_paths, name)
    assert f"_{name}" not in vars(eager_source)
