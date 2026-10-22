"""A configuration's field-naming parameters are converted once, by their annotation.

A parameter annotated ``FieldName`` names one field: a spelling, or a list or tuple of strings
that becomes one key tuple. A parameter annotated ``FieldNames`` names several: a list or tuple
whose elements are spellings or lists of strings, each list a key tuple. TOML, Python tuples and
Python lists build equal, hashable configurations.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

import pytest

from datarax.core.config import DataraxModuleConfig
from datarax.core.field_paths import FieldName, FieldNames
from datarax.core.modality import ModalityOperatorConfig


@dataclass(frozen=True)
class _Naming(DataraxModuleConfig):
    target: FieldName = "out"
    sources: FieldNames | None = None


def test_a_list_for_one_field_becomes_one_key_tuple() -> None:
    assert _Naming(target=["meta", "score"]).target == ("meta", "score")


def test_a_list_for_several_fields_becomes_a_tuple_of_names() -> None:
    config = _Naming(sources=["image", ["meta", "score"]])
    assert config.sources == ("image", ("meta", "score"))


def test_toml_tuples_and_lists_build_equal_hashable_configs() -> None:
    table = tomllib.loads('target = ["meta", "score"]\nsources = ["image", ["meta", "id"]]')
    from_toml = _Naming(**table)
    from_tuples = _Naming(target=("meta", "score"), sources=("image", ("meta", "id")))
    from_lists = _Naming(target=["meta", "score"], sources=["image", ["meta", "id"]])
    assert from_toml == from_tuples == from_lists
    assert hash(from_toml) == hash(from_tuples) == hash(from_lists)


def test_a_bare_string_for_several_fields_is_refused_naming_the_parameter() -> None:
    with pytest.raises(TypeError, match="sources names several fields"):
        _Naming(sources="image")


def test_an_unset_optional_parameter_stays_none() -> None:
    assert _Naming().sources is None


def test_the_modality_configs_take_key_tuples() -> None:
    config = ModalityOperatorConfig(field_key=["meta", "image"], target_key="meta/out")
    assert config.field_key == ("meta", "image")
    assert config.target_key == "meta/out"
    assert ModalityOperatorConfig(
        field_key="x",
        auxiliary_fields=["mask", ["meta", "box"]],
    ).auxiliary_fields == ("mask", ("meta", "box"))
