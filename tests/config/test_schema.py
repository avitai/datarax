"""What ``ConfigSchema.validate`` accepts, refuses and returns.

A schema is a subclass with ``SchemaField`` class attributes. Its fields include every base
class's, a subclass redefining a field wins, and the defaults a validation returns are its own
copies. Types follow what TOML produces: an integer where a float is expected is a float, and
``True`` is not an integer.
"""

from __future__ import annotations

import pytest

from datarax.config import ConfigSchema, SchemaField, ValidationError


class Training(ConfigSchema):
    """A small schema with a required, a defaulted and a mutable-default field."""

    batch_size: SchemaField = SchemaField(int)
    learning_rate: SchemaField = SchemaField(float, required=False, default=1e-3)
    streams: SchemaField = SchemaField(dict, required=False, default={"default": 0})


class FineTuning(Training):
    """Adds a field and redefines a default."""

    freeze_backbone: SchemaField = SchemaField(bool, required=False, default=True)
    learning_rate: SchemaField = SchemaField(float, required=False, default=1e-4)


class Aliased(Training):
    """Adds nothing: its fields are exactly its base's."""


class Optimizer(ConfigSchema):
    """A nested schema."""

    name: SchemaField = SchemaField(str)
    momentum: SchemaField = SchemaField(float, required=False, default=0.9)


class Experiment(ConfigSchema):
    """A schema with a nested one."""

    optimizer: SchemaField = SchemaField(Optimizer)


def test_fields_come_from_every_base_and_a_subclass_redefinition_wins() -> None:
    assert set(FineTuning.get_schema_fields()) == {
        "batch_size",
        "learning_rate",
        "streams",
        "freeze_backbone",
    }
    assert FineTuning.validate({"batch_size": 8})["learning_rate"] == 1e-4
    assert Training.validate({"batch_size": 8})["learning_rate"] == 1e-3


def test_a_subclass_with_no_fields_of_its_own_has_its_bases() -> None:
    assert set(Aliased.get_schema_fields()) == set(Training.get_schema_fields())
    assert Aliased.validate({"batch_size": 8})["batch_size"] == 8


def test_each_validation_returns_its_own_copy_of_a_default() -> None:
    first = Training.validate({"batch_size": 8})
    first["streams"]["augment"] = 1

    assert Training.validate({"batch_size": 8})["streams"] == {"default": 0}


def test_an_integer_where_a_float_is_expected_is_a_float() -> None:
    validated = Training.validate({"batch_size": 8, "learning_rate": 1})

    assert validated["learning_rate"] == 1.0
    assert isinstance(validated["learning_rate"], float)


@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_is_not_an_integer(value: bool) -> None:
    with pytest.raises(ValidationError, match="batch_size"):
        Training.validate({"batch_size": value})


def test_a_boolean_is_not_a_float() -> None:
    with pytest.raises(ValidationError, match="learning_rate"):
        Training.validate({"batch_size": 8, "learning_rate": True})


def test_a_nested_schema_is_validated_with_its_defaults() -> None:
    assert Experiment.validate({"optimizer": {"name": "sgd"}}) == {
        "optimizer": {"name": "sgd", "momentum": 0.9}
    }


def test_a_nested_failure_names_the_path_and_the_inner_reason() -> None:
    with pytest.raises(ValidationError, match=r"optimizer.*Required field 'name' is missing"):
        Experiment.validate({"optimizer": {"momentum": 0.5}})


def test_missing_and_unknown_fields_are_refused_by_name() -> None:
    with pytest.raises(ValidationError, match="Required field 'batch_size' is missing"):
        Training.validate({})
    with pytest.raises(ValidationError, match="Unknown fields in configuration: epochs, seed"):
        Training.validate({"batch_size": 8, "seed": 0, "epochs": 2})


def test_a_custom_validator_refuses_by_name() -> None:
    class Positive(ConfigSchema):
        batch_size: SchemaField = SchemaField(int, validator=lambda value: value > 0)

    assert Positive.validate({"batch_size": 1}) == {"batch_size": 1}
    with pytest.raises(ValidationError, match="batch_size"):
        Positive.validate({"batch_size": 0})


def test_an_absent_optional_nested_schema_gets_its_defaults() -> None:
    class WithOptimizer(ConfigSchema):
        optimizer: SchemaField = SchemaField(Optimizer, required=False, default={"name": "adam"})

    assert WithOptimizer.validate({}) == {"optimizer": {"name": "adam", "momentum": 0.9}}
