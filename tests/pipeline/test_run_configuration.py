"""The configuration a pipeline's saved state is valid for, and its refusal of any other.

``Pipeline.get_state()`` records the configuration that produced it under ``"fingerprint"``:
the batch rule, the length, the epochs, the key and the order. ``set_state`` refuses a state of
another configuration naming the first field that differs, and one holding none. A checkpoint
store may hand the state back with NumPy leaves, which read as plain values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.pipeline.run_configuration import (
    plain_state,
    refuse_another_configuration,
    run_configuration,
    SESSION_VERSION,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _pipeline(**options: Any) -> Pipeline:
    settings: dict[str, Any] = {"batch_size": 4, "shuffle": True, "num_epochs": 2} | options
    return Pipeline(
        source=MemorySource(MemorySourceConfig(), {"x": np.arange(10, dtype=np.int32)}),
        stages=[],
        rngs=nnx.Rngs(3),
        **settings,
    )


def test_the_configuration_names_the_batch_rule_length_epochs_key_and_order() -> None:
    pipe = _pipeline()
    configuration = run_configuration(pipe)
    assert configuration == {
        "batch_size": 4,
        "length": 10,
        "drop_last": False,
        "num_epochs": 2,
        "shuffled": True,
        "seed": configuration["seed"],
        "order": {"kind": "global"},
    }
    assert configuration["seed"] and all(isinstance(word, int) for word in configuration["seed"])
    assert configuration == pipe.get_state()["fingerprint"]


def test_the_pipeline_s_own_configuration_is_accepted() -> None:
    pipe = _pipeline()
    refuse_another_configuration(pipe, run_configuration(_pipeline()))


@pytest.mark.parametrize(
    ("field", "options"),
    [
        ("batch_size", {"batch_size": 5}),
        ("num_epochs", {"num_epochs": 3}),
        ("shuffled", {"shuffle": False}),
    ],
)
def test_another_configuration_is_refused_naming_the_field(
    field: str, options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match=rf"produced with {field}="):
        refuse_another_configuration(_pipeline(), run_configuration(_pipeline(**options)))


def test_a_state_holding_no_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no fingerprint"):
        refuse_another_configuration(_pipeline(), None)


def test_numpy_leaves_read_as_plain_values() -> None:
    saved = {
        "a": np.int64(3),
        "b": np.asarray([1, 2], np.uint32),
        "c": (np.float32(0.5), [np.bool_(True)]),
    }
    plain = plain_state(saved)
    assert plain == {"a": 3, "b": [1, 2], "c": [0.5, [True]]}
    assert type(plain["a"]) is int and type(plain["c"][1][0]) is bool


def test_the_session_layout_s_version() -> None:
    assert SESSION_VERSION == 2
