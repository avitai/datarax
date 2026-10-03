"""TFDS records are decoded on the protobuf runtime a plain install selects, ``upb``.

Each TFDS record is a serialized ``tf.Example``, parsed by protobuf before TFDS's NumPy decode, so
the runtime prices every record: on CIFAR-10 the pure-Python runtime costs 132 us per record
against 71 us on ``upb``. Neither datarax nor its test suite chooses a runtime; the variable that
would (:data:`~tests.test_common.protobuf_runtime.RUNTIME_VARIABLE`) is left to whoever runs the
process, and a tooling contract keeps every tracked file from naming it
(``tests/test_tooling_contracts.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from google.protobuf.internal import api_implementation
from substrax.testing import run_python

from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common.protobuf_runtime import RUNTIME_VARIABLE
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture, TRAIN_RECORDS


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_SECONDS = 300.0

# Starts from the default runtime (the variable unset), reads both TFDS sources, and reports the
# runtime that parsed the records and whether anything set the variable on the way.
_READ_ON_THE_DEFAULT_RUNTIME = """
import json, os, sys
os.environ.pop(VARIABLE, None)
import jax
import datarax
from datarax.sources import (
    TFDSEagerConfig, TFDSEagerSource, TFDSStreamingConfig, TFDSStreamingSource,
)
from google.protobuf.internal import api_implementation

array_record, tfrecord = sys.argv[1], sys.argv[2]
eager = TFDSEagerSource(TFDSEagerConfig(name=NAME, split="train", data_dir=array_record))
stream = TFDSStreamingSource(TFDSStreamingConfig(name=NAME, split="train", data_dir=tfrecord))
streamed = stream.get_batch(4, key=jax.random.key(0)).batch_size
print(json.dumps({
    "datarax": datarax.__file__,
    "runtime": api_implementation.Type(),
    "variable": os.environ.get(VARIABLE),
    "eager records": len(eager),
    "streamed": streamed,
}))
"""


def test_the_suite_runs_on_the_runtime_a_plain_install_selects() -> None:
    """The test process parses protobuf on ``upb``, as a user's process does."""
    assert api_implementation.Type() == "upb", (
        f"{RUNTIME_VARIABLE}={os.environ.get(RUNTIME_VARIABLE)!r} in this process; a shell "
        "activated with an older .datarax.env still exports it: rerun ./setup.sh"
    )


@pytest.mark.tfds
def test_reading_tfds_leaves_the_default_runtime_in_place(tfds_fixture: TFDSFixture) -> None:
    result = run_python(
        _READ_ON_THE_DEFAULT_RUNTIME.replace("VARIABLE", repr(RUNTIME_VARIABLE)).replace(
            "NAME", repr(FIXTURE)
        ),
        str(tfds_fixture.array_record),
        str(tfds_fixture.tfrecord),
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )

    report = json.loads(result.check().stdout.strip().splitlines()[-1])
    assert report["eager records"] == TRAIN_RECORDS
    assert report["streamed"] == 4
    assert report["variable"] is None
    assert report["runtime"] == "upb"
