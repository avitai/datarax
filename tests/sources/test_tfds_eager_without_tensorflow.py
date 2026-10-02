"""The eager TFDS source keeps TensorFlow out of the process and the records off the device.

TensorFlow in a JAX process breaks JAX's NCCL collectives, and on a GPU machine it takes most of
the card's memory, so the process that trains never imports it. The test process itself imports
TensorFlow (the suite's conftest configures it when installed), so each check here runs in a
fresh interpreter, beside a positive control showing the same check sees TensorFlow when it is
imported. Building the source places nothing on a device: the host columns are NumPy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import run_python

from datarax.sources import TFDSEagerConfig, TFDSEagerSource
from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture, TRAIN_RECORDS


pytestmark = pytest.mark.tfds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_SECONDS = 300.0

# Each stage prints whether TensorFlow is in the process after it, as one JSON line at the end.
_READ_WITHOUT_TENSORFLOW = """
import json, sys
import numpy as np
from flax import nnx

seen = {}
import datarax.sources
seen["import datarax.sources"] = "tensorflow" in sys.modules
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

array_record, tfrecord = sys.argv[1], sys.argv[2]
source = TFDSEagerSource(TFDSEagerConfig(name=NAME, split="train", data_dir=array_record))
seen["construct"] = "tensorflow" in sys.modules
records = list(source)
seen["iterate"] = "tensorflow" in sys.modules
batch = source.get_batch(to_words(np.arange(4, dtype=np.uint64)))
seen["get_batch"] = "tensorflow" in sys.modules
pipeline = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0), shuffle=True)
step = pipeline.step()
seen["pipeline step"] = "tensorflow" in sys.modules
try:
    TFDSEagerSource(TFDSEagerConfig(name=NAME, split="train", data_dir=tfrecord))
except FileNotFoundError:
    seen["refuse a tfrecord copy"] = "tensorflow" in sys.modules
print(json.dumps({"records": len(records), "batch": int(batch["image"].shape[0]), "seen": seen}))
"""


def _child_env() -> dict[str, str]:
    return forwarded_jax_environment(os.environ)


def test_reading_iterating_and_stepping_never_import_tensorflow(
    tfds_fixture: TFDSFixture,
) -> None:
    result = run_python(
        _READ_WITHOUT_TENSORFLOW.replace("NAME", repr(FIXTURE)),
        str(tfds_fixture.array_record),
        str(tfds_fixture.tfrecord),
        timeout=_CHILD_SECONDS,
        env=_child_env(),
        cwd=_REPO_ROOT,
    )

    report = json.loads(result.check().stdout.strip().splitlines()[-1])
    assert report["records"] == TRAIN_RECORDS
    assert report["batch"] == 4
    assert report["seen"] == {
        "import datarax.sources": False,
        "construct": False,
        "iterate": False,
        "get_batch": False,
        "pipeline step": False,
        "refuse a tfrecord copy": False,
    }


def test_the_check_sees_tensorflow_when_a_process_imports_it() -> None:
    """The positive control: the same check in a fresh interpreter that imports TensorFlow."""
    result = run_python(
        "import json, sys; import tensorflow; print(json.dumps('tensorflow' in sys.modules))",
        timeout=_CHILD_SECONDS,
        env={**_child_env(), "CUDA_VISIBLE_DEVICES": ""},
    )

    assert result.check().last_json() is True


def _dataset_sized_device_arrays() -> set[int]:
    return {id(a) for a in jax.live_arrays() if a.ndim and a.shape[0] == TRAIN_RECORDS}


def test_building_the_source_places_nothing_on_a_device(tfds_fixture: TFDSFixture) -> None:
    config = TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.array_record))
    control = jnp.zeros((TRAIN_RECORDS, 2))  # the instrument finds a dataset-sized device array
    assert id(control) in _dataset_sized_device_arrays()
    with jax.transfer_guard("disallow_explicit"):
        with pytest.raises(RuntimeError, match="[Dd]isallowed"):
            jax.device_put(np.zeros(3))  # the guard fires on an upload here
    before = _dataset_sized_device_arrays()

    with jax.transfer_guard("disallow_explicit"):
        source = TFDSEagerSource(config)

    assert _dataset_sized_device_arrays() - before == set()
    assert len(source) == TRAIN_RECORDS


_PEAK_DEVICE_BYTES = """
import json, sys
import jax
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

device = jax.devices()[0]
before = device.memory_stats()["peak_bytes_in_use"]
source = TFDSEagerSource(TFDSEagerConfig(name=NAME, split="train", data_dir=sys.argv[1]))
after = device.memory_stats()["peak_bytes_in_use"]
print(json.dumps({"platform": device.platform, "before": before, "after": after}))
"""


@pytest.mark.accelerator(kind="gpu")
def test_building_the_source_raises_no_device_memory_peak(tfds_fixture: TFDSFixture) -> None:
    """On a GPU the device's peak memory is the same before and after construction."""
    result = run_python(
        _PEAK_DEVICE_BYTES.replace("NAME", repr(FIXTURE)),
        str(tfds_fixture.array_record),
        timeout=_CHILD_SECONDS,
        env={"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        cwd=_REPO_ROOT,
    )

    report = json.loads(result.check().stdout.strip().splitlines()[-1])
    assert report["platform"] == "gpu"
    assert report["after"] == report["before"]
