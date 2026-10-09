"""Building a TFDS source mid-run imports nothing: TFDS is imported where the source is.

The first ``import tensorflow_datasets`` of a process enters ``etils.epy.lazy_imports()``, which
swaps ``builtins.__import__`` for the whole process while it runs. A garbage collection inside
that window runs finalizers whose imports then fail: a finished Grain ``mp_prefetch`` iterator
left in a reference cycle (google/grain#1420) holds named semaphores, and each one's finalizer
fails with ``ValueError: Relative import statements not supported``, so the semaphore stays
linked in ``/dev/shm`` and the error is reported as unraisable.

``datarax.sources.tfds_source`` is TFDS's integration module and imports it at its top, so the
window opens on the line that imports a TFDS source, never inside a constructor or ``from_tfds``
in the middle of a run. ``import datarax.sources`` still imports no TFDS: those names are
exported lazily. Each check runs in a fresh interpreter, since the test process has imported
TFDS long before. The window is made to matter: a full collection is forced on the first import
made under etils' swap, as the automatic collection in the suite did by chance. The positive
control runs the same program with a first TFDS import of its own mid-run, and sees the window
and the leak.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from substrax.testing import run_python

from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


pytestmark = pytest.mark.tfds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_SECONDS = 300.0
_CONTROL = "a first import of tensorflow_datasets"

# argv: the entry under test, the ArrayRecord data directory, the TFRecord data directory.
# Prints one JSON line: what was imported after each stage, every module that imported under
# etils' swap while the source was built, and the Grain semaphores still linked after collection.
_BUILD_MID_RUN = """
import gc, json, multiprocessing.synchronize, os, sys

from absl import app
import grain

entry, array_record, tfrecord = sys.argv[1:4]
seen = {}
import datarax.sources
seen["import datarax.sources"] = "tensorflow_datasets" in sys.modules
if entry == "TFDSStreamingSource":
    from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource
    build = lambda: TFDSStreamingSource(
        TFDSStreamingConfig(name=NAME, split="train", data_dir=tfrecord)
    )
elif entry == "TFDSEagerSource":
    from datarax.sources import TFDSEagerConfig, TFDSEagerSource
    build = lambda: TFDSEagerSource(
        TFDSEagerConfig(name=NAME, split="train", data_dir=array_record)
    )
elif entry == "from_tfds":
    from datarax.sources import from_tfds
    build = lambda: from_tfds(NAME, "train", data_dir=tfrecord)
else:
    build = lambda: __import__("tensorflow_datasets")
seen["import the entry"] = "tensorflow_datasets" in sys.modules


def main(argv):
    del argv
    # Collections happen only where this program makes them: one on the first import made
    # under etils' swap, one at the end. An automatic one could free the garbage before or
    # after the window and decide the result by timing.
    gc.disable()
    # A finished worker-process read, left in a reference cycle with its semaphores.
    dataset = grain.MapDataset.range(8).to_iter_dataset().mp_prefetch(
        grain.MultiprocessingOptions(num_workers=2)
    )
    iterator = iter(dataset)
    for _ in iterator:
        pass
    del iterator, dataset
    names = [
        o._semlock.name
        for o in gc.get_objects()
        if isinstance(o, multiprocessing.synchronize.SemLock) and o._semlock.name is not None
    ]
    swapped = []

    def watch(frame, event, arg):
        # etils' swapped __import__ calls _lazy_import, and only inside its window.
        if event == "call" and frame.f_code.co_name == "_lazy_import":
            swapped.append(frame.f_back.f_code.co_filename)
            if len(swapped) == 1:
                gc.collect()

    sys.setprofile(watch)
    build()
    sys.setprofile(None)
    gc.collect()
    linked = [n for n in names if os.path.exists("/dev/shm/sem." + n.lstrip("/"))]
    print(json.dumps({
        "seen": seen,
        "imported_under_swap": sorted(set(swapped)),
        "semaphores": len(names),
        "still_linked": len(linked),
    }))


app.run(main, argv=sys.argv[:1])
"""


def _build_mid_run(entry: str, fixture: TFDSFixture) -> tuple[dict[str, Any], str]:
    result = run_python(
        _BUILD_MID_RUN.replace("NAME", repr(FIXTURE)),
        entry,
        str(fixture.array_record),
        str(fixture.tfrecord),
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    report = json.loads(result.check().stdout.strip().splitlines()[-1])
    return report, result.stderr


@pytest.mark.parametrize("entry", ["TFDSStreamingSource", "TFDSEagerSource", "from_tfds"])
def test_building_a_tfds_source_mid_run_opens_no_import_window(
    entry: str, tfds_fixture: TFDSFixture
) -> None:
    report, stderr = _build_mid_run(entry, tfds_fixture)

    assert report["semaphores"] > 0  # there was Grain garbage for a window to catch
    assert report["imported_under_swap"] == []
    assert report["still_linked"] == 0
    assert "Exception ignored" not in stderr
    assert report["seen"] == {"import datarax.sources": False, "import the entry": True}


def test_the_check_sees_the_window_and_the_leak_of_a_first_import_mid_run(
    tfds_fixture: TFDSFixture,
) -> None:
    """The positive control: the program's own first TFDS import, made mid-run."""
    report, stderr = _build_mid_run(_CONTROL, tfds_fixture)

    assert report["seen"] == {"import datarax.sources": False, "import the entry": False}
    assert any(
        path.endswith("tensorflow_datasets/__init__.py") for path in report["imported_under_swap"]
    )
    assert report["semaphores"] > 0
    assert report["still_linked"] == report["semaphores"]
    assert "Relative import statements not supported" in stderr
