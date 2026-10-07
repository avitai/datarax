"""The host stage stages up to ``d`` placed batches on the device ahead of the consumer.

With a device buffer of ``d`` batches the placement runs on a thread of its own, Grain's
``experimental.device_put`` composition (a host read buffer, ``map(jax.device_put)``, a thread
prefetch of ``d``); ``d = 0`` places each batch on the consumer as it is taken. Whatever ``d``,
the cursor names only the batches delivered, a read or placement error ends the run at them,
``close()``/``reset()``/``set_state()`` release the batches placed ahead and the thread, and the
placement stays explicit, uncommitted and in the caller's precision mode.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import run_python

from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.streams import RecordStream
from tests.test_common.transfers import implicit_upload_raises


_DEPTHS = [0, 1, 2, 3]
_IMAGE = (5, 5, 3)  # a shape no other array of these tests has, to find placed batches


def _columns(length: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(length)
    return {
        "image": rng.integers(0, 256, (length, *_IMAGE), dtype=np.uint8),
        "label": np.arange(length, dtype=np.int32),
    }


def _memory(length: int = 40) -> MemorySource:
    return MemorySource(MemorySourceConfig(), _columns(length))


def _pipeline(
    source: DataSourceModule,
    depth: int,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    num_epochs: int | None = 2,
    threads: int = 1,
) -> Pipeline:
    pipe = Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(3),
        shuffle=shuffle,
        num_epochs=num_epochs,
    )
    pipe.host_stage._device_buffer = depth
    pipe.host_stage._read_threads = threads
    pipe.host_stage._read_buffer = max(2, threads)
    return pipe


def _names(batch: Batch) -> list[int]:
    return [int(v) for v in from_words(np.asarray(batch.indices))]


def _placed_batches() -> int:
    """Live arrays holding a placed batch's images."""
    return sum(1 for array in jax.live_arrays() if array.shape[1:] == _IMAGE)


def _grain_threads() -> list[str]:
    """The run's read threads (Grain's) and its placement thread."""
    return [
        t.name
        for t in threading.enumerate()
        if "grain" in t.name.lower() or t.name == "datarax-device-staging"
    ]


class _Placements:
    """Counts the batches placed, wherever the placement runs, and lets a test wait for them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        self.threads: set[str] = set()
        self._changed = threading.Condition()
        put = jax.device_put

        def counted(value: Any, *args: Any, **kwargs: Any) -> Any:
            placed = put(value, *args, **kwargs)
            if isinstance(value, Batch):
                with self._changed:
                    self.count += 1
                    self.threads.add(threading.current_thread().name)
                    self._changed.notify_all()
            return placed

        monkeypatch.setattr(jax, "device_put", counted)

    def wait_for(self, count: int) -> int:
        with self._changed:
            self._changed.wait_for(lambda: self.count >= count, timeout=30)
            return self.count


@pytest.mark.parametrize("depth", _DEPTHS)
def test_the_device_holds_exactly_the_depth_ahead_of_the_batch_taken(
    depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``d`` placed batches wait on the device beside the one the consumer holds, never more.

    Counted twice: the placements (a spy on ``jax.device_put``) and the live device arrays of
    a batch's shape, after each ``next()`` once the placement thread has filled its room. ``d = 0``
    places each batch on the consumer as it is taken.
    """
    gc.collect()
    assert _placed_batches() == 0
    placements = _Placements(monkeypatch)
    pipe = _pipeline(_memory(), depth, num_epochs=None)
    batches = iter(pipe.raw_batches())
    held = None
    for taken in range(1, 5):
        held = next(batches)
        placements.wait_for(taken + depth)
        threading.Event().wait(0.3)  # time for a placement past the room, were there one
        assert placements.count - taken == depth
        assert _placed_batches() == 1 + depth
    consumer = threading.current_thread().name
    assert (consumer in placements.threads) == (depth == 0), placements.threads
    del held
    pipe.close()


@pytest.mark.parametrize("threads", [1, 4, 8])
@pytest.mark.parametrize("depth", _DEPTHS)
def test_resume_after_any_batch_is_exact_whatever_is_placed_ahead(depth: int, threads: int) -> None:
    def build() -> Pipeline:
        return _pipeline(_memory(23), depth, num_epochs=3, threads=threads)

    whole = [_names(b) for b in build().raw_batches()]
    for done in range(len(whole) + 1):
        pipe = build()
        batches = iter(pipe.raw_batches())
        served = [_names(next(batches)) for _ in range(done)]
        state = pipe.get_state()
        pipe.close()
        resumed = build()
        resumed.set_state(state)
        assert served + [_names(b) for b in resumed.raw_batches()] == whole


class _Failed(Exception):
    pass


@pytest.mark.parametrize("depth", _DEPTHS)
@pytest.mark.parametrize("fails", ["read", "placement", "stream read"])
def test_an_error_ends_the_run_at_the_batches_delivered(
    depth: int, fails: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build() -> Pipeline:
        if fails == "stream read":
            stream = RecordStream(_columns(40), kind=RecordIdentity.ARRIVAL, chunk=4)
            return _pipeline(stream, depth, shuffle=False, num_epochs=1)
        return _pipeline(_memory(), depth, shuffle=False, num_epochs=1)

    whole = [_names(b) for b in build().raw_batches()]
    calls = [0]
    if fails == "placement":
        put = jax.device_put

        def flaky(value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, Batch):
                calls[0] += 1
                if calls[0] == 4:
                    raise _Failed("placement 4 failed")
            return put(value, *args, **kwargs)

        monkeypatch.setattr(jax, "device_put", flaky)
    else:
        owner: Any = MemorySource if fails == "read" else RecordStream
        name = "get_batch" if fails == "read" else "read_from"
        original = getattr(owner, name)

        def flaky_read(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls[0] += 1
            if calls[0] == 4:
                raise _Failed("read 4 failed")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(owner, name, flaky_read)
    pipe = build()
    served: list[list[int]] = []
    with pytest.raises(_Failed):
        for batch in pipe.raw_batches():
            served.append(_names(batch))
    assert served == whole[: len(served)]
    state = pipe.get_state()
    delivered = 4 * len(served)
    assert (state["position"] if fails != "stream read" else state["stream"]["records"]) == (
        delivered
    )
    served += [_names(b) for b in pipe.raw_batches()]
    assert served == whole


@pytest.mark.parametrize("depth", _DEPTHS)
@pytest.mark.parametrize("ending", ["close", "reset", "set_state", "exhausted", "dropped"])
def test_the_batches_placed_ahead_and_the_thread_are_released(depth: int, ending: str) -> None:
    gc.collect()
    assert _placed_batches() == 0
    pipe = _pipeline(_memory(), depth, num_epochs=1)
    state = pipe.get_state()
    batches = iter(pipe.raw_batches())
    if ending == "exhausted":
        list(batches)
    else:
        next(batches)
        threading.Event().wait(0.3)  # time for the buffer to fill
        if ending == "close":
            pipe.close()
        elif ending == "reset":
            pipe.reset()
        elif ending == "set_state":
            pipe.set_state(state)
        else:
            del pipe
    del batches
    for _ in range(100):
        gc.collect()
        if not _grain_threads() and _placed_batches() == 0:
            break
        threading.Event().wait(0.05)
    assert _grain_threads() == []
    assert _placed_batches() == 0


@pytest.mark.parametrize("depth", [1, 3])
def test_no_exception_is_ignored_at_exit(depth: int) -> None:
    code = (
        "import numpy as np\n"
        "from flax import nnx\n"
        "from datarax.pipeline import Pipeline\n"
        "from datarax.sources.memory_source import MemorySource, MemorySourceConfig\n"
        "source = MemorySource(MemorySourceConfig(), {'x': np.arange(40.0)})\n"
        "pipe = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0),"
        " shuffle=True, num_epochs=None)\n"
        f"pipe.host_stage._device_buffer = {depth}\n"
        "batches = pipe.raw_batches()\n"
        "next(iter(batches))\n"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, no shell
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "Exception ignored" not in result.stderr


@pytest.mark.parametrize("depth", _DEPTHS)
def test_placement_stays_explicit_uncommitted_and_in_the_caller_s_mode(depth: int) -> None:
    assert implicit_upload_raises(), "the guard must fire on an implicit upload"
    pipe = _pipeline(_memory(), depth)
    with jax.transfer_guard("disallow"):
        served = list(pipe.raw_batches())
    assert served and not served[0]["image"].committed
    with jax.enable_x64(True):
        wide = MemorySource(MemorySourceConfig(), {"x": np.ones((8, 3), np.float64)})
        batches = list(_pipeline(wide, depth, batch_size=8, num_epochs=1).raw_batches())
    assert [b["x"].dtype for b in batches] == [jnp.float64]


_PEAK = """
import collections, json, sys, threading
import jax
import numpy as np
from flax import nnx
from datarax.core.index_words import to_words
from datarax.core.prng import key_words
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig

mode, records, held, depth = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(0)
source = MemorySource(
    MemorySourceConfig(),
    {
        "image": rng.integers(0, 256, (records, 32, 32, 3), dtype=np.uint8),
        "label": np.arange(records, dtype=np.int32),
    },
)
if mode == "pipeline":
    pipe = Pipeline(source=source, stages=[], batch_size=256, rngs=nnx.Rngs(0), shuffle=True)
    pipe.host_stage._device_buffer = depth
    key_words(pipe._epoch_key_base.get_value())  # the run reads its key once, before the snapshot
    batches = pipe.raw_batches()
else:  # the control: each host batch placed by hand, ``held`` kept
    rows = [np.arange(k * 256, (k + 1) * 256, dtype=np.uint64) for k in range(records // 256)]
    batches = (jax.device_put(source.get_batch(to_words(r))) for r in rows)
device = jax.devices()[0]
before = device.memory_stats()["bytes_in_use"]
kept = collections.deque(maxlen=held)
for batch in batches:
    kept.append(batch)
    if mode == "pipeline":
        threading.Event().wait(0.02)  # a step's time, in which the placement thread fills its room
growth = device.memory_stats()["peak_bytes_in_use"] - before
print(json.dumps({"platform": device.platform, "growth": growth}))
"""


@pytest.mark.accelerator(kind="gpu")
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_on_a_gpu_the_peak_is_the_control_s_with_the_depth_more_batches(depth: int) -> None:
    """Each variant in its own process (a device's peak cannot be reset in one).

    The consumer keeps its last ``held`` batches and takes the next after a step's time; the
    device then holds ``held + d`` batches. Placing by hand while keeping ``k`` peaks at ``k + 1``,
    the new batch placed beside the kept ones. Staged, the placement thread refills its room as
    the consumer takes a batch, before or after the consumer drops its oldest (a race), so the
    peak lies between the controls keeping ``held + d - 1`` and ``held + d``: the by-hand control
    plus at most ``d`` batches, whatever the dataset's size (measured on an RTX 4090).
    """
    held = 2

    def growth(mode: str, records: int, kept: int) -> int:
        result = run_python(
            _PEAK,
            mode,
            str(records),
            str(kept),
            str(depth),
            timeout=600,
            # The order is named on the CPU device, so the CPU platform runs beside the GPU.
            env={"JAX_PLATFORMS": "cuda,cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
            cwd=Path(__file__).resolve().parents[2],
        )
        report = json.loads(result.check().stdout.strip().splitlines()[-1])
        assert report["platform"] == "gpu"
        return int(report["growth"])

    lower, upper = growth("by hand", 2560, held + depth - 2), growth("by hand", 2560, held + depth)
    for records in (2560, 25600):
        assert lower < growth("pipeline", records, held) <= upper
