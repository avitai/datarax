"""A pytest plugin: every test ends with the threads it started, and lists them if it does not.

Work a test leaves running on another thread runs inside the tests after it. A pipeline dropped
mid-epoch closes its run at once, but Grain's read pool finishes the reads it had started (its
``close()`` cancels only reads that have not started), and each read names its records with a
compiled program on the CPU, making device arrays a later test's live-array check reports. A
thread that never ends runs inside every later test.

After each test the plugin waits, up to :data:`BOUND` seconds in all, for the threads the test
started, then fails the test if one of them is still running, naming it and what it runs. The
wait lets a read the test started finish inside the test that started it.

Two library threads live as long as the process by design, so the plugin settles them before the
first test rather than blame the first test that needs them:

- tqdm's progress-bar monitor (one thread, waking every 10 seconds to retune bars) is turned off
  with tqdm's own switch, ``monitor_interval = 0``. It only adjusts how often a bar redraws, and
  a process-wide count of Python calls would otherwise include its wakes. tqdm comes only with
  the data extras; where it is not installed no monitor can start and there is nothing to set.
- Grain's pool that releases shared memory in the background is made with the call Grain makes
  at its first multiprocess read, ``SharedMemoryArray.enable_async_del(1)``. The first call fixes
  the pool for the process, so the pool is the one Grain makes either way.

The root ``conftest.py`` registers the plugin for ``tests/`` and ``benchmarks/tests/``. It loads
before ``tests/conftest.py`` resolves the JAX test environment (the emulated CPU devices among it)
and imports jax, so nothing this module imports at load may import jax: Grain is imported in
``pytest_configure``, which runs once the initial conftests are loaded, as substrax's plugin does,
and the module sits in ``tests`` rather than ``tests.test_common``, whose package imports jax.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Iterator

import pytest


BOUND = 2.0
"""Seconds the check waits, in all, for a test's threads. Grain's read-pool tails take 5-125 ms."""


def pytest_configure(config: pytest.Config) -> None:
    """Settle the library threads that live as long as the process, before any test.

    Args:
        config: The pytest configuration (unused).
    """
    del config
    try:
        import tqdm  # noqa: PLC0415 - installed only with the data extras
    except ModuleNotFoundError:
        pass  # no tqdm, so no monitor thread to turn off
    else:
        tqdm.tqdm.monitor_interval = 0
    # Imported here, after tests/conftest.py set the JAX environment: Grain imports jax.
    from grain._src.python.ipc.shared_memory_array import SharedMemoryArray  # noqa: PLC0415

    SharedMemoryArray.enable_async_del(1)


def threads_left(before: Collection[threading.Thread], *, within: float) -> list[threading.Thread]:
    """The threads started since ``before`` that are still running after waiting ``within``.

    Args:
        before: The threads that were running when the wait's window opened.
        within: Seconds to wait, in all, for the threads started since to end.

    Returns:
        The threads started since ``before`` still running, in ``threading.enumerate()`` order.
    """
    started = [thread for thread in threading.enumerate() if thread not in before]
    deadline = time.monotonic() + within
    for thread in started:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [thread for thread in started if thread.is_alive()]


def grain_thread_names() -> list[str]:
    """The names of the running threads Grain started (its read, prefetch and process threads).

    Returns:
        The names that contain ``grain``, in ``threading.enumerate()`` order.
    """
    return [thread.name for thread in threading.enumerate() if "grain" in thread.name.lower()]


def _describe(thread: threading.Thread) -> str:
    target = getattr(thread, "_target", None)  # set by Thread.__init__; None for a subclass's run
    runs = getattr(target, "__qualname__", None) or type(thread).__qualname__ + ".run"
    return f"{thread.name} ({'daemon' if thread.daemon else 'non-daemon'}, runs {runs})"


@pytest.fixture(autouse=True)
def _threads_end_with_their_test() -> Iterator[None]:
    """Fail the test if a thread it started is still running :data:`BOUND` seconds after it."""
    before = frozenset(threading.enumerate())
    yield
    left = threads_left(before, within=BOUND)
    if left:
        pytest.fail(
            f"the test left {len(left)} thread(s) running {BOUND} s after it ended: "
            + "; ".join(_describe(thread) for thread in left)
            + ". Close or join what started them inside the test.",
            pytrace=False,
        )
