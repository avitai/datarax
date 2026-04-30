"""Reserved namespace for the future multiprocessing backend.

This module is intentionally empty. It reserves the ``datarax.workers`` namespace
for the planned multiprocessing-worker subsystem that will scale CPU-heavy
transforms (image decoding, tokenization, etc.) beyond the GIL by running them
in a process pool with shared-memory output.

Why a placeholder instead of dead code or no module
---------------------------------------------------

The directory once held a worker-pool implementation that was removed during a
refactor; the namespace is preserved because it is the architecturally correct
home for cross-cutting multiprocess infrastructure. Without a documented
placeholder, the empty directory reads as dead code; with one, the reservation
is unambiguous.

Existing parallel-worker concepts elsewhere in datarax (do not duplicate)
-------------------------------------------------------------------------

- ``MemorySource(num_workers=N)`` (``datarax.sources.memory_source``) shards an
  in-memory dataset by ``positions[k::num_workers]`` per worker.
- ``beam_num_workers`` on ``TFDSEagerSource`` / ``TFDSStreamingSource``
  (``datarax.sources.tfds_source``) controls Apache Beam DirectRunner workers
  for TFDS preparation.
- ``datarax.control.prefetcher`` provides asynchronous prefetching via threads
  (limited by the GIL for CPU-heavy work).

What goes here when the multiprocessing backend is implemented
--------------------------------------------------------------

A ``ProcessPoolExecutor``-based producer for the prefetcher and a
multiprocessing-backed source/transform option that uses ``SharedMemoryManager``
(``datarax.memory.SharedMemoryManager``) for cross-process data sharing. The
expected public entry point is ``WorkerPoolModule``.

Until then, accessing any symbol in this module raises ``NotImplementedError``
with a pointer to the existing alternatives, so the reservation is unmistakable
and contributors don't reinvent worker logic in other modules.
"""

from __future__ import annotations

from typing import Any


_RESERVATION_NOTICE = (
    "datarax.workers is the reserved namespace for the planned multiprocessing "
    "backend (process-pool prefetcher + multiprocessing-backed transforms). "
    "It is not implemented yet. "
    "Existing parallel-worker concepts: MemorySource(num_workers=...), "
    "beam_num_workers on TFDS sources, datarax.control.prefetcher (threaded)."
)


def __getattr__(name: str) -> Any:
    """Raise a clear ``NotImplementedError`` for any access to a not-yet-implemented symbol.

    This converts the default ``AttributeError`` (which is ambiguous between
    "typo" and "reserved-but-not-implemented") into a discoverable signal that
    points contributors to the existing alternatives.
    """
    raise NotImplementedError(
        f"datarax.workers.{name} is not implemented yet. {_RESERVATION_NOTICE}"
    )


__all__: list[str] = []
