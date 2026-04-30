"""Shared-memory utilities for multi-worker data pipelines.

Re-exports the public API documented at ``docs/memory/index.md``. Consumers
should import via this package path; submodule paths like
``datarax.memory.shared_memory_manager`` are implementation details.
"""

from __future__ import annotations

from datarax.memory.shared_memory_manager import SharedMemoryManager


__all__ = ["SharedMemoryManager"]
