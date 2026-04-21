"""Tests for shared JAX synchronization helpers."""

from __future__ import annotations

from dataclasses import dataclass

from datarax.performance.synchronization import (
    block_until_ready_tree,
    copy_to_host_async_tree,
)


@dataclass
class SynchronizableLeaf:
    """Leaf object recording synchronization calls."""

    block_calls: int = 0
    copy_calls: int = 0

    def block_until_ready(self) -> SynchronizableLeaf:
        self.block_calls += 1
        return self

    def copy_to_host_async(self) -> None:
        self.copy_calls += 1


def test_block_until_ready_tree_waits_on_all_leaves_and_returns_input() -> None:
    """Tree synchronization should visit every leaf without rebuilding the tree."""
    left = SynchronizableLeaf()
    right = SynchronizableLeaf()
    nested = {"left": left, "items": [right, object()]}

    result = block_until_ready_tree(nested)

    assert result is nested
    assert left.block_calls == 1
    assert right.block_calls == 1
    assert left.copy_calls == 0
    assert right.copy_calls == 0


def test_copy_to_host_async_tree_is_explicit_and_non_blocking() -> None:
    """Async host copies are separate from blocking synchronization."""
    leaf = SynchronizableLeaf()

    assert copy_to_host_async_tree({"leaf": leaf}) == {"leaf": leaf}

    assert leaf.copy_calls == 1
    assert leaf.block_calls == 0
