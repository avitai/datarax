"""Shared synchronization helpers for timing JAX work."""

from __future__ import annotations

from typing import Any

import jax


def block_until_ready_tree(value: Any) -> Any:
    """Synchronize every JAX-like leaf in a pytree and return the original value."""
    return jax.block_until_ready(value)


def copy_to_host_async_tree(value: Any) -> Any:
    """Start async host copies for every supporting leaf and return the original value."""
    return jax.copy_to_host_async(value)
