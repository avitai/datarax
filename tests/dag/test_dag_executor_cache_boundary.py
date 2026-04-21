"""Cache-boundary regression tests for DAGExecutor iteration."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import jax.numpy as jnp

from datarax.dag.dag_executor import DAGExecutor
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def test_element_iteration_disables_cache_without_mutating_cache_slot() -> None:
    """Element fallback should not mutate executor cache state during cleanup."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(3)})
    executor = DAGExecutor(enable_caching=True, enforce_batch=False).add(source)
    original_cache = executor._memo
    use_cache_values: list[bool] = []

    def execute_spy(
        self: DAGExecutor,
        node: Any,
        data: Any,
        key: Any = None,
        *,
        use_cache: bool = True,
    ) -> Any:
        del node, key
        use_cache_values.append(use_cache)
        assert self._memo is original_cache
        return data

    with patch.object(DAGExecutor, "_execute", execute_spy):
        iterator = iter(executor)
        try:
            next(iterator)
        finally:
            executor.close()

    assert use_cache_values == [False]
    assert executor._memo is original_cache
