from __future__ import annotations
import jax
import hashlib
from collections import OrderedDict
import flax.nnx as nnx
from typing import Any

from datarax.dag.nodes.base import Node
from datarax.typing import Batch


class Cache(Node):
    """Caching node that stores results for repeated inputs.

    Useful for expensive transformations that might be called
    with the same inputs multiple times.

    Examples:
        Basic caching:

        ```python
        from datarax.nodes import Cache, Identity
        cached = Cache(Identity(name="expensive"), cache_size=100)
        ```
    """

    def __init__(self, node: Node, cache_size: int = 100):
        """Initialize caching node.

        Args:
            node: Node to wrap with caching
            cache_size: Maximum number of entries to cache
        """
        super().__init__()
        self.node = node
        self.cache_size = cache_size
        self._cache = nnx.Dict({})  # NNX-compatible cache storage
        self._cache_order = nnx.List([])  # For LRU eviction

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Execute with caching.

        Args:
            data: Input data
            key: Optional RNG key

        Returns:
            Cached or computed result
        """
        # Only cache deterministic calls (no RNG)
        if key is None:
            cache_key = self._compute_cache_key(data)

            # Check cache
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._cache_order.remove(cache_key)
                self._cache_order.append(cache_key)
                return self._cache[cache_key]

            # Compute result
            result = self.node(data, key=key)

            # Add to cache
            self._cache[cache_key] = result
            self._cache_order.append(cache_key)

            # Evict oldest if cache full
            if len(self._cache) > self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]

            return result
        else:
            # Don't cache non-deterministic calls
            return self.node(data, key=key)

    def _compute_cache_key(self, data: Any) -> int:
        """Compute cache key for data."""
        if isinstance(data, jax.Array):
            return hash((data.shape, data.dtype, float(data.sum())))
        elif isinstance(data, dict):
            return hash(tuple(sorted(data.keys())))
        else:
            return hash(str(data))

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()

    def __repr__(self) -> str:
        """String representation."""
        node_name = self.node.name if hasattr(self.node, "name") else str(self.node)
        return f"Cache(node={node_name}, size={self.cache_size})"


class CacheNode(Node):
    """Node that caches results of expensive operations.

    Uses LRU-style caching with configurable size.
    """

    def __init__(self, inner_node: Node, cache_size: int = 100, name: str | None = None):
        """Initialize cache node.

        Args:
            inner_node: Node whose results to cache
            cache_size: Maximum cache entries
            name: Optional name
        """
        super().__init__(name=name or f"Cache({inner_node.name})")
        self.inner_node = inner_node
        self.cache_size = cache_size
        self.cache = nnx.Variable(OrderedDict())
        self.hits = nnx.Variable(0)
        self.misses = nnx.Variable(0)

    def __call__(self, data: Batch, *, key: jax.Array | None = None) -> Batch:
        """Execute with caching.

        Args:
            data: Input batch
            key: RNG key (disables caching if present)

        Returns:
            Cached or computed result
        """
        # Don't cache stochastic operations
        if key is not None:
            self.misses.set_value(self.misses.get_value() + 1)
            return self.inner_node(data, key=key)

        # Compute cache key
        cache_key = self._compute_cache_key(data)

        # Get cache dict copy (NNX Variables return copies, not references)
        cache_dict = self.cache.get_value()

        # Check cache
        if cache_key in cache_dict:
            self.hits.set_value(self.hits.get_value() + 1)
            # Move to end (LRU) and persist
            cache_dict.move_to_end(cache_key)
            self.cache.set_value(cache_dict)
            return cache_dict[cache_key]

        # Compute and cache
        self.misses.set_value(self.misses.get_value() + 1)
        result = self.inner_node(data, key=key)

        # Add to cache
        cache_dict[cache_key] = result
        cache_dict.move_to_end(cache_key)

        # Enforce size limit
        while len(cache_dict) > self.cache_size:
            cache_dict.popitem(last=False)

        # Persist cache updates
        self.cache.set_value(cache_dict)

        return result

    def _compute_cache_key(self, data: Any) -> str:
        """Compute hash key for data."""
        if isinstance(data, dict):
            # Hash dict structure and values
            key_parts = []
            for k, v in sorted(data.items()):
                if hasattr(v, "shape"):
                    # Include shape, dtype, and a hash of the actual values
                    key_parts.append(f"{k}:{v.shape}:{v.dtype}:{float(v.sum())}")
                else:
                    key_parts.append(f"{k}:{hash(v)}")
            key_str = "|".join(key_parts)
        elif hasattr(data, "shape"):
            key_str = f"{data.shape}:{data.dtype}:{float(data.sum())}"
        else:
            key_str = str(hash(data))

        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.set_value(OrderedDict())

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        hits = self.hits.get_value()
        misses = self.misses.get_value()
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache.get_value()),
        }
