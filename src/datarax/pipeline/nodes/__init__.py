"""JAX-native DAG nodes for ``Pipeline.from_dag``.

``SplitField`` is a structural DAG node that is not an element transform: field selection.
``CachingIterator`` provides iteration-boundary caching (its grain counterpart is
``grain.experimental.CacheIterDataset``). A ``(K, B, ...)`` chunk of batches is
``batch_ops.stack(batch_ops.split(batch, K))``.
"""

from __future__ import annotations

from datarax.pipeline.nodes.cache import CachingIterator
from datarax.pipeline.nodes.split_field import SplitField


__all__ = ["CachingIterator", "SplitField"]
