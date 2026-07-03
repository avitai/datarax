"""JAX-native DAG nodes for ``Pipeline.from_dag``.

The ``nnx.Module`` nodes (``RebatchNode``, ``SplitField``) provide structural
DAG operations that are not element transforms: differentiable rebatching and
field selection. ``CachingIterator`` provides iteration-boundary caching (its
grain counterpart is ``grain.experimental.CacheIterDataset``); cross-step,
non-differentiable rebatching is likewise available via
``grain.experimental.RebatchIterDataset`` at the iteration level.
"""

from __future__ import annotations

from datarax.pipeline.nodes.cache import CachingIterator
from datarax.pipeline.nodes.rebatch import RebatchNode
from datarax.pipeline.nodes.split_field import SplitField


__all__ = ["CachingIterator", "RebatchNode", "SplitField"]
