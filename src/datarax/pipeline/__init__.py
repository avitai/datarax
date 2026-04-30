"""``datarax.pipeline`` — JAX-native data pipeline package.

Provides :class:`Pipeline`, an ``nnx.Module`` whose ``__call__`` is a
Python composition of source + stages that JAX traces directly into a
single XLA graph. Replaces the legacy tree-walking executor with the
canonical Flax NNX pattern: stateful ``nnx.Module`` composition with
``nnx.scan`` over the iteration loop.

Public API:

- :class:`Pipeline` — linear chain of stages over a data source. Each
  stage is an ``nnx.Module`` whose ``__call__(batch) -> batch``
  transforms the batch. ``Pipeline.epoch`` runs the entire epoch
  (data fetch + stages + user body function) as one XLA call via
  ``nnx.scan``.
"""

from datarax.pipeline.pipeline import Pipeline
from datarax.pipeline.topo import topological_sort, validate_dag


__all__ = ["Pipeline", "topological_sort", "validate_dag"]
