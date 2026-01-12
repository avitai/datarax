# from typing import Iterator
# import jax
# import jax.numpy as jnp
# import flax.nnx as nnx

# from datarax.core.data_source import DataSourceModule
# from datarax.config.registry import register_component
# from datarax.typing import Element


# @register_component("source", "MixDataSources")
# class MixDataSourcesNode(DataSourceModule):
#     """Mix multiple data sources with weights."""
#     def __init__(self,
#         sources: list[DataSourceModule],
#         weights: list[float],
#         rngs: nnx.Rngs | None = None,
#         shuffle: bool = False,
#         cache_size: int = 0,
#         prefetch_size: int = 0,
#         track_metadata: bool = False,
#         shard_id: int | None = None,
#     ):
#         super().__init__(rngs=rngs,
#             cacheable=cache_size > 0,
#             prefetchable=prefetch_size > 0,
#             track_metadata=track_metadata,
#             shard_id=shard_id,
#         )
#         self.sources = sources
#         self.weights = weights
#         self.rng = jax.random.key(0)
#         assert len(sources) == len(weights), "Number of sources and weights must match"

#     def __iter__(self) -> Iterator[Element]:
#         """Iterate over the mixed data sources."""
#         return iter([iter(source) for source in self.sources])

#     def __next__(self) -> Element:
#         """Get the next element from the mixed data sources."""
#         return [next(source)  for source in self.sources]
