"""Datarax sharding components.

Datarax slices data for the current JAX process, and a pipeline given a ``sharding`` places its
batches on that data mesh (``Pipeline(..., sharding=...)``). Building meshes and placing other
arrays come from ``substrax.mesh`` and ``substrax.spmd``.
"""

from datarax.sharding.jax_process_sharder import JaxProcessSharderConfig, JaxProcessSharderModule


__all__ = ["JaxProcessSharderConfig", "JaxProcessSharderModule"]
