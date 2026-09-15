"""Datarax sharding components.

Datarax slices data for the current JAX process. Placing batches on a device mesh and naming
mesh axes come from ``substrax.spmd`` and ``substrax.mesh``.
"""

from datarax.sharding.jax_process_sharder import JaxProcessSharderConfig, JaxProcessSharderModule


__all__ = ["JaxProcessSharderConfig", "JaxProcessSharderModule"]
