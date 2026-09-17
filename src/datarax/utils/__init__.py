"""Datarax utilities module.

This module exposes common utility submodules for working with:

- External library integration (`external`)
- PyTree and Batch manipulation (`pytree_utils`)
- Two-tier dataset cache layout (`cache`)
- Multirate signal alignment (`multirate`)

``jax.ShapeDtypeStruct`` PyTree spec helpers, per-record PRNG key derivation and
the named ``nnx.Rngs`` streams live in :mod:`datarax.core.spec` and
:mod:`datarax.core.prng` (the foundational layer that both operators and sources
depend on).
"""

from . import cache, external, multirate, pytree_utils


__all__ = ["cache", "external", "multirate", "pytree_utils"]
