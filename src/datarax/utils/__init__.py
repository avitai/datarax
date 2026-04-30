"""Datarax utilities module.

This module exposes common utility submodules for working with:

- External library integration (`external`)
- Random number generation (`prng`)
- PyTree and Batch manipulation (`pytree_utils`)
- Two-tier dataset cache layout (`cache`)
- ``jax.ShapeDtypeStruct`` PyTree spec helpers (`spec`)
- Multirate signal alignment (`multirate`)
"""

from . import cache, external, multirate, prng, pytree_utils, spec


__all__ = ["cache", "external", "multirate", "prng", "pytree_utils", "spec"]
