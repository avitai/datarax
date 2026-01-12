"""PRNG handling utilities for Datarax.

This module provides utilities for managing random number generation in Datarax,
focusing on compatibility with Flax NNX and JAX's functional paradigm.

Key Utilities:
    create_rngs: Creates `flax.nnx.Rngs` objects with multiple named streams
                 derived from a single master seed.

Usage Note:
    Internal Datarax code should exclusively use `flax.nnx.Rngs` for randomness.
    When interfacing with external libraries that require raw JAX PRNG keys, call
    the specific stream (e.g., `rngs.params()`) to generate a unique key.
"""

import flax.nnx as nnx
import jax

# Standard stream names for consistency
DEFAULT_RNG_STREAMS = ["augment", "dropout", "params", "shuffling", "default"]


def create_rngs(seed: int | None = None, streams: list[str] | None = None) -> nnx.Rngs:
    """Create an Rngs object with the specified streams.

    This is a convenience function that creates multiple RNG streams from a single seed.
    For simple cases, you can use nnx.Rngs directly:
        rngs = nnx.Rngs(42)  # Single default stream
        rngs = nnx.Rngs(params=0, dropout=1)  # Multiple streams

    Args:
        seed: Optional seed for PRNG. Defaults to 0.
        streams: List of stream names. Defaults to DEFAULT_RNG_STREAMS.

    Returns:
        An nnx.Rngs object with keys for each stream.
    """
    if streams is None:
        streams = DEFAULT_RNG_STREAMS

    seed = seed if seed is not None else 0
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(streams))
    return nnx.Rngs(**{name: keys[i] for i, name in enumerate(streams)})
