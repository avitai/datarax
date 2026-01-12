"""Tests for module-based PRNG handling in Datarax.

This module contains tests for the module-first approach to PRNG handling in Datarax,
focusing on compatibility with NNX transformations.

See tests/operators/test_element_operator.py for ElementOperator tests.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.typing import Element


def test_with_jax_key_wrapper():
    """Test the with_jax_key wrapper function for external library integration."""
    from datarax.utils.external import with_jax_key_wrapper

    # Define a function that requires a raw JAX PRNG key
    def external_function(data: Element, key: jax.Array) -> Element:
        # Generate noise for the data field
        noise = jax.random.normal(key, shape=data["data"].shape)  # type: ignore
        # Add noise only to the 'data' field
        return {"data": data["data"] + noise}

    # Wrap the function to work with RngStreams
    wrapped_function = with_jax_key_wrapper(external_function)

    # Create an RngStream
    rngs: nnx.Rngs = nnx.Rngs(augment=jax.random.key(42))
    stream = rngs["augment"]

    # Create test data
    data: Element = {"data": jnp.zeros((3, 3))}

    # Apply the wrapped function
    result = wrapped_function(data, stream)

    # Verify the result is not zeros (noise was added)
    assert not jnp.allclose(result["data"], data["data"])  # type: ignore


def test_with_jax_key_decorator():
    """Test the with_jax_key decorator for external library integration."""
    from datarax.utils.external import with_jax_key

    # Define a function that requires a raw JAX PRNG key and decorate it
    @with_jax_key
    def decorated_function(data: Element, key: jax.Array) -> Element:
        # Generate noise for each field in the data dict
        noise = jax.tree.map(lambda x: jax.random.normal(key, shape=x.shape), data)
        # Use tree.map to add noise to each field
        return jax.tree.map(lambda x, y: x + y, data, noise)

    rngs: nnx.Rngs = nnx.Rngs(augment=jax.random.key(42))
    stream = rngs["augment"]

    # Create test data
    data: Element = {"data": jnp.zeros((3, 3))}

    # Apply the decorated function directly with a stream
    result = decorated_function(data, stream)

    # Verify the result is not zeros (noise was added)
    assert not jnp.allclose(result["data"], data["data"])  # type: ignore
