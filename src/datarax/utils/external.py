"""External utility functions for Datarax.

This module provides utility functions for working with external libraries and
interfaces, particularly focused on JAX and Flax NNX integration.
"""

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import flax.nnx as nnx
import jax
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


T = TypeVar("T")


@dataclass
class ExternalAdapterConfig(OperatorConfig):
    """Configuration for ExternalLibraryAdapter.

    Inherits from OperatorConfig. Always stochastic since external
    functions typically require RNG keys.

    Attributes:
        stream_name: Name of the RNG stream to use (default: "augment").
    """

    stochastic: bool = True
    stream_name: str = "augment"


class ExternalLibraryAdapter(OperatorModule):
    """Adapter for external libraries that require raw JAX PRNG keys.

    This adapter provides a module-based approach for integrating with external
    libraries that require raw JAX PRNG keys, maintaining compatibility with
    NNX transformations like nnx.jit, nnx.vmap, etc.

    Use this when you need to:

    - Wrap external functions that use JAX keys in an NNX module
    - Apply NNX transformations (jit, vmap) to functions using JAX keys
    - Integrate external augmentation libraries into Datarax pipelines

    Examples:
        def augment_fn(data, key):
            noise = jax.random.normal(key, shape=data["image"].shape)
            return {**data, "image": data["image"] + noise * 0.1}
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, augment_fn, rngs=rngs)
        batch = Batch(...)
        augmented = adapter(batch)
    """

    def __init__(
        self,
        config: ExternalAdapterConfig,
        fn: Callable[[dict[str, Any], jax.Array], dict[str, Any]],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the ExternalLibraryAdapter.

        Args:
            config: Configuration for the adapter.
            fn: Function that takes (data_dict, key) where data_dict is the
                element's data dictionary and key is a raw JAX PRNG key.
            rngs: Rngs object for randomness (required since always stochastic).
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.fn = fn

    def get_output_structure(
        self,
        sample_data: PyTree,
        sample_state: PyTree,
    ) -> tuple[PyTree, PyTree]:
        """Declare output structure for vmap axis specification.

        ExternalLibraryAdapter.apply() requires random_params which isn't available
        during jax.eval_shape tracing. We trace through fn with a dummy key.

        Args:
            sample_data: Single element data (not batched)
            sample_state: Single element state (not batched)

        Returns:
            Tuple of (output_data_structure, output_state_structure) with 0 leaves.
        """

        def apply_wrapper(data: PyTree, state: PyTree) -> tuple[PyTree, PyTree]:
            # Use dummy key for tracing (value doesn't matter for shape inference)
            dummy_key = jax.random.key(0)
            transformed_data = self.fn(data, dummy_key)
            return transformed_data, state

        out_shapes = jax.eval_shape(apply_wrapper, sample_data, sample_state)
        out_data_struct = jax.tree.map(lambda _: 0, out_shapes[0])
        out_state_struct = jax.tree.map(lambda _: 0, out_shapes[1])
        return out_data_struct, out_state_struct

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> jax.Array:
        """Generate random key for batch - one key per element.

        Args:
            rng: JAX random key
            data_shapes: PyTree with data shapes (used to determine batch size)

        Returns:
            Array of random keys, one per batch element
        """
        # Get batch size from the first leaf shape
        # Get batch size from the first leaf shape
        # We treat tuple as a leaf to preserve the shape tuple
        leaves = jax.tree_util.tree_leaves(data_shapes, is_leaf=lambda x: isinstance(x, tuple))
        first_shape = leaves[0]
        batch_size = first_shape[0] if len(first_shape) > 0 else 1

        # Generate one key per batch element
        return jax.random.split(rng, batch_size)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply the external function to a single element.

        Args:
            data: Element data dictionary
            state: Element state (passed through)
            metadata: Element metadata (passed through)
            random_params: Random key for this element
            stats: Statistics (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
        """
        if random_params is None:
            raise ValueError("ExternalLibraryAdapter requires random_params (RNG key)")

        # Apply the external function
        transformed_data = self.fn(data, random_params)
        return transformed_data, state, metadata


class PureJaxAdapter(OperatorModule):
    """Adapter for pure JAX functions (stateless, no RNG).

    This adapter wraps pure JAX functions of the form `fn(data) -> data`.
    It sets stochastic=False by default and does not generate random params.

    Examples:
        def normalize(data):
            return {**data, "image": data["image"] / 255.0}

        config = ExternalAdapterConfig(stochastic=False, stream_name=None)
        adapter = PureJaxAdapter(config, normalize)
        batch = Batch(...)
        normalized = adapter(batch)
    """

    def __init__(
        self,
        config: ExternalAdapterConfig,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        name: str | None = None,
    ):
        """Initialize PureJaxAdapter.

        Args:
            config: Configuration (must have stochastic=False).
            fn: Pure function taking data dict and returning data dict.
            name: Optional module name.
        """
        if config.stochastic:
            raise ValueError("PureJaxAdapter requires stochastic=False config")

        super().__init__(config, rngs=None, name=name)
        self.fn = fn

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> None:
        """No random params for pure functions."""
        return None

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply pure function."""
        transformed_data = self.fn(data)
        return transformed_data, state, metadata


def to_datarax_operator(
    fn: Callable[..., Any],
    stochastic: bool = True,
    *,
    stream_name: str | None = "augment",
    rngs: nnx.Rngs | None = None,
    name: str | None = None,
) -> OperatorModule:
    """Convert a function into a Datarax OperatorModule.

    This utility simplifies the creation of operator adapters.

    Args:
        fn: The function to adapt.
        stochastic: Whether the function uses randomness.
        stream_name: Name of the RNG stream (required if stochastic=True).
        rngs: Rngs object (required if stochastic=True).
        name: Name of the module.

    Returns:
        An OperatorModule (either ExternalLibraryAdapter or PureJaxAdapter).

    Examples:
        # Pure function
        op = to_datarax_operator(lambda d: d, stochastic=False)

        # Stochastic function
        op = to_datarax_operator(aug_fn, stochastic=True, rngs=rngs)
    """
    if stochastic:
        config = ExternalAdapterConfig(stochastic=True, stream_name=stream_name)
        return ExternalLibraryAdapter(config, fn, rngs=rngs, name=name)
    else:
        # If stochastic is False, stream_name must be None.
        # If user passed a stream_name (or default "augment"), we override it to None
        # to ensure valid config.
        config = ExternalAdapterConfig(stochastic=False, stream_name=None)
        return PureJaxAdapter(config, fn, name=name)


def with_jax_key_wrapper(
    fn: Callable[[Any, jax.Array], Any],
) -> Callable[[Any, nnx.RngStream], Any]:
    """Wrap a function that requires a raw JAX PRNG key to work with RngStream.

    This function takes a function that expects a raw JAX PRNG key and returns
    a function that can work with NNX RngStream objects.

    Args:
        fn: Function that takes (data, key) where key is a raw JAX PRNG key

    Returns:
        Function that takes (data, stream) where stream is an RngStream

    Examples:
        def external_fn(data, key):
            noise = jax.random.normal(key, shape=data.shape)
            return data + noise
        wrapped_fn = with_jax_key_wrapper(external_fn)
        rngs = nnx.Rngs(augment=42)
        result = wrapped_fn(data, rngs['augment'])
    """

    def wrapped_fn(data: Any, stream: nnx.RngStream) -> Any:
        key = stream()
        return fn(data, key)

    return wrapped_fn


def with_jax_key(fn: Callable[[Any, jax.Array], Any]) -> Callable[[Any, nnx.RngStream], Any]:
    """Decorator version of with_jax_key_wrapper.

    This decorator can be applied to functions that require raw JAX PRNG keys
    to make them compatible with NNX RngStream objects.

    Args:
        fn: Function that takes (data, key) where key is a raw JAX PRNG key

    Returns:
        Function that takes (data, stream) where stream is an RngStream

    Examples:
        @with_jax_key
        def external_fn(data, key):
            noise = jax.random.normal(key, shape=data.shape)
            return data + noise
        rngs = nnx.Rngs(augment=42)
        result = external_fn(data, rngs['augment'])
    """
    return with_jax_key_wrapper(fn)
