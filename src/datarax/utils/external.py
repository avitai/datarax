"""External utility functions for Datarax.

This module provides utility functions for working with external libraries and
interfaces, particularly focused on JAX and Flax NNX integration.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass(frozen=True)
class ExternalAdapterConfig(OperatorConfig):
    """Configuration for ExternalLibraryAdapter.

    Inherits from OperatorConfig. Always stochastic since external
    functions typically require RNG keys.

    Attributes:
        stream_name: Name of the RNG stream to use (default: "augment").
    """

    stochastic: bool = True
    stream_name: str | None = "augment"


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
    ) -> None:
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

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply the external function to a single element.

        Args:
            data: Element data dictionary
            state: Element state (passed through)
            metadata: Element metadata (passed through)
            key: This record's PRNG key, or ``None`` for a deterministic adapter
            stats: Statistics (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
        """
        del stats
        # Stochastic adapters receive a per-record key; deterministic ones get none, so
        # fall back to a fixed key for reproducible "deterministic noise" (the external fn
        # always requires a key argument).
        key = key if key is not None else jax.random.key(0)
        transformed_data = self.fn(data, key)
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
    ) -> None:
        """Initialize PureJaxAdapter.

        Args:
            config: Configuration (must have stochastic=False).
            fn: Pure function taking data dict and returning data dict.
            name: Optional module name.

        Raises:
            ValueError: If ``config.stochastic`` is set; the adapter wraps a pure function.
        """
        if config.stochastic:
            raise ValueError("PureJaxAdapter requires stochastic=False config")

        super().__init__(config, rngs=None, name=name)
        self.fn = fn

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply pure function."""
        del key, stats
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
