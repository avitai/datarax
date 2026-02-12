"""Utilities for Datarax testing."""

import functools
import time
from typing import Any, TypeVar, Union
from collections.abc import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


# Type variables for generics
T = TypeVar("T")
F = TypeVar("F", bound=Callable)


def assert_tree_equal(tree1: Any, tree2: Any, rtol: float = 1e-5, atol: float = 1e-5) -> None:
    """Assert that two JAX PyTrees are equal.

    Args:
        tree1: First PyTree.
        tree2: Second PyTree.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.
    """
    # Check tree structure first
    jax.tree.map(lambda x, y: assert_array_shape_equal(x, y), tree1, tree2)

    # Check values
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol), tree1, tree2)


def assert_array_shape_equal(x: Any, y: Any) -> None:
    """Assert that two arrays have the same shape.

    Args:
        x: First array.
        y: Second array.
    """
    # Only check shapes if both objects have a shape attribute (i.e., they're arrays)
    if hasattr(x, "shape") and hasattr(y, "shape"):
        assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    # Skip type checking for scalars vs 0-d arrays (common in transformations)
    elif (
        isinstance(x, int | float) and hasattr(y, "shape") and getattr(y, "shape", None) == ()
    ) or (isinstance(y, int | float) and hasattr(x, "shape") and getattr(x, "shape", None) == ()):
        # Scalar vs 0-d array is acceptable
        pass
    # For non-array types, just check type equality
    else:
        assert type(x) == type(y), f"Type mismatch: {type(x)} vs {type(y)}"


def assert_nnx_module_equal(
    module1: nnx.Module, module2: nnx.Module, rtol: float = 1e-5, atol: float = 1e-5
) -> None:
    """Assert that two NNX modules have the same structure and parameters.

    Args:
        module1: First module.
        module2: Second module.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.
    """
    # Split modules to get their states
    graph1, state1 = nnx.split(module1)
    graph2, state2 = nnx.split(module2)

    # Compare states with tree_equal
    assert_tree_equal(state1, state2, rtol=rtol, atol=atol)


def assert_batch_contains_keys(batch: dict[str, Any], keys: list[str]) -> None:
    """Assert that a batch contains all specified keys.

    Args:
        batch: The batch to check.
        keys: List of keys that should be in the batch.
    """
    for key in keys:
        assert key in batch, f"Expected key '{key}' in batch, but it was not found"


def assert_batch_shape(
    batch: dict[str, Any], key: str, expected_shape: Union[tuple[int, ...], list[int]]
) -> None:
    """Assert that a batch key has the expected shape.

    Args:
        batch: The batch to check.
        key: The key to check the shape for.
        expected_shape: The expected shape.
    """
    assert key in batch, f"Expected key '{key}' in batch, but it was not found"
    assert batch[key].shape == tuple(expected_shape), (
        f"Expected shape {expected_shape} for key '{key}', but got {batch[key].shape}"
    )


def time_execution(func: F) -> F:
    """Time the execution of a function.

    Args:
        func: The function to time.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper  # type: ignore[return-value]


def requires_device_type(device_type: str) -> Callable[[F], F]:
    """Mark a test as requiring a specific device type.

    Args:
        device_type: The device type ('cpu', 'gpu', or 'tpu').

    Returns:
        The decorator function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if jax.local_device_count(device_type) == 0:
                import pytest

                pytest.skip(f"Test requires {device_type}")
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def requires_multi_devices(min_count: int = 2) -> Callable[[F], F]:
    """Mark a test as requiring multiple devices.

    Args:
        min_count: The minimum number of devices required.

    Returns:
        The decorator function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if jax.local_device_count() < min_count:
                import pytest

                pytest.skip(f"Test requires at least {min_count} devices")
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def data_generator(size: int = 100, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate reproducible test data.

    Args:
        size: The number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary containing test data.
    """
    np.random.seed(seed)

    # Generate features and labels
    features = np.random.randn(size, 16).astype(np.float32)
    labels = np.random.randint(0, 5, size=(size,))

    return {
        "features": features,
        "labels": labels,
    }


def create_test_nnx_module(rng_seed: int = 0) -> nnx.Module:
    """Create a test NNX module with initialized parameters.

    Args:
        rng_seed: Random seed for initialization.

    Returns:
        An initialized NNX module.
    """

    class TestModule(nnx.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = nnx.Linear(in_features=16, out_features=32, rngs=nnx.Rngs(0))
            self.dense2 = nnx.Linear(in_features=32, out_features=10, rngs=nnx.Rngs(1))

        def __call__(self, x, training: bool = False):
            x = self.dense1(x)
            x = jax.nn.relu(x)
            x = self.dense2(x)
            return x

    # Create and initialize the module
    module = TestModule()
    x = jnp.ones((1, 16))
    # Module is already initialized with rngs, just call it
    module(x)

    return module
