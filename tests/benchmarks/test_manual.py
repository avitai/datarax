"""Focused vmap behavior tests for JAX and Flax NNX.

These tests validate explicit expected behavior instead of acting as
print-based manual diagnostics.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def create_test_data() -> dict[str, np.ndarray]:
    """Create deterministic test data."""
    rng = np.random.default_rng(42)
    images = rng.random((10, 32, 32, 3), dtype=np.float32)
    labels = rng.integers(0, 10, size=(10,), dtype=np.int32)
    return {"image": images, "label": labels}


def normalize(element: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Normalize image field while preserving label field."""
    return {
        "image": element["image"] / 255.0,
        "label": element["label"],
    }


def test_jax_vmap_over_dict_batch() -> None:
    """jax.vmap should handle dict batches as PyTrees."""
    data = create_test_data()
    batch = {k: v[:4] for k, v in data.items()}
    vmapped_fn = jax.vmap(normalize, in_axes=0)

    result = vmapped_fn(batch)  # type: ignore[reportArgumentType]

    assert result["image"].shape == (4, 32, 32, 3)
    assert result["label"].shape == (4,)
    assert jnp.allclose(result["image"], batch["image"] / 255.0)
    assert jnp.array_equal(result["label"], batch["label"])


def test_jax_vmap_over_tuple_batch() -> None:
    """jax.vmap should handle tuple batches."""
    data = create_test_data()
    batch = {k: v[:4] for k, v in data.items()}
    batch_as_tuple = (batch["image"], batch["label"])

    def normalize_tuple(data_tuple: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        images, labels = data_tuple
        return images / 255.0, labels

    vmapped_tuple_fn = jax.vmap(normalize_tuple)
    result_images, result_labels = vmapped_tuple_fn(batch_as_tuple)  # type: ignore[reportArgumentType]

    assert result_images.shape == (4, 32, 32, 3)
    assert result_labels.shape == (4,)
    assert jnp.allclose(result_images, batch["image"] / 255.0)
    assert jnp.array_equal(result_labels, batch["label"])


def test_nnx_vmap_over_array() -> None:
    """nnx.vmap should apply elementwise array transformation."""
    data = create_test_data()
    test_array = data["image"][:4]

    def scale_array(x: jax.Array) -> jax.Array:
        return x / 255.0

    nnx_vmapped = nnx.vmap(scale_array)
    result_array = nnx_vmapped(test_array)  # type: ignore[reportArgumentType]

    assert result_array.shape == test_array.shape
    assert jnp.allclose(result_array, test_array / 255.0)
