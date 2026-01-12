"""Manual test for nnx transformations in Datarax.

This is a simplified test to debug the batch structure and vmap issues.
"""

import jax
import numpy as np
from flax import nnx


# Create simple test data
def create_test_data():
    np.random.seed(42)
    images = np.random.rand(10, 32, 32, 3).astype(np.float32)
    labels = np.random.randint(0, 10, size=(10,)).astype(np.int32)
    return {"image": images, "label": labels}


# Simple transform function
def normalize(element):
    if isinstance(element, dict) and "image" in element:
        result = dict(element)
        result["image"] = result["image"] / 255.0
        return result
    return element


# Create a vmapped function
def test_vmap():
    print("\n=== Testing vmap ===")

    # Create data
    data = create_test_data()
    print("Data structure:", jax.tree.map(lambda x: x.shape, data))

    # Create a batch manually by taking a slice
    batch = {k: v[:4] for k, v in data.items()}
    print("Batch structure:", jax.tree.map(lambda x: x.shape, batch))

    # Define a simple vmap function
    vmapped_fn = jax.vmap(normalize, in_axes=0)

    # Try alternative format
    batch_as_tuple = tuple(batch[k] for k in ["image", "label"])
    print("Batch as tuple:", jax.tree.map(lambda x: x.shape, batch_as_tuple))

    try:
        # Try to apply to the batch directly (expected to fail)
        print("Attempting to apply vmap to batch dict...")
        result = vmapped_fn(batch)
        print("Result:", jax.tree.map(lambda x: x.shape, result))
    except Exception as e:
        print("Failed with error:", str(e))

    # Define a function that works on a tuple of arrays
    def normalize_tuple(data_tuple):
        images, labels = data_tuple
        return images / 255.0, labels

    vmapped_tuple_fn = jax.vmap(normalize_tuple)

    try:
        print("\nAttempting to apply vmap to batch as tuple...")
        result_tuple = vmapped_tuple_fn(batch_as_tuple)
        print("Result tuple:", jax.tree.map(lambda x: x.shape, result_tuple))
    except Exception as e:
        print("Failed with error:", str(e))

    # Try a simpler approach - apply to individual elements first
    print("\nProcessing batch by elements:")
    processed = []
    for i in range(len(batch["image"])):
        element = {"image": batch["image"][i], "label": batch["label"][i]}
        processed.append(normalize(element))

    print("Processed first element:", processed[0])

    # Create a new flax.nnx vmap transformation function
    print("\nTesting flax.nnx.vmap with a simple array function:")

    # Simple function that works on a single array
    def scale_array(x):
        return x / 255.0

    # Get a sample array
    test_array = batch["image"]
    print("Test array shape:", test_array.shape)

    # Create a vmapped function using nnx.vmap
    nnx_vmapped = nnx.vmap(scale_array)

    try:
        result_array = nnx_vmapped(test_array)
        print("nnx.vmap result shape:", result_array.shape)
        print("First element scaled:", result_array[0, 0, 0, :3])
    except Exception as e:
        print("Failed with error:", str(e))


if __name__ == "__main__":
    test_vmap()
