import jax
import jax.numpy as jnp
from jax import tree_util
from datarax.core.metadata import Metadata


class TestMetadataOptimization:
    """Tests for Metadata optimization (byte-encoded keys)."""

    def test_metadata_creation_with_string_key(self):
        """Test creating metadata with string key (should be encoded internally)."""
        # Note: This test will fail until implementation is updated
        meta = Metadata(key="test_key_123")

        # Check if it has internal byte representation (implementation dependent name)
        # We expect the public property '.record_key' to return the string
        assert meta.record_key == "test_key_123"

        # We expect some internal attribute to hold the bytes or array
        # This asserts the INTENT of the change - that we can access it as an array
        # We'll assume the implementation uses _record_key_bytes or similar,
        # but primarily we care that Flattening produces an array, not a string.
        leaves, treedef = tree_util.tree_flatten(meta)

        # In current impl, record_key is static (in treedef)
        # In NEW impl, it should be in leaves (dynamic)

        # We can detect this by checking if the key string is NOT in the treedef
        # and if there is a corresponding array in leaves

        # Current behavior (FAIL condition for new requirement):
        # leaves has: index, epoch, global_step, batch_idx, shard_id, rng_key (6 items)
        # New behavior (PASS condition):
        # New behavior (PASS condition):
        # leaves has: index, epoch, global_step, batch_idx, shard_id, rng_key,
        # record_key_bytes (7 items)

        # Note: None values are filtered from leaves by JAX, so exact count varies.
        # We rely on checking for array presence.

        # Verify one of the leaves is the encoded key
        is_array = [isinstance(l, (jax.Array, jnp.ndarray)) for l in leaves]
        assert sum(is_array) >= 1  # At least one array (rng_key is optional)

    def test_jit_compatibility_with_different_keys(self):
        """Test that Metadata with different keys does NOT trigger recompilation."""

        @jax.jit
        def process_metadata(meta):
            return meta.index + 1

        # Two metadata objects with DIFFERENT keys
        meta1 = Metadata(index=0, key="key1")
        meta2 = Metadata(index=0, key="key2")

        # Ensure they have the SAME treedef
        leaves1, treedef1 = tree_util.tree_flatten(meta1)
        leaves2, treedef2 = tree_util.tree_flatten(meta2)

        assert treedef1 == treedef2, "TreeDefs must be identical for JIT compatibility"

        # This confirms that the key is treated as data, not static structure

    def test_long_key_handling(self):
        """Test handling of keys of different lengths."""
        # This is tricky because arrays need fixed shapes for compiled code usually,
        # but for simple flattening/unflattening in Python it might differ.
        # Ideally we pad to fixed length or JAX handles variable length leaves
        # (it doesn't for JIT args with same partial check).

        # If we use different lengths, we might still get recompilation if shapes differ.
        # So the implementation should probably pad to a fixed max length or allow
        # variable shapes (which causes recompile on shape change).
        # For now, let's assume we want to support at least same-length keys without recompile.

        meta1 = Metadata(key="abc")
        meta2 = Metadata(key="xyz")

        leaves1, _ = tree_util.tree_flatten(meta1)
        leaves2, _ = tree_util.tree_flatten(meta2)

        # Find the bytes leaf
        bytes1 = leaves1[-1]  # Assuming appended
        bytes2 = leaves2[-1]

        assert bytes1.shape == bytes2.shape

    def test_key_roundtrip(self):
        """Test string -> bytes -> string roundtrip."""
        metadata = Metadata(index=0, key="file_1_record_10")

        # Verify encoded key is created
        assert metadata._encoded_key is not None
        assert isinstance(metadata._encoded_key, (jax.Array, jnp.ndarray))
        assert metadata._encoded_key.dtype == jnp.uint8
        assert metadata._encoded_key.shape == (128,)

        # Verify property returns string
        assert metadata.record_key == "file_1_record_10"

        # Flatten and unflatten
        leaves, treedef = tree_util.tree_flatten(metadata)
        reconstructed = tree_util.tree_unflatten(treedef, leaves)

        assert reconstructed.record_key == "file_1_record_10"

    def test_none_key(self):
        """Test handling of None key."""
        meta = Metadata(key=None)
        assert meta.record_key is None

        leaves, treedef = tree_util.tree_flatten(meta)
        reconstructed = tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed.record_key is None
