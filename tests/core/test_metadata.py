"""Tests for the Metadata module."""

import jax
import jax.numpy as jnp

from datarax.core.metadata import (
    Metadata,
    create_metadata,
    split_rng_tree,
    batch_metadata,
    update_metadata_batch,
)


class TestMetadataBasics:
    """Test basic Metadata functionality."""

    def test_creation_default(self):
        """Test default metadata creation."""
        meta = Metadata()

        assert meta.index == 0
        assert meta.epoch == 0
        assert meta.global_step == 0
        assert meta.batch_idx is None
        assert meta.shard_id is None
        assert meta.rng_key is None
        assert meta.record_key is None
        assert meta.source_info is None

    def test_creation_with_values(self):
        """Test creating metadata with specific values."""
        metadata = Metadata(
            index=10,
            epoch=5,
            global_step=100,
            batch_idx=2,
            shard_id=1,
            key="record_123",
            source_info={"path": "/tmp/data"},
        )

        assert metadata.index == 10
        assert metadata.epoch == 5
        assert metadata.global_step == 100
        assert metadata.batch_idx == 2
        assert metadata.shard_id == 1
        assert metadata.record_key == "record_123"
        assert metadata.source_info == {"path": "/tmp/data"}

    def test_replace_method(self):
        """Test creating new instance with replaced values."""
        metadata = Metadata(index=1, key="k1")
        new_meta = metadata.replace(index=2, key="k2")

        # Original unchanged
        assert metadata.index == 1
        assert metadata.record_key == "k1"

        # New updated
        assert new_meta.index == 2
        assert new_meta.record_key == "k2"

    def test_increment_methods(self):
        """Test helper methods for incrementing counters."""
        metadata = Metadata(
            index=10,
            epoch=5,
            global_step=100,
            batch_idx=2,
            key="k1",
        )

        # Increment step
        m1 = metadata.increment_step()
        assert m1.global_step == 101
        assert m1.index == 10  # Unchanged

        # Increment epoch
        m2 = metadata.increment_epoch()
        assert m2.epoch == 6
        assert m2.batch_idx == 0

        # Increment batch
        m3 = metadata.increment_batch()
        assert m3.batch_idx == 3

    def test_with_shard(self):
        """Test adding shard ID."""
        metadata = Metadata(index=1)
        m1 = metadata.with_shard(5)
        assert m1.shard_id == 5

    def test_to_dict_from_dict(self):
        """Test dictionary serialization."""
        original = Metadata(
            index=10,
            key="rec1",
            source_info={"a": 1},
        )

        # Convert to dict
        data = original.to_dict()
        assert data["index"] == 10
        assert (
            data["record_key"] == "rec1"
        )  # Property access should still work with 'record_key' in dict
        assert data["source_info"] == {"a": 1}

        # Create from dict
        # Note: from_dict handles mapping 'record_key' -> 'key'
        reconstructed = Metadata.from_dict(data)
        assert reconstructed.index == 10
        assert reconstructed.record_key == "rec1"
        assert reconstructed.source_info == {"a": 1}

    def test_merge(self):
        """Test merging metadata."""
        meta1 = Metadata(index=1, epoch=0, global_step=10)
        meta2 = Metadata(epoch=2, global_step=20, batch_idx=5)

        merged = meta1.merge(meta2)

        assert merged.index == 1  # From meta1 (meta2 was 0)
        assert merged.epoch == 2  # From meta2
        assert merged.global_step == 20  # From meta2
        assert merged.batch_idx == 5  # From meta2

    def test_merge_with_none(self):
        """Test merging with None returns self."""
        meta = Metadata(index=1, epoch=2)
        merged = meta.merge(None)

        assert merged.index == meta.index
        assert merged.epoch == meta.epoch


class TestMetadataRNG:
    """Test RNG-related functionality."""

    def test_split_rng(self):
        """Test RNG key splitting."""
        # With RNG key
        meta = create_metadata(seed=42)
        keys = meta.split_rng(3)

        assert len(keys) == 3
        assert all(k is not None for k in keys)
        # Keys should be different
        assert not jnp.array_equal(keys[0], keys[1])
        assert not jnp.array_equal(keys[1], keys[2])

        # Without RNG key
        meta2 = Metadata()
        keys2 = meta2.split_rng(3)

        assert len(keys2) == 3
        assert all(k is None for k in keys2)

    def test_next_rng(self):
        """Test getting next RNG state."""
        meta = create_metadata(seed=42)
        original_key = meta.rng_key

        meta2 = meta.next_rng()

        assert meta2.rng_key is not None
        assert not jnp.array_equal(original_key, meta2.rng_key)
        assert jnp.array_equal(meta.rng_key, original_key)  # Original unchanged

    def test_next_rng_without_key(self):
        """Test next_rng without initial key returns self."""
        meta = Metadata()
        meta2 = meta.next_rng()

        assert meta2.rng_key is None
        assert meta2.index == meta.index  # Same instance behavior

    def test_split_rng_tree(self):
        """Test RNG tree splitting with named keys."""
        # With RNG
        meta = create_metadata(seed=42)
        result = split_rng_tree(meta, 3)

        assert "a" in result
        assert "b" in result
        assert "key_2" in result
        assert result["a"] is not None
        assert result["b"] is not None
        assert result["key_2"] is not None

        # Without RNG
        meta2 = Metadata()
        result2 = split_rng_tree(meta2, 2)

        assert len(result2) == 2
        assert all(v is None for v in result2.values())


class TestMetadataPytree:
    """Test pytree behavior."""

    def test_is_pytree(self):
        """Test that Metadata is a valid PyTree."""
        metadata = Metadata(index=1, key="k1", rng_key=jax.random.PRNGKey(0))

        leaves, treedef = jax.tree_util.tree_flatten(metadata)

        # Should have leaves for index, epoch, global_step, batch_idx, shard_id,
        # rng_key, _encoded_key
        # Note: Depending on None fields, count might vary.
        # But we expect at least the non-None numeric ones + array key + rng key
        assert len(leaves) >= 3

        # Reconstruct
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed.index == 1
        assert reconstructed.record_key == "k1"

    def test_tree_map(self):
        """Test working with tree_map."""
        metadata = Metadata(index=1, global_step=10)

        # Increment all numeric leaves
        def increment(x):
            if isinstance(x, int):
                return x + 1
            return x

        new_meta = jax.tree.map(increment, metadata)
        assert new_meta.index == 2
        assert new_meta.global_step == 11

    def test_static_fields_preserved(self):
        """Test that static fields are preserved through flattening."""
        source_info = {"dataset": "train", "version": 1}
        metadata = Metadata(index=1, source_info=source_info)

        leaves, treedef = jax.tree_util.tree_flatten(metadata)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert reconstructed.source_info == source_info

    def test_jit_compilation(self):
        """Test using Metadata in JIT function."""
        metadata = Metadata(index=10, global_step=100)

        @jax.jit
        def get_next_step(m):
            return m.global_step + 1

        result = get_next_step(metadata)
        assert result == 101

    def test_static_fields_no_recompilation(self):
        """Test that changing static fields doesn't trigger recompilation."""
        # Note: metadata keys are now dynamic so they don't trigger recompilation either!
        # Static fields only include source_info now.

        metadata_1 = Metadata(index=1, key="k1", source_info={"a": 1})
        metadata_2 = Metadata(index=1, key="k1", source_info={"a": 1})  # Same static info

        # Check treedefs
        _, treedef1 = jax.tree_util.tree_flatten(metadata_1)
        _, treedef2 = jax.tree_util.tree_flatten(metadata_2)
        assert treedef1 == treedef2

        # Change static field -> New treedef
        metadata_3 = Metadata(index=1, key="k1", source_info={"b": 2})
        _, treedef3 = jax.tree_util.tree_flatten(metadata_3)
        assert treedef1 != treedef3

        # Change dynamic field (key) -> SAME treedef (because encoded key is array leaf)
        metadata_4 = Metadata(index=1, key="new_key", source_info={"a": 1})
        _, treedef4 = jax.tree_util.tree_flatten(metadata_4)
        assert treedef1 == treedef4

    def test_rng_in_jit(self):
        """Test RNG operations in JIT."""

        @jax.jit
        def split_key(meta: Metadata) -> Metadata:
            return meta.next_rng()

        meta = create_metadata(seed=42)
        result = split_key(meta)

        assert result.rng_key is not None
        assert not jnp.array_equal(result.rng_key, meta.rng_key)


class TestBatchMetadata:
    """Test batch metadata operations."""

    def test_batch_empty(self):
        """Test batching empty list."""
        result = batch_metadata([])

        assert result.index == 0
        assert result.epoch == 0
        assert result.global_step == 0

    def test_batch_single(self):
        """Test batching single metadata."""
        meta = Metadata(index=5, epoch=2, global_step=100)
        result = batch_metadata([meta])

        assert result.index == 5
        assert result.epoch == 2
        assert result.global_step == 100

    def test_batch_multiple(self):
        """Test batching multiple metadata."""
        metas = [
            Metadata(index=0, epoch=1, global_step=10),
            Metadata(index=0, epoch=2, global_step=20),
            Metadata(index=0, epoch=1, global_step=15),
        ]

        result = batch_metadata(metas)

        assert result.epoch == 2  # Max
        assert result.global_step == 20  # Max
        assert result.index == 0  # From first

    def test_batch_with_optional_fields(self):
        """Test batching with optional fields."""
        metas = [
            Metadata(batch_idx=None, shard_id=None),
            Metadata(batch_idx=5, shard_id=None),
            Metadata(batch_idx=None, shard_id=2),
        ]

        result = batch_metadata(metas)

        assert result.batch_idx == 5  # First non-None
        assert result.shard_id == 2  # First non-None

    def test_update_metadata_batch(self):
        """Test batch update function."""
        meta = Metadata(global_step=100, batch_idx=5)

        updated = update_metadata_batch(meta, batch_size=32)

        assert updated.global_step == 132
        assert updated.batch_idx == 6

    def test_update_metadata_batch_jit(self):
        """Test batch update is JIT compilable."""
        meta = Metadata(global_step=0, batch_idx=None)

        # Should be JIT compiled by default
        updated = update_metadata_batch(meta, 10)

        assert updated.global_step == 10
        assert updated.batch_idx == 1  # 0 + 1


class TestCreateMetadata:
    """Test metadata creation helper."""

    def test_create_with_seed(self):
        """Test creation with seed initializes RNG."""
        meta = create_metadata(seed=42)

        assert meta.rng_key is not None
        # jax.random.key() returns scalar-shaped key, PRNGKey returns (2,)
        # Accept either for compatibility with different JAX versions
        assert meta.rng_key.shape in ((), (2,))

    def test_create_with_explicit_rng(self):
        """Test creation with explicit RNG key."""
        key = jax.random.key(123)
        meta = create_metadata(rng_key=key)

        assert jnp.array_equal(meta.rng_key, key)

    def test_create_with_all_fields(self):
        """Test creation with all fields specified."""
        meta = create_metadata(
            index=10,
            epoch=2,
            global_step=100,
            batch_idx=5,
            shard_id=0,
            record_key="test",
            source_info={"data": "value"},
            seed=42,
        )

        assert meta.index == 10
        assert meta.epoch == 2
        assert meta.global_step == 100
        assert meta.batch_idx == 5
        assert meta.shard_id == 0
        assert meta.record_key == "test"
        assert meta.source_info == {"data": "value"}
        assert meta.rng_key is not None
