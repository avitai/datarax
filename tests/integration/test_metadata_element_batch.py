"""Integration tests for metadata and element_batch modules.

Tests the interaction between Metadata and Element/Batch to ensure
they work together properly in realistic scenarios.

Updated to use current API (removed MetadataTracker, transform_batch references).
"""

import jax
import jax.numpy as jnp
import pytest

from datarax.core.element_batch import Batch, BatchOps, Element
from datarax.core.metadata import batch_metadata, Metadata


class TestMetadataElementIntegration:
    """Test integration between Metadata and Element."""

    def test_element_with_full_metadata(self):
        """Test element with complete metadata."""
        meta = Metadata(
            index=100,
            epoch=2,
            global_step=5000,
            batch_idx=10,
            shard_id=0,
            key="file_05_rec_100",
            rng_key=jax.random.key(42),
            source_info={"source": "s3://bucket/data"},
        )

        element = Element(data={"x": jnp.zeros((10,))}, metadata=meta)

        assert element.metadata is not None
        assert element.metadata.index == 100
        assert element.metadata.entry_key == "file_05_rec_100"
        assert element.metadata.rng_key is not None

        # Test transformation preserves metadata (except specific updates if any)
        # Note: transform creates new element with same metadata
        transformed = element.transform(lambda x: x + 1)
        assert transformed.metadata is not None
        assert transformed.metadata.index == 100
        assert transformed.metadata.rng_key is not None
        assert meta.rng_key is not None
        assert jnp.array_equal(transformed.metadata.rng_key, meta.rng_key)

    def test_element_metadata_in_jax_transforms(self):
        """Test Element with Metadata in JAX transformations."""

        @jax.jit
        def process_element(elem: Element) -> Element:
            # Use metadata's RNG if available
            if elem.metadata and elem.metadata.rng_key is not None:
                noise = jax.random.normal(elem.metadata.rng_key, elem.data["x"].shape)
                new_data = {"x": elem.data["x"] + noise * 0.01}
                return elem.replace(data=new_data)
            return elem

        metadata = Metadata(rng_key=jax.random.key(0))
        elem = Element(data={"x": jnp.zeros(5)}, metadata=metadata)

        processed = process_element(elem)

        # Should have added noise
        assert not jnp.allclose(processed.data["x"], jnp.zeros(5))
        # Metadata preserved
        assert processed.metadata is not None
        assert processed.metadata.rng_key is not None

    def test_element_metadata_update_workflow(self):
        """Test realistic workflow of updating element metadata."""
        # Create element with initial metadata
        elem = Element(data={"x": jnp.array([1, 2, 3])}, metadata=Metadata(index=0, epoch=0))

        # Process and update metadata using current API
        for step in range(3):
            # Update metadata for next step using current API methods
            assert elem.metadata is not None
            new_metadata = elem.metadata.increment_step().replace(index=elem.metadata.index + 1)
            elem = elem.with_metadata(new_metadata)

            # Transform data
            elem = elem.transform(lambda x: x + 1)

        # Check final state
        assert elem.metadata is not None
        assert elem.metadata.index == 3
        assert elem.metadata.global_step == 3
        assert jnp.array_equal(elem.data["x"], jnp.array([4, 5, 6]))


class TestMetadataBatchIntegration:
    """Test integration between Metadata and Batch."""

    def test_batch_with_element_metadata(self):
        """Test Batch creation from Elements with Metadata."""
        # Create elements with metadata
        elements = [
            Element(
                data={"x": jnp.ones(3) * i},
                state={"id": i},
                metadata=Metadata(index=i, epoch=0, rng_key=jax.random.key(i)),
            )
            for i in range(4)
        ]

        batch = Batch(elements)

        # Check elements preserve metadata
        for i in range(4):
            elem = batch.get_element(i)
            assert elem.metadata is not None
            assert elem.metadata.index == i
            assert elem.metadata.rng_key is not None

    def test_batch_level_metadata(self):
        """Test batch with its own metadata."""
        batch = Batch(
            [
                Element(
                    data={"x": jnp.zeros((2,))},
                    metadata=Metadata(index=i, key=f"elem_{i}"),
                )
                for i in range(3)
            ]
        )

        batch_meta = Metadata(
            index=5,
            key="batch_05",  # Changed from record_key
            batch_idx=5,
        )

        batch.set_batch_metadata(batch_meta)

        # Retrieve and verify
        retrieved = batch.get_batch_metadata()
        assert retrieved is not None
        assert retrieved.index == 5
        assert retrieved.entry_key == "batch_05"
        elem0_meta = batch.get_element(0).metadata
        assert elem0_meta is not None
        assert elem0_meta.index == 0

    def test_batch_metadata_aggregation(self):
        """Test aggregating metadata from elements to batch."""
        # Create elements with RNG keys
        elements = [
            Element(
                data={"x": jnp.ones(2)},
                metadata=Metadata(index=i, rng_key=jax.random.key(i), epoch=1, global_step=100 + i),
            )
            for i in range(4)
        ]

        batch = Batch(elements)

        # Get all element metadata
        metadata_list = [batch.get_element(i).metadata for i in range(4)]

        # Create batch metadata from elements (current API - no batch_size arg)
        # All elements have metadata, so filter out None for type safety
        valid_metadata = [m for m in metadata_list if m is not None]
        batch_meta = batch_metadata(valid_metadata)
        batch.set_batch_metadata(batch_meta)

        batch_meta_result = batch.get_batch_metadata()
        assert batch_meta_result is not None
        assert batch_meta_result.epoch == 1
        assert batch_meta_result.rng_key is not None


class TestDataPipelineScenarios:
    """Test realistic data pipeline scenarios."""

    def test_data_augmentation_with_metadata_rng(self):
        """Test data augmentation using metadata RNG keys."""

        def augment_with_metadata(elem: Element) -> Element:
            """Augment element using its metadata RNG."""
            if elem.metadata and elem.metadata.rng_key is not None:
                # Split key for different augmentations
                keys = elem.metadata.split_rng(2)
                assert keys[0] is not None
                assert keys[1] is not None

                # Add random noise
                noise = jax.random.normal(keys[0], elem.data["image"].shape)
                augmented_image = elem.data["image"] + noise * 0.1

                # Random flip
                flip = jax.random.uniform(keys[1]) > 0.5
                augmented_image = jax.lax.cond(
                    flip, lambda x: jnp.flip(x, axis=1), lambda x: x, augmented_image
                )

                return elem.replace(data={"image": augmented_image})
            return elem

        # Create batch with metadata
        elements = [
            Element(
                data={"image": jnp.ones((8, 8))},
                metadata=Metadata(index=i, rng_key=jax.random.key(i)),
            )
            for i in range(4)
        ]

        batch = Batch(elements)

        # Apply augmentation
        augmented_batch = Batch(
            [augment_with_metadata(batch.get_element(i)) for i in range(batch.batch_size)]
        )

        # Check augmentation applied
        for i in range(4):
            original = batch.get_element(i).data["image"]
            augmented = augmented_batch.get_element(i).data["image"]
            assert not jnp.allclose(original, augmented)

    def test_distributed_batch_processing(self):
        """Test distributed processing with shard metadata."""
        # Create batch with shard information
        elements = [
            Element(data={"x": jnp.ones(10) * i}, metadata=Metadata(index=i, shard_id=0))
            for i in range(8)
        ]

        batch = Batch(elements)

        # Split for 4 devices
        device_batches = batch.split_for_devices(4)

        # Update shard IDs for each device using current API
        for device_id, device_batch in enumerate(device_batches):
            # Update metadata for this shard
            for i in range(device_batch.batch_size):
                elem = device_batch.get_element(i)
                assert elem.metadata is not None
                # Use with_shard() instead of for_shard()
                new_metadata = elem.metadata.with_shard(device_id)
                assert new_metadata.shard_id == device_id

    def test_filtering_with_metadata_preservation(self):
        """Test that filtering preserves metadata correctly."""
        # Create elements with varying metadata
        elements = []
        for i in range(6):
            metadata = Metadata(
                index=i,
                epoch=i // 3,  # 0,0,0,1,1,1
                rng_key=jax.random.key(i),
            )
            elem = Element(
                data={"x": jnp.array([i])}, state={"valid": i % 2 == 0}, metadata=metadata
            )
            elements.append(elem)

        batch = Batch(elements)

        # Filter for valid elements
        mask = jnp.array([e.state.get("valid", False) for e in elements])
        filtered = BatchOps.select_batch_rows(batch, mask)

        # Check metadata preserved correctly
        assert filtered.batch_size == 3
        meta0 = filtered.get_element(0).metadata
        meta1 = filtered.get_element(1).metadata
        meta2 = filtered.get_element(2).metadata
        assert meta0 is not None
        assert meta1 is not None
        assert meta2 is not None
        assert meta0.index == 0
        assert meta1.index == 2
        assert meta2.index == 4

        # Check epochs preserved
        assert meta0.epoch == 0
        assert meta2.epoch == 1


class TestEdgeCasesIntegration:
    """Test edge cases in integration."""

    def test_none_metadata_handling(self):
        """Test handling of None metadata in batch."""
        elements = [
            Element(data={"x": jnp.ones(2)}, metadata=Metadata(index=0)),
            Element(data={"x": jnp.ones(2)}, metadata=None),
            Element(data={"x": jnp.ones(2)}, metadata=Metadata(index=2)),
        ]

        batch = Batch(elements)

        # Should handle mixed metadata
        meta0 = batch.get_element(0).metadata
        assert meta0 is not None
        assert meta0.index == 0
        assert batch.get_element(1).metadata is None
        meta2 = batch.get_element(2).metadata
        assert meta2 is not None
        assert meta2.index == 2

    def test_metadata_with_large_source_info(self):
        """Test metadata with large source_info dictionary."""
        large_info = {f"key_{i}": f"value_{i}" for i in range(1000)}

        metadata = Metadata(index=0, source_info=large_info)

        elem = Element(data={"x": jnp.ones(5)}, metadata=metadata)
        batch = Batch([elem])

        # Should handle large metadata
        retrieved = batch.get_element(0).metadata
        assert retrieved is not None
        assert retrieved.source_info is not None
        assert len(retrieved.source_info) == 1000
        assert retrieved.source_info["key_500"] == "value_500"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
