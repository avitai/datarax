# File: tests/samplers/test_epoch_aware_sampler.py

"""Tests for EpochAwareSamplerModule.

Comprehensive test suite for epoch-based sampling with shuffle
and callback functionality.
"""

import flax.nnx as nnx
import pytest

from datarax.samplers.epoch_aware_sampler import (
    EpochAwareSamplerConfig,
    EpochAwareSamplerModule,
)


class TestEpochAwareSamplerInitialization:
    """Tests for EpochAwareSamplerModule initialization and state."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        config = EpochAwareSamplerConfig(num_records=10)
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        assert sampler.num_records.get_value() == 10
        assert sampler.num_epochs.get_value() == 1
        assert sampler.shuffle.get_value() is True
        assert sampler.base_seed.get_value() == 42
        assert sampler.current_epoch.get_value() == 0
        assert sampler.current_index.get_value() == 0
        assert sampler.epoch_indices.get_value() is None
        assert sampler.epoch_complete_callbacks.get_value() == []

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        config = EpochAwareSamplerConfig(
            num_records=100,
            num_epochs=5,
            shuffle=False,
            seed=123,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        assert sampler.num_records.get_value() == 100
        assert sampler.num_epochs.get_value() == 5
        assert sampler.shuffle.get_value() is False
        assert sampler.base_seed.get_value() == 123

    def test_initialization_with_rngs(self):
        """Test initialization with custom rngs."""
        rngs = nnx.Rngs(default=0)
        config = EpochAwareSamplerConfig(num_records=10)
        sampler = EpochAwareSamplerModule(config, rngs=rngs)

        assert sampler.num_records.get_value() == 10
        assert sampler.rngs is not None


class TestEpochAwareSamplerIteration:
    """Tests for iteration protocol."""

    def test_single_epoch_no_shuffle(self):
        """Test iteration over a single epoch without shuffling."""
        config = EpochAwareSamplerConfig(
            num_records=5,
            num_epochs=1,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        # Manually collect indices without calling __len__
        indices = []
        for idx in sampler:
            indices.append(idx)

        assert len(indices) == 5
        assert indices == [0, 1, 2, 3, 4]
        assert sampler.current_epoch.get_value() == 1

    def test_single_epoch_with_shuffle(self):
        """Test iteration over a single epoch with shuffling."""
        config = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=1,
            shuffle=True,
            seed=42,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        indices = []
        for idx in sampler:
            indices.append(idx)

        assert len(indices) == 10
        # Should contain all indices
        assert sorted(indices) == list(range(10))
        # Should be shuffled (not sequential)
        assert indices != list(range(10))

    def test_shuffle_deterministic(self):
        """Test that shuffle is deterministic with same seed."""
        config1 = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=1,
            shuffle=True,
            seed=42,
        )
        sampler1 = EpochAwareSamplerModule(config1, rngs=nnx.Rngs(0))
        indices1 = [idx for idx in sampler1]

        config2 = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=1,
            shuffle=True,
            seed=42,
        )
        sampler2 = EpochAwareSamplerModule(config2, rngs=nnx.Rngs(0))
        indices2 = [idx for idx in sampler2]

        assert indices1 == indices2

    def test_shuffle_different_per_seed(self):
        """Test that different seeds produce different shuffles."""
        config1 = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=1,
            shuffle=True,
            seed=42,
        )
        sampler1 = EpochAwareSamplerModule(config1, rngs=nnx.Rngs(0))
        indices1 = [idx for idx in sampler1]

        config2 = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=1,
            shuffle=True,
            seed=99,
        )
        sampler2 = EpochAwareSamplerModule(config2, rngs=nnx.Rngs(0))
        indices2 = [idx for idx in sampler2]

        assert indices1 != indices2


class TestEpochAwareSamplerMultiEpoch:
    """Tests for multi-epoch iteration."""

    def test_multiple_epochs_no_shuffle(self):
        """Test iteration over multiple epochs without shuffling."""
        config = EpochAwareSamplerConfig(
            num_records=3,
            num_epochs=2,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        indices = [idx for idx in sampler]

        assert len(indices) == 6  # 3 records × 2 epochs
        assert indices == [0, 1, 2, 0, 1, 2]
        assert sampler.current_epoch.get_value() == 2

    def test_multiple_epochs_with_shuffle(self):
        """Test iteration over multiple epochs with shuffling."""
        config = EpochAwareSamplerConfig(
            num_records=5,
            num_epochs=3,
            shuffle=True,
            seed=42,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        indices = [idx for idx in sampler]

        assert len(indices) == 15  # 5 records × 3 epochs
        # Each epoch should contain all indices (shuffled)
        epoch1 = indices[0:5]
        epoch2 = indices[5:10]
        epoch3 = indices[10:15]

        assert sorted(epoch1) == list(range(5))
        assert sorted(epoch2) == list(range(5))
        assert sorted(epoch3) == list(range(5))

    def test_shuffle_different_per_epoch(self):
        """Test that shuffle order differs between epochs."""
        config = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=2,
            shuffle=True,
            seed=42,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        indices = [idx for idx in sampler]

        epoch1 = indices[0:10]
        epoch2 = indices[10:20]

        # Both epochs should be shuffled differently
        assert sorted(epoch1) == list(range(10))
        assert sorted(epoch2) == list(range(10))
        assert epoch1 != epoch2

    def test_infinite_epochs(self):
        """Test infinite epochs with num_epochs=-1."""
        config = EpochAwareSamplerConfig(
            num_records=3,
            num_epochs=-1,  # Infinite epochs
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        # Manually iterate a limited number
        indices = []
        iterator = iter(sampler)
        for _ in range(10):  # Get 10 samples across multiple epochs
            indices.append(next(iterator))

        # Should cycle through records
        assert len(indices) == 10
        # First 3 complete cycles
        assert indices[:9] == [0, 1, 2, 0, 1, 2, 0, 1, 2]


class TestEpochAwareSamplerCallbacks:
    """Tests for epoch completion callbacks."""

    def test_epoch_callback_invoked(self):
        """Test that epoch completion callbacks are invoked."""
        callback_epochs = []

        def track_epoch(epoch: int):
            callback_epochs.append(epoch)

        config = EpochAwareSamplerConfig(
            num_records=3,
            num_epochs=2,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))
        sampler.add_epoch_callback(track_epoch)

        # Consume all samples
        for _ in sampler:
            pass

        # Should be called once per completed epoch (before advancing)
        assert callback_epochs == [0, 1]

    def test_multiple_callbacks(self):
        """Test that multiple callbacks are all invoked."""
        callback1_epochs = []
        callback2_epochs = []

        def track_epoch1(epoch: int):
            callback1_epochs.append(epoch)

        def track_epoch2(epoch: int):
            callback2_epochs.append(epoch)

        config = EpochAwareSamplerConfig(
            num_records=2,
            num_epochs=2,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))
        sampler.add_epoch_callback(track_epoch1)
        sampler.add_epoch_callback(track_epoch2)

        # Consume all samples
        for _ in sampler:
            pass

        assert callback1_epochs == [0, 1]
        assert callback2_epochs == [0, 1]


class TestEpochAwareSamplerProgress:
    """Tests for progress tracking functionality."""

    def test_progress_at_start(self):
        """Test progress tracking at start of iteration."""
        config = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=3,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        # Before iteration
        progress = sampler.get_epoch_progress()

        assert progress["current_epoch"] == 0
        assert progress["total_epochs"] == 3
        assert progress["current_index"] == 0
        assert progress["records_per_epoch"] == 10
        assert progress["progress_percent"] == 0.0

    def test_progress_during_epoch(self):
        """Test progress tracking during epoch."""
        config = EpochAwareSamplerConfig(
            num_records=10,
            num_epochs=2,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        iterator = iter(sampler)

        # Consume 5 samples
        for _ in range(5):
            next(iterator)

        progress = sampler.get_epoch_progress()

        assert progress["current_epoch"] == 0
        assert progress["current_index"] == 5
        assert progress["progress_percent"] == 50.0

    def test_progress_between_epochs(self):
        """Test progress tracking between epochs."""
        config = EpochAwareSamplerConfig(
            num_records=5,
            num_epochs=3,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        iterator = iter(sampler)

        # Consume first epoch + 2 samples of second epoch
        for _ in range(7):
            next(iterator)

        progress = sampler.get_epoch_progress()

        assert progress["current_epoch"] == 1  # Second epoch (0-indexed)
        assert progress["current_index"] == 2
        assert progress["progress_percent"] == 40.0


class TestEpochAwareSamplerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_record(self):
        """Test with single record."""
        config = EpochAwareSamplerConfig(
            num_records=1,
            num_epochs=2,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        indices = [idx for idx in sampler]

        assert indices == [0, 0]

    def test_stop_iteration(self):
        """Test that StopIteration is raised after all epochs."""
        config = EpochAwareSamplerConfig(
            num_records=2,
            num_epochs=1,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        iterator = iter(sampler)
        next(iterator)  # 0
        next(iterator)  # 1

        # Should raise StopIteration on next call
        with pytest.raises(StopIteration):
            next(iterator)

    def test_reinitialize_iteration(self):
        """Test that calling iter() reinitializes state."""
        config = EpochAwareSamplerConfig(
            num_records=3,
            num_epochs=1,
            shuffle=False,
        )
        sampler = EpochAwareSamplerModule(config, rngs=nnx.Rngs(0))

        # First iteration
        indices1 = [idx for idx in sampler]

        # Second iteration (should reinitialize)
        indices2 = [idx for idx in sampler]

        assert indices1 == indices2 == [0, 1, 2]
