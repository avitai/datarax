# File: src/datarax/samplers/epoch_aware_sampler.py

from dataclasses import dataclass
from typing import Any, Callable

import flax.nnx as nnx
import numpy as np

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule


@dataclass
class EpochAwareSamplerConfig(StructuralConfig):
    """Configuration for EpochAwareSamplerModule.

    Attributes:
        num_records: Total number of records in the dataset (required)
        num_epochs: Number of epochs to iterate (default: 1, -1 for infinite)
        shuffle: Whether to shuffle indices per epoch (default: True)
        seed: Base seed for shuffling (default: 42, only used if shuffle=True)
    """

    # Required parameter
    num_records: int | None = None
    # Optional parameters
    num_epochs: int = 1
    shuffle: bool = True
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set stochastic based on shuffle (override if needed)
        if self.shuffle:
            # Shuffling requires stochastic mode
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "sample")

        # Call parent validation
        super().__post_init__()

        # Validate num_records (required)
        if self.num_records is None:
            raise ValueError("num_records is required")
        if self.num_records <= 0:
            raise ValueError(f"num_records must be positive, got {self.num_records}")

        # Validate num_epochs
        if self.num_epochs < -1 or self.num_epochs == 0:
            raise ValueError(f"num_epochs must be positive or -1 (infinite), got {self.num_epochs}")


# TODO: This module is not used in the current implementation.
# It is a placeholder for future use.
# It is not used in the current implementation.
# It is a placeholder for future use.
class EpochAwareSamplerModule(SamplerModule):
    """Sampler with explicit epoch boundary handling.

    Manages epochs with different shuffling per epoch while
    maintaining state for checkpointing.
    """

    def __init__(
        self,
        config: EpochAwareSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        super().__init__(config, rngs=rngs, name=name)

        # Store config values as Variables for NNX state tracking
        self.num_records = nnx.Variable(config.num_records)
        self.num_epochs = nnx.Variable(config.num_epochs)
        self.shuffle = nnx.Variable(config.shuffle)
        self.base_seed = nnx.Variable(config.seed)

        # Epoch state
        self.current_epoch = nnx.Variable(0)
        self.current_index = nnx.Variable(0)
        self.epoch_indices: nnx.Variable[list[int] | None] = nnx.Variable(None)
        self.epoch_complete_callbacks: nnx.Variable[list[Callable[[int], None]]] = nnx.Variable([])

    def _generate_epoch_indices(self):
        """Generate indices for current epoch."""
        indices = np.arange(self.num_records.get_value())

        if self.shuffle.get_value():
            # Different shuffle per epoch
            epoch_seed = self.base_seed.get_value() + self.current_epoch.get_value()
            rng = np.random.RandomState(epoch_seed)
            rng.shuffle(indices)

        self.epoch_indices.set_value(indices)
        self.current_index.set_value(0)

    def __iter__(self):
        """Initialize iteration."""
        self.current_epoch.set_value(0)
        self._generate_epoch_indices()
        return self

    def __next__(self) -> int:
        """Get next index with epoch management."""
        current_epoch = self.current_epoch.get_value()
        num_epochs = self.num_epochs.get_value()
        if current_epoch >= num_epochs and num_epochs != -1:
            raise StopIteration

        current_index = self.current_index.get_value()
        num_records = self.num_records.get_value()
        if current_index >= num_records:
            # Epoch complete
            self._on_epoch_complete()

            current_epoch += 1
            self.current_epoch.set_value(current_epoch)
            if current_epoch >= num_epochs and num_epochs != -1:
                raise StopIteration

            self._generate_epoch_indices()
            current_index = 0

        epoch_indices = self.epoch_indices.get_value()
        if epoch_indices is None:
            raise ValueError("Epoch indices are not generated yet")

        idx = epoch_indices[current_index]
        self.current_index.set_value(current_index + 1)

        return idx

    def _on_epoch_complete(self):
        """Handle epoch completion."""
        callbacks = self.epoch_complete_callbacks.get_value()
        current_epoch = self.current_epoch.get_value()
        for callback in callbacks:
            callback(current_epoch)

    def add_epoch_callback(self, callback: Callable):
        """Add callback for epoch completion."""
        callbacks = self.epoch_complete_callbacks.get_value()
        callbacks.append(callback)
        self.epoch_complete_callbacks.set_value(callbacks)

    def get_epoch_progress(self) -> dict[str, Any]:
        """Get current epoch progress."""
        current_index = self.current_index.get_value()
        num_records = self.num_records.get_value()
        return {
            "current_epoch": self.current_epoch.get_value(),
            "total_epochs": self.num_epochs.get_value(),
            "current_index": current_index,
            "records_per_epoch": num_records,
            "progress_percent": (current_index / num_records) * 100,
        }
