# File: src/datarax/samplers/sequential_sampler.py

from dataclasses import dataclass
from typing import Any, Iterator

import flax.nnx as nnx

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule


@dataclass
class SequentialSamplerConfig(StructuralConfig):
    """Configuration for SequentialSamplerModule.

    Attributes:
        num_records: Total number of records in the dataset (required)
        num_epochs: Number of epochs to iterate (-1 for infinite)
    """

    # Required parameter (use None as sentinel, validated in __post_init__)
    num_records: int | None = None
    num_epochs: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        # Validate num_records (required)
        if self.num_records is None:
            raise ValueError("num_records is required")
        if self.num_records <= 0:
            raise ValueError(f"num_records must be positive, got {self.num_records}")

        # Validate num_epochs
        if self.num_epochs < -1 or self.num_epochs == 0:
            raise ValueError(f"num_epochs must be positive or -1 (infinite), got {self.num_epochs}")


class SequentialSamplerModule(SamplerModule):
    """Sequential sampler that iterates through indices in order.

    This sampler provides deterministic sequential iteration through
    the dataset without shuffling, useful for evaluation and testing.
    """

    def __init__(
        self,
        config: SequentialSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize sequential sampler with config.

        Args:
            config: Configuration for the sampler.
            rngs: Optional RNGs (not used for sequential sampling).
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Store config values as Variables for NNX state tracking
        self.num_records = nnx.Variable(config.num_records)
        self.num_epochs = nnx.Variable(config.num_epochs)

        # State variables
        self.current_index = nnx.Variable(0)
        self.current_epoch = nnx.Variable(0)
        self._initialized: bool = False
        self._state_restored: bool = False

    def __iter__(self) -> Iterator[int]:
        """Initialize iteration."""
        # Only reset if state wasn't explicitly restored
        if not getattr(self, "_state_restored", False):
            self.current_index.set_value(0)
            self.current_epoch.set_value(0)
        self._initialized = True
        self._state_restored = False  # Reset flag after first iteration
        return self

    def __next__(self) -> int:
        """Get next index in sequence."""
        num_epochs = self.num_epochs.get_value()
        current_epoch = self.current_epoch.get_value()

        # Check if we've completed all epochs
        if num_epochs != -1:
            if current_epoch >= num_epochs:
                raise StopIteration

        current_index = self.current_index.get_value()
        num_records = self.num_records.get_value()

        # Check if we need to start a new epoch
        if current_index >= num_records:
            current_epoch += 1
            self.current_epoch.set_value(current_epoch)
            current_index = 0
            self.current_index.set_value(0)

            # Check epoch limit again
            if num_epochs != -1:
                if current_epoch >= num_epochs:
                    raise StopIteration

        # Return current index and increment
        idx = current_index
        self.current_index.set_value(current_index + 1)

        return idx

    def __len__(self) -> int:
        """Return total number of indices across all epochs."""
        num_epochs = self.num_epochs.get_value()
        if num_epochs == -1:
            raise ValueError("Cannot determine length for infinite epochs")
        return self.num_records.get_value() * num_epochs

    def get_state(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        state = super().get_state()
        state.update(
            {
                "current_index": self.current_index.get_value(),
                "current_epoch": self.current_epoch.get_value(),
                "num_records": self.num_records.get_value(),
                "num_epochs": self.num_epochs.get_value(),
            }
        )
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        super().set_state(state)
        if "current_index" in state:
            self.current_index.set_value(state["current_index"])
        if "current_epoch" in state:
            self.current_epoch.set_value(state["current_epoch"])
        if "num_records" in state:
            self.num_records.set_value(state["num_records"])
        if "num_epochs" in state:
            self.num_epochs.set_value(state["num_epochs"])
        # Mark as initialized and state restored to prevent __iter__ from resetting
        self._initialized = True
        self._state_restored = True
