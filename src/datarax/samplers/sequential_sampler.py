"""Sequential sampler for iterating over dataset indices in order."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule
from datarax.samplers._iteration import (
    consume_epoch_step_index,
    read_epoch_step,
    total_epoch_length,
)
from datarax.samplers._validation import validate_sampler_bounds
from datarax.utils.state import (
    build_state_with_iteration_fields,
    restore_iteration_variables,
    restore_optional_variable_fields,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequentialSamplerConfig(StructuralConfig):
    """Configuration for SequentialSamplerModule.

    Attributes:
        num_records: Total number of records in the dataset (required)
        num_epochs: Number of epochs to iterate (-1 for infinite)
    """

    # Required parameter (use None as sentinel, validated in __post_init__)
    num_records: int | None = None
    num_epochs: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        super().__post_init__()
        validate_sampler_bounds(self.num_records, self.num_epochs)


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
    ) -> None:
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
        self._is_state_restored: bool = False

    def __iter__(self) -> Iterator[int]:
        """Initialize iteration."""
        # Only reset if state wasn't explicitly restored
        if not getattr(self, "_is_state_restored", False):
            self.current_index.set_value(0)
            self.current_epoch.set_value(0)
        self._initialized = True
        self._is_state_restored = False  # Reset flag after first iteration
        return self

    def __next__(self) -> int:
        """Get next index in sequence."""
        return consume_epoch_step_index(
            epoch_step=read_epoch_step(
                current_epoch=self.current_epoch.get_value,
                num_epochs=self.num_epochs.get_value,
                current_index=self.current_index.get_value,
                num_records=self.num_records.get_value,
            ),
            current_epoch=self.current_epoch.set_value,
            current_index=self.current_index.set_value,
        )

    def __len__(self) -> int:
        """Return total number of indices across all epochs."""
        return total_epoch_length(
            self.num_records.get_value(),
            self.num_epochs.get_value(),
        )

    def get_state(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        base_state = super().get_state()
        base_state["num_records"] = self.num_records.get_value()
        base_state["num_epochs"] = self.num_epochs.get_value()
        return build_state_with_iteration_fields(
            base_state,
            current_index=self.current_index.get_value(),
            current_epoch=self.current_epoch.get_value(),
        )

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        super().set_state(state)
        restore_iteration_variables(
            state,
            current_index=self.current_index,
            current_epoch=self.current_epoch,
        )
        restore_optional_variable_fields(
            state,
            {
                "num_records": self.num_records,
                "num_epochs": self.num_epochs,
            },
        )
        # Mark as initialized and state restored to prevent __iter__ from resetting
        self._initialized = True
        self._is_state_restored = True
