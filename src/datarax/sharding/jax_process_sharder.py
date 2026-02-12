"""JAX process sharder for distributing data across multiple processes."""

from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

from datarax.core.sharder import SharderModule, SharderModuleConfig


@dataclass
class JaxProcessSharderConfig(SharderModuleConfig):
    """Configuration for JaxProcessSharderModule.

    Attributes:
        drop_remainder: Whether to drop remainder samples to ensure equal shards.
    """

    drop_remainder: bool = True


class JaxProcessSharderModule(SharderModule):
    """Shard data across JAX processes.

    Implements Grain's ShardByJaxProcess pattern with state tracking.
    """

    def __init__(
        self,
        config: JaxProcessSharderConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize JaxProcessSharderModule.

        Args:
            config: Sharder configuration with drop_remainder setting.
            rngs: Optional Flax NNX random number generators.
            name: Optional module name for identification.
        """
        if config is None:
            config = JaxProcessSharderConfig()
        super().__init__(config, rngs=rngs, name=name)
        self.process_index = nnx.Variable(jax.process_index())
        self.process_count = nnx.Variable(jax.process_count())
        self.local_device_count = nnx.Variable(jax.local_device_count())

    def shard_data(self, data: Any) -> Any:
        """Shard data for current process."""
        process_count = self.process_count.get_value()
        process_index = self.process_index.get_value()

        if isinstance(data, list | tuple):
            total_size = len(data)
            shard_size = total_size // process_count

            if self.config.drop_remainder:
                # Ensure equal shards
                shard_size = total_size // process_count
                start_idx = process_index * shard_size
                end_idx = start_idx + shard_size
            else:
                # Allow unequal shards
                start_idx = process_index * shard_size
                if process_index == process_count - 1:
                    end_idx = total_size
                else:
                    end_idx = start_idx + shard_size

            return data[start_idx:end_idx]

        elif isinstance(data, jax.Array | np.ndarray):
            return self._shard_array(data)

        else:
            raise ValueError(f"Cannot shard data of type {type(data)}")

    def _shard_array(self, array: jax.Array | np.ndarray) -> jax.Array | np.ndarray:
        """Shard array across first dimension."""
        process_count = self.process_count.get_value()
        process_index = self.process_index.get_value()

        total_size = array.shape[0]
        shard_size = total_size // process_count

        if self.config.drop_remainder:
            shard_size = total_size // process_count
            start_idx = process_index * shard_size
            end_idx = start_idx + shard_size
        else:
            start_idx = process_index * shard_size
            if process_index == process_count - 1:
                end_idx = total_size
            else:
                end_idx = start_idx + shard_size

        return array[start_idx:end_idx]
