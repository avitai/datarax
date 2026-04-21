"""JAX process sharder for distributing data across multiple processes."""

import logging
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

from datarax.core.sharder import SharderModule, SharderModuleConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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
    ) -> None:
        """Initialize JaxProcessSharderModule.

        Args:
            config: Sharder configuration with drop_remainder setting.
            rngs: Optional Flax NNX random number generators.
            name: Optional module name for identification.
        """
        if config is None:
            config = JaxProcessSharderConfig()
        super().__init__(config, rngs=rngs, name=name)
        self.config: JaxProcessSharderConfig = config
        self.process_index = nnx.Variable(jax.process_index())
        self.process_count = nnx.Variable(jax.process_count())
        self.local_device_count = nnx.Variable(jax.local_device_count())

    def shard_data(self, data: Any) -> Any:
        """Shard data for current process."""
        if isinstance(data, list | tuple):
            start_idx, end_idx = self._shard_bounds(len(data))
            return data[start_idx:end_idx]

        elif isinstance(data, jax.Array | np.ndarray):
            return self._shard_array(data)

        else:
            raise ValueError(f"Cannot shard data of type {type(data)}")

    def _shard_array(self, array: jax.Array | np.ndarray) -> jax.Array | np.ndarray:
        """Shard array across first dimension."""
        start_idx, end_idx = self._shard_bounds(array.shape[0])
        return array[start_idx:end_idx]

    def _shard_bounds(self, total_size: int) -> tuple[int, int]:
        """Return Grain-style process shard bounds."""
        process_count = self.process_count.get_value()
        process_index = self.process_index.get_value()

        if self.config.drop_remainder:
            shard_size = total_size // process_count
            start_idx = process_index * shard_size
            return start_idx, start_idx + shard_size

        base_size = total_size // process_count
        remainder = total_size % process_count
        start_idx = process_index * base_size + min(process_index, remainder)
        shard_size = base_size + int(process_index < remainder)
        return start_idx, start_idx + shard_size
