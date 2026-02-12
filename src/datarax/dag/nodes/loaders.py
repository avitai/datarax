"""Composite loader nodes combining sourcing, shuffling, and batching."""

from __future__ import annotations
import jax
from typing import Any

from datarax.dag.nodes.base import Node
from datarax.dag.nodes.control_flow import Sequential
from datarax.dag.nodes.data_source import DataSourceNode, BatchNode, ShuffleNode
from datarax.core.data_source import DataSourceModule


class DataLoader(Sequential):
    """Data loader node that combines data source, shuffling, and batching.

    This node creates a complete data loading pipeline by combining:
    1. DataSourceNode - provides the raw data
    2. ShuffleNode - shuffles the data for better training
    3. BatchNode - creates batches from individual elements

    The resulting pipeline applies shuffling first, then batching to the
    output of the data source, following Datarax's batch-first principle.

    Examples:
        Basic data loader:

        ```python
        from datarax.dag.nodes import DataLoader
        from datarax.sources import MemorySource, MemorySourceConfig
        source = MemorySource(MemorySourceConfig(), [{"x": 1}])
        loader = DataLoader(source, batch_size=32, shuffle_buffer_size=1000)
        ```
    """

    def __init__(
        self,
        source: DataSourceModule | DataSourceNode,
        batch_size: int,
        shuffle_buffer_size: int | None = None,
        drop_remainder: bool = False,
        shuffle_seed: int | None = None,
        name: str | None = None,
    ):
        """Initialize data loader.

        Args:
            source: Data source module or node
            batch_size: Number of elements per batch
            shuffle_buffer_size: Size of shuffle buffer (None to disable shuffling)
            drop_remainder: Whether to drop incomplete final batch
            shuffle_seed: Random seed for shuffling
            name: Optional name for the loader
        """
        # Convert source to DataSourceNode if needed
        if isinstance(source, DataSourceModule):
            source_node = DataSourceNode(source)
        elif isinstance(source, DataSourceNode):
            source_node = source
        else:
            raise ValueError(
                f"Source must be DataSourceModule or DataSourceNode, got {type(source)}"
            )

        # Build the pipeline: source -> [shuffle] -> batch
        # Use plain list first to build pipeline, then pass to Sequential
        # Sequential's __init__ will convert to nnx.List
        nodes_list: list[Node] = [source_node]

        # Add shuffle node if buffer size is specified
        if shuffle_buffer_size is not None and shuffle_buffer_size > 0:
            shuffle_node: Node = ShuffleNode(
                buffer_size=shuffle_buffer_size,
                seed=shuffle_seed,
                name=f"{name or 'DataLoader'}_Shuffle",
            )
            nodes_list.append(shuffle_node)

        # Add batch node
        batch_node: Node = BatchNode(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            name=f"{name or 'DataLoader'}_Batch",
        )
        nodes_list.append(batch_node)

        # Initialize as Sequential (which will convert nodes_list to nnx.List)
        super().__init__(nodes_list)
        self.name = name or "DataLoader"

        # Store configuration for inspection
        self.source = source_node
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_remainder = drop_remainder
        self.shuffle_seed = shuffle_seed

        # Iterator state for iteration support
        # CRITICAL: Use object.__setattr__ to bypass NNX attribute tracking.
        # - nnx.Variable creates copies on access, breaking iterator state
        # - Plain assignment triggers NNX's static/data type checking
        object.__setattr__(self, "_iterator", None)
        self._iteration_count = 0

    def __iter__(self):
        """Make DataLoader iterable."""
        return self._create_iterator()

    def _create_iterator(self):
        """Create iterator for DataLoader."""
        self._iteration_count = 0
        while True:
            # Keep calling the pipeline until we get a batch or run out of data
            batch = None
            attempts = 0
            max_attempts = self.batch_size * 2  # Reasonable limit to avoid infinite loops

            while batch is None and attempts < max_attempts:
                try:
                    batch = self(None)  # Call the Sequential pipeline
                    attempts += 1
                except StopIteration:
                    # Data source is exhausted
                    break

            if batch is not None:
                self._iteration_count += 1
                yield batch
            else:
                # No more data or couldn't form a batch
                # Try to flush any remaining data from BatchNode
                if hasattr(self, "nodes") and len(self.nodes) > 0:
                    # Find the BatchNode and flush it
                    for node in self.nodes:
                        if hasattr(node, "flush"):
                            final_batch = node.flush()
                            if final_batch is not None:
                                self._iteration_count += 1
                                yield final_batch
                break

    def get_state(self) -> dict[str, Any]:
        """Get DataLoader state for checkpointing."""
        # Get state from all child nodes
        state = {"iteration_count": self._iteration_count, "nodes": []}

        # Get state from each node in the pipeline
        for i, node in enumerate(self.nodes):
            if hasattr(node, "get_state"):
                node_state = node.get_state()
            else:
                # For nodes without get_state, try to get basic state info
                node_state = {
                    "type": type(node).__name__,
                    "name": getattr(node, "name", f"node_{i}"),
                }
            state["nodes"].append(node_state)

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Set DataLoader state from checkpoint."""
        if "iteration_count" in state:
            self._iteration_count = state["iteration_count"]

        # Set state for each node if available
        if "nodes" in state and len(state["nodes"]) == len(self.nodes):
            for i, (node, node_state) in enumerate(zip(self.nodes, state["nodes"])):
                if hasattr(node, "set_state") and isinstance(node_state, dict):
                    try:
                        node.set_state(node_state)
                    except Exception:
                        # If setting state fails, continue with other nodes
                        pass

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Execute DataLoader pipeline.

        DataLoader overrides Sequential's __call__ to handle the special case
        where DataSourceNode needs to generate data from None input.
        """
        result = data
        for i, node in enumerate(self.nodes):
            # For the first node (DataSourceNode), allow None input
            if i == 0 or result is not None:
                result = node(result, key=key)
            else:
                # If we get None from a middle node (e.g., BatchNode buffering), return None
                return None
        return result

    def __repr__(self) -> str:
        """String representation."""
        shuffle_info = f", shuffle={self.shuffle_buffer_size}" if self.shuffle_buffer_size else ""
        return f"DataLoader(batch_size={self.batch_size}{shuffle_info})"


def dataloader(
    source: DataSourceModule | DataSourceNode,
    batch_size: int,
    shuffle_buffer_size: int | None = None,
    drop_remainder: bool = False,
    shuffle_seed: int | None = None,
) -> DataLoader:
    """Create a data loader with source, shuffling, and batching.

    Args:
        source: Data source module or node
        batch_size: Number of elements per batch
        shuffle_buffer_size: Size of shuffle buffer (None to disable shuffling)
        drop_remainder: Whether to drop incomplete final batch
        shuffle_seed: Random seed for shuffling

    Returns:
        DataLoader node
    """
    return DataLoader(
        source=source,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        drop_remainder=drop_remainder,
        shuffle_seed=shuffle_seed,
    )
