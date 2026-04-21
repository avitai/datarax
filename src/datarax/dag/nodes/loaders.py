"""Composite loader nodes combining sourcing, shuffling, and batching."""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from typing import Any, Literal

import grain
import jax

from datarax.core.data_source import DataSourceModule
from datarax.dag.nodes.base import Node
from datarax.dag.nodes.control_flow import Sequential
from datarax.dag.nodes.data_source import BatchNode, DataSourceNode
from datarax.sources._grain_bridge import DataraxRandomAccessAdapter, records_to_batch


logger = logging.getLogger(__name__)


class DataLoaderRestoreError(RuntimeError):
    """Structured DataLoader state-restore failure."""

    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        """Initialize with failed restore targets and exceptions."""
        self.failures = failures
        details = "; ".join(f"{target}: {error}" for target, error in failures)
        super().__init__(f"Failed to restore DataLoader state: {details}")


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
        loader = DataLoader(source, batch_size=32, shuffle=True, seed=0)
        ```
    """

    _read_options: grain.ReadOptions | None
    _shard_options: grain.sharding.ShardOptions | None
    _source_transforms: tuple[Any, ...]
    _iterator: Generator[Any, None, None] | None
    _grain_loader: grain.DataLoader | None

    def __init__(
        self,
        source: DataSourceModule | DataSourceNode,
        *,
        batch_size: int,
        drop_remainder: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
        backend: Literal["auto", "grain", "datarax"] = "auto",
        worker_count: int = 0,
        worker_buffer_size: int = 1,
        read_options: grain.ReadOptions | None = None,
        shard_options: grain.sharding.ShardOptions | None = None,
        source_transforms: Sequence[grain.transforms.Transformation] = (),
        name: str | None = None,
    ) -> None:
        """Initialize data loader.

        Args:
            source: Data source module or node
            batch_size: Number of elements per batch
            drop_remainder: Whether to drop incomplete final batch
            shuffle: Whether source-level Grain sampling should shuffle
            seed: Grain sampler seed
            backend: Loader backend. ``auto`` uses Grain for random-access sources.
            worker_count: Grain worker count
            worker_buffer_size: Grain worker buffer size
            read_options: Optional Grain read options
            shard_options: Optional Grain shard options
            source_transforms: Grain transforms applied before batching
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

        if backend not in {"auto", "grain", "datarax"}:
            raise ValueError("backend must be one of 'auto', 'grain', or 'datarax'")

        source_module = source_node.source

        selected_backend = (
            "grain" if backend == "auto" and self._has_random_access_api(source_module) else backend
        )
        if selected_backend == "auto":
            selected_backend = "datarax"

        if selected_backend == "grain" and not self._has_random_access_api(source_module):
            raise ValueError("backend='grain' requires a known-size random-access source")
        if selected_backend == "datarax" and shuffle:
            raise ValueError(
                "DataLoader shuffle=True requires the Grain backend. "
                "Use ShuffleNode explicitly for batch-level DAG shuffling."
            )

        # Build the Datarax DAG backend as source -> batch.
        # Use plain list first to build pipeline, then pass to Sequential
        # Sequential's __init__ will convert to nnx.List
        nodes_list: list[Node] = [source_node]

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
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle
        self.seed = seed
        self.backend = selected_backend
        self.worker_count = worker_count
        self.worker_buffer_size = worker_buffer_size
        object.__setattr__(self, "_read_options", read_options)
        object.__setattr__(self, "_shard_options", shard_options)
        object.__setattr__(self, "_source_transforms", tuple(source_transforms))

        # Iterator state for iteration support
        # CRITICAL: Use object.__setattr__ to bypass NNX attribute tracking.
        # - nnx.Variable creates copies on access, breaking iterator state
        # - Plain assignment triggers NNX's static/data type checking
        object.__setattr__(self, "_iterator", None)
        object.__setattr__(self, "_grain_loader", None)
        self._iteration_count = 0

    @staticmethod
    def _has_random_access_api(source: DataSourceModule) -> bool:
        """Return whether a source has a concrete known-size random-access API."""
        try:
            len(source)
        except (AttributeError, NotImplementedError, TypeError):
            return False
        return type(source).__getitem__ is not DataSourceModule.__getitem__

    def __iter__(self) -> Generator[Any, None, None]:
        """Make DataLoader iterable."""
        if self.backend == "grain":
            return self._create_grain_iterator()
        return self._create_iterator()

    def _get_grain_loader(self) -> grain.DataLoader:
        """Build or return the Grain loader for the random-access backend."""
        if self._grain_loader is not None:
            return self._grain_loader

        adapter = DataraxRandomAccessAdapter(self.source.source)
        shard_options = self._shard_options or grain.sharding.NoSharding()
        sampler = grain.samplers.IndexSampler(
            num_records=len(adapter),
            shard_options=shard_options,
            shuffle=self.shuffle,
            seed=self.seed if self.shuffle else None,
            num_epochs=1,
        )
        operations = (
            *self._source_transforms,
            grain.transforms.Batch(
                batch_size=self.batch_size,
                drop_remainder=self.drop_remainder,
            ),
        )
        grain_loader = grain.DataLoader(
            data_source=adapter,
            sampler=sampler,
            operations=operations,
            worker_count=self.worker_count,
            worker_buffer_size=self.worker_buffer_size,
            shard_options=shard_options,
            read_options=self._read_options,
        )
        object.__setattr__(self, "_grain_loader", grain_loader)
        return grain_loader

    def _create_grain_iterator(self) -> Generator[Any, None, None]:
        """Create iterator using Grain's random-access DataLoader."""
        self._iteration_count = 0
        for records in self._get_grain_loader():
            self._iteration_count += 1
            yield records_to_batch(records)

    def _create_iterator(self) -> Generator[Any, None, None]:
        """Create iterator for DataLoader."""
        self._iteration_count = 0
        finite_length = self._finite_source_length()
        if finite_length is not None:
            yield from self._create_finite_datarax_iterator(finite_length)
            return

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
                        flush_fn = getattr(node, "flush", None)
                        if flush_fn is not None:
                            final_batch = flush_fn()
                            if final_batch is not None:
                                self._iteration_count += 1
                                yield final_batch
                break

    def _finite_source_length(self) -> int | None:
        """Return known source length for a single Datarax epoch when available."""
        try:
            return len(self.source.source)
        except (AttributeError, NotImplementedError, TypeError):
            return None

    def _run_tail_nodes(self, value: Any, nodes: Sequence[Node]) -> Any:
        """Run non-source DAG nodes for finite-source iteration."""
        result = value
        for node in nodes:
            if result is None:
                return None
            result = node(result)
        return result

    def _create_finite_datarax_iterator(self, source_length: int) -> Generator[Any, None, None]:
        """Iterate a known-size source once without allowing DataSourceNode to restart it."""
        source_iter = iter(self.source.source)
        tail_nodes = list(self.nodes)[1:]

        for _ in range(source_length):
            try:
                element = next(source_iter)
            except StopIteration:
                break
            batch = self._run_tail_nodes(element, tail_nodes)
            if batch is not None:
                self._iteration_count += 1
                yield batch

        for node_index, node in enumerate(tail_nodes):
            flush_fn = getattr(node, "flush", None)
            if flush_fn is None:
                continue
            while True:
                flushed = flush_fn()
                if flushed is None:
                    break
                batch = self._run_tail_nodes(flushed, tail_nodes[node_index + 1 :])
                if batch is not None:
                    self._iteration_count += 1
                    yield batch

    def get_state(self) -> dict[str, Any]:
        """Get DataLoader state for checkpointing."""
        # Get state from all child nodes
        state = {"iteration_count": self._iteration_count, "nodes": []}

        # Get state from each node in the pipeline
        for i, node in enumerate(self.nodes):
            get_state_fn = getattr(node, "get_state", None)
            if get_state_fn is not None:
                node_state = get_state_fn()
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
        failures: list[tuple[str, Exception]] = []
        if "nodes" in state:
            node_states = state["nodes"]
            if len(node_states) != len(self.nodes):
                failures.append(
                    (
                        "nodes",
                        ValueError(
                            f"Expected {len(self.nodes)} node states, got {len(node_states)}"
                        ),
                    )
                )
            else:
                for i, (node, node_state) in enumerate(zip(self.nodes, node_states)):
                    set_state_fn = getattr(node, "set_state", None)
                    if set_state_fn is not None and isinstance(node_state, dict):
                        try:
                            set_state_fn(node_state)
                        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                            failures.append((getattr(node, "name", f"node_{i}"), error))

        if failures:
            raise DataLoaderRestoreError(failures)

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
        shuffle_info = ", shuffle=True" if self.shuffle else ""
        return f"DataLoader(batch_size={self.batch_size}, backend={self.backend}{shuffle_info})"


def dataloader(
    source: DataSourceModule | DataSourceNode,
    batch_size: int,
    drop_remainder: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    backend: Literal["auto", "grain", "datarax"] = "auto",
) -> DataLoader:
    """Create a data loader with source, shuffling, and batching.

    Args:
        source: Data source module or node
        batch_size: Number of elements per batch
        drop_remainder: Whether to drop incomplete final batch
        shuffle: Whether source-level Grain sampling should shuffle
        seed: Grain sampler seed
        backend: Loader backend.

    Returns:
        DataLoader node
    """
    return DataLoader(
        source=source,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        shuffle=shuffle,
        seed=seed,
        backend=backend,
    )
