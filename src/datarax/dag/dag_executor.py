"""
Complete DAG pipeline executor for Datarax.

This module provides the main DAGExecutor class for executing complex
data processing workflows as directed acyclic graphs (DAGs).

COMPLETE IMPLEMENTATION following TDD principles.
"""

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast, Literal

import flax.nnx as nnx
import jax
from flax.nnx.transforms.compilation import JitFn, JitWrapped


_logger = logging.getLogger(__name__)
_IS_XLA_INITIALIZED = False


def _ensure_xla_optimized() -> None:
    """Apply hardware-specific XLA flags and enable compilation cache.

    Called once per process on first DAGExecutor creation. Idempotent — repeated
    calls are no-ops. This replaces the need for manual XLAOptimizer() instantiation.
    """
    global _IS_XLA_INITIALIZED  # noqa: PLW0603
    if _IS_XLA_INITIALIZED:
        return
    _IS_XLA_INITIALIZED = True

    from datarax.performance.xla_optimization import apply_xla_flags

    backend = jax.default_backend()

    # --- Hardware-specific XLA flags (delegated to canonical source) ---
    apply_xla_flags(backend)

    # --- Persistent compilation cache ---
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        cache_path = Path(tempfile.gettempdir()) / "datarax_jax_cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_path)

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    _logger.debug("Compilation cache enabled at: %s", cache_dir)


from datarax.core.batcher import BatcherModule
from datarax.core.data_source import DataSourceModule
from datarax.core.element_batch import BatchView
from datarax.core.module import DataraxModule
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule
from datarax.dag.nodes import (
    BatchNode,
    Branch,
    CacheNode,
    DataLoader,
    DataSourceNode,
    FusedOperatorNode,
    Identity,
    Merge,
    Node,
    OperatorNode,
    Parallel,
    SamplerNode,
    Sequential,
    SharderNode,
    ShuffleNode,
)
from datarax.typing import Batch


@dataclass
class _ExecutorConfigState:
    """Configuration state for DAGExecutor."""

    enable_caching: bool
    jit_compile: bool
    enforce_batch: bool
    prefetch_size: int
    device_prefetch: bool
    name: str


@dataclass
class _ExecutorRuntimeState:
    """Mutable runtime state for DAGExecutor."""

    rngs_seed: nnx.Rngs | None
    rngs: nnx.Rngs | None = None
    needs_rng: bool | None = None
    has_stateful_ops: bool | None = None
    iteration_count: int = 0
    epoch_count: int = 0
    source_node: DataSourceNode | None = None
    batch_node: BatchNode | None = None
    iterator: Iterator[Any] | None = None
    cache: dict[int, Any] | None = None
    fused_step_cache: dict[tuple[Any, ...], Callable[[dict, dict], tuple[dict, dict]]] = field(
        default_factory=dict
    )
    jit_execute: JitFn | JitWrapped | partial | None = None


class DAGExecutor(nnx.Module):
    """Complete DAG executor for complex data processing workflows.

    This class provides a DAG-based execution model for complex
    data processing workflows with automatic optimization.

    Key Features:

        - Full DAG execution with topological sorting
        - Automatic caching and optimization
        - JIT compilation support
        - Complete checkpointing for pipeline state
        - Operator-based composition
        - ENFORCES BATCH-FIRST PROCESSING

    Examples:
        Linear pipeline:

        ```python
        executor = DAGExecutor()
        executor.add(source).add(batch(32)).add(transform())
        ```

        Using operators:

        ```python
        executor = source >> batch(32) >> transform()
        ```

        Complex DAG:

        ```python
        executor = DAGExecutor()
        executor.add(source)
        executor.add(batch(32))
        executor.parallel([transform_a(), transform_b()])
        executor.merge('concat')
        executor.add(shuffle(1000))
        ```

        Iterate over batches:

        ```python
        for batch in executor:
            process(batch)
        ```
    """

    def __init__(
        self,
        graph: Node | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        enable_caching: bool = True,
        jit_compile: bool = False,
        enforce_batch: bool = True,
        prefetch_size: int = 2,
        device_prefetch: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize DAG executor.

        Args:
            graph: Initial graph node or None for empty
            rngs: RNG state for the pipeline (lazy - only used if stochastic ops exist)
            enable_caching: Whether to cache intermediate results
            jit_compile: Whether to JIT compile the pipeline
            enforce_batch: Whether to enforce batch-first processing
            prefetch_size: Number of batches to prefetch ahead (0 = disabled)
            device_prefetch: Enable two-stage prefetch with async device transfer.
                When True, adds a DevicePrefetcher stage after CPU prefetch that
                calls jax.device_put in a background thread. Requires prefetch_size > 0.
            name: Optional name for the executor
        """
        super().__init__()
        _ensure_xla_optimized()

        # Initialize graph
        if graph is None:
            self.graph: Node = Identity()
        elif isinstance(graph, DataraxModule):
            # Wrap Datarax module in appropriate node
            if isinstance(graph, DataSourceModule):
                self.graph = DataSourceNode(graph)
            elif isinstance(graph, OperatorModule):
                self.graph = OperatorNode(graph)
            else:
                raise ValueError(f"Unsupported module type: {type(graph)}")
        else:
            self.graph = graph

        self._config_state = _ExecutorConfigState(
            enable_caching=enable_caching,
            jit_compile=jit_compile,
            enforce_batch=enforce_batch,
            prefetch_size=prefetch_size,
            device_prefetch=device_prefetch,
            name=name or "DAGExecutor",
        )
        self._runtime_state = _ExecutorRuntimeState(
            rngs_seed=rngs,
            cache={} if enable_caching else None,
        )

        if jit_compile:
            self._compile()

        # Validate graph
        self._validate_graph()

    @property
    def enable_caching(self) -> bool:
        """Whether executor caching is enabled."""
        return self._config_state.enable_caching

    @enable_caching.setter
    def enable_caching(self, value: bool) -> None:
        self._config_state.enable_caching = value

    @property
    def jit_compile(self) -> bool:
        """Whether executor uses JIT compilation."""
        return self._config_state.jit_compile

    @jit_compile.setter
    def jit_compile(self, value: bool) -> None:
        self._config_state.jit_compile = value

    @property
    def enforce_batch(self) -> bool:
        """Whether batch-first ordering is enforced."""
        return self._config_state.enforce_batch

    @enforce_batch.setter
    def enforce_batch(self, value: bool) -> None:
        self._config_state.enforce_batch = value

    @property
    def prefetch_size(self) -> int:
        """Configured prefetch size."""
        return self._config_state.prefetch_size

    @prefetch_size.setter
    def prefetch_size(self, value: int) -> None:
        self._config_state.prefetch_size = value

    @property
    def device_prefetch(self) -> bool:
        """Whether two-stage device prefetch is enabled."""
        return self._config_state.device_prefetch

    @device_prefetch.setter
    def device_prefetch(self, value: bool) -> None:
        self._config_state.device_prefetch = value

    @property
    def name(self) -> str:
        """Executor display name."""
        return self._config_state.name

    @name.setter
    def name(self, value: str) -> None:
        self._config_state.name = value

    @property
    def _rngs_seed(self) -> nnx.Rngs | None:
        return self._runtime_state.rngs_seed

    @_rngs_seed.setter
    def _rngs_seed(self, value: nnx.Rngs | None) -> None:
        self._runtime_state.rngs_seed = value

    @property
    def _rngs(self) -> nnx.Rngs | None:
        return self._runtime_state.rngs

    @_rngs.setter
    def _rngs(self, value: nnx.Rngs | None) -> None:
        self._runtime_state.rngs = value

    @property
    def _needs_rng(self) -> bool | None:
        return self._runtime_state.needs_rng

    @_needs_rng.setter
    def _needs_rng(self, value: bool | None) -> None:
        self._runtime_state.needs_rng = value

    @property
    def _has_stateful_ops(self) -> bool | None:
        return self._runtime_state.has_stateful_ops

    @_has_stateful_ops.setter
    def _has_stateful_ops(self, value: bool | None) -> None:
        self._runtime_state.has_stateful_ops = value

    @property
    def _iteration_count(self) -> int:
        return self._runtime_state.iteration_count

    @_iteration_count.setter
    def _iteration_count(self, value: int) -> None:
        self._runtime_state.iteration_count = value

    @property
    def _epoch_count(self) -> int:
        return self._runtime_state.epoch_count

    @_epoch_count.setter
    def _epoch_count(self, value: int) -> None:
        self._runtime_state.epoch_count = value

    @property
    def _source_node(self) -> DataSourceNode | None:
        return self._runtime_state.source_node

    @_source_node.setter
    def _source_node(self, value: DataSourceNode | None) -> None:
        self._runtime_state.source_node = value

    @property
    def _batch_node(self) -> BatchNode | None:
        return self._runtime_state.batch_node

    @_batch_node.setter
    def _batch_node(self, value: BatchNode | None) -> None:
        self._runtime_state.batch_node = value

    @property
    def _iterator(self) -> Iterator[Any] | None:
        return self._runtime_state.iterator

    @_iterator.setter
    def _iterator(self, value: Iterator[Any] | None) -> None:
        self._runtime_state.iterator = value

    @property
    def _cache(self) -> dict[int, Any] | None:
        return self._runtime_state.cache

    @_cache.setter
    def _cache(self, value: dict[int, Any] | None) -> None:
        self._runtime_state.cache = value

    @property
    def _jit_execute(self) -> JitFn | JitWrapped | partial | None:
        """JIT-compiled execute function cached in runtime state."""
        return self._runtime_state.jit_execute

    @property
    def rngs(self) -> nnx.Rngs | None:
        """Lazy RNG property - only creates RNG if pipeline needs randomness.

        This optimization follows Grain's pattern where RNG is only generated
        for stochastic operations, not for deterministic pipelines.

        If user explicitly provides rngs at construction, those are always returned
        (user knows their use case). Otherwise, rngs is created lazily only when
        stochastic operations are detected.
        """
        # Return cached RNG if available
        if self._rngs is not None:
            return self._rngs

        # If user provided explicit rngs, always use them
        if self._rngs_seed is not None:
            self._rngs = self._rngs_seed
            return self._rngs

        # Check if pipeline needs RNG (cache result)
        if self._needs_rng is None:
            self._needs_rng = self._detect_stochastic_ops()

        # Only create default RNG if pipeline has stochastic ops
        if self._needs_rng:
            new_rngs = nnx.Rngs(0)
            self._rngs = new_rngs
            return self._rngs

        # Pipeline is deterministic and user didn't provide rngs - return None
        return None

    def _any_node_matches(self, predicate: Callable[[Node], bool]) -> bool:
        """Walk the DAG and return True if any leaf node satisfies predicate.

        Handles recursion into composite nodes (Sequential, Parallel, Branch)
        so callers only need to define the leaf-node check.

        Args:
            predicate: Function that checks a single non-composite node.

        Returns:
            True if any node in the graph satisfies the predicate.
        """

        def walk(node: Node) -> bool:
            if isinstance(node, Sequential):
                return any(walk(n) for n in node.nodes)
            if isinstance(node, Parallel):
                return any(walk(n) for n in node.nodes)
            if isinstance(node, Branch):
                return walk(node.true_path) or walk(node.false_path)
            return predicate(node)

        return walk(self.graph)

    def _detect_stochastic_ops(self) -> bool:
        """Detect if pipeline contains any stochastic operations.

        Traverses the DAG to check for stochastic nodes/operators.
        This follows Grain's pattern of separating deterministic (map)
        from stochastic (random_map) operations.

        Returns:
            True if any stochastic operations exist, False otherwise.
        """

        def is_stochastic(node: Node) -> bool:
            if isinstance(node, ShuffleNode):
                return True
            for attr in ("operator", "sampler", "source"):
                inner = getattr(node, attr, None)
                if inner is not None and getattr(
                    getattr(inner, "config", None), "stochastic", False
                ):
                    return True
            return False

        return self._any_node_matches(is_stochastic)

    def _detect_stateful_ops(self) -> bool:
        """Detect if pipeline contains stateful operators that affect output.

        Checks for operators with config.stateful=True. This is an explicit
        marker for operators whose output depends on internal mutable state
        (not just diagnostic counters like _iteration_count).

        Note: This check is deliberately conservative. Most operators with
        _iteration_count use it for diagnostics only, not to affect output.
        Only operators explicitly marked as stateful=True will disable caching.

        Returns:
            True if any stateful operations exist, False otherwise.
        """

        def is_stateful(node: Node) -> bool:
            if isinstance(node, OperatorNode):
                operator = node.operator
                if hasattr(operator, "config"):
                    return getattr(operator.config, "stateful", False)
            return False

        return self._any_node_matches(is_stateful)

    def _topological_sort(self) -> list[Node]:
        """Flatten the graph tree into a topological execution order.

        Walks the tree structure (Sequential, Parallel, Branch) and returns
        a flat list of leaf nodes in execution order.  Sequential children
        are ordered left-to-right; Parallel children are all included
        (independent of each other).  Composite nodes (Sequential, Parallel)
        are traversed but NOT included in the output — only leaf nodes appear.

        Returns:
            Flat list of leaf nodes in valid topological order.
        """
        order: list[Node] = []
        seen: set[int] = set()

        def walk(node: Node) -> None:
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)

            if isinstance(node, Sequential):
                for child in node.nodes:
                    walk(child)
            elif isinstance(node, Parallel):
                for child in node.nodes:
                    walk(child)
            elif isinstance(node, Branch):
                walk(node.true_path)
                walk(node.false_path)
            elif isinstance(node, FusedOperatorNode):
                # Treat fused group as a single opaque node
                order.append(node)
            else:
                # Leaf node (OperatorNode, BatchNode, ShuffleNode, Merge, etc.)
                order.append(node)

        walk(self.graph)
        return order

    def _compute_fusion_groups(self) -> list[list[Node]]:
        """Identify groups of consecutive fusible nodes in the graph.

        Walks the graph's nodes (if Sequential) or treats the root as a
        single-node sequence. Accumulates consecutive is_jit_fusible nodes
        into fusion groups. Groups of size >= 2 are returned.

        Returns:
            List of fusible groups, each a list of consecutive fusible nodes.
        """
        # Get flat node list from the graph
        if isinstance(self.graph, Sequential):
            nodes = list(self.graph.nodes)
        else:
            nodes = [self.graph]

        groups: list[list[Node]] = []
        current_group: list[Node] = []

        for node in nodes:
            if node.is_jit_fusible:
                current_group.append(node)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = []

        # Don't forget the last group
        if len(current_group) >= 2:
            groups.append(current_group)

        return groups

    def _apply_fusion(self) -> None:
        """Replace fusible groups in the graph with FusedOperatorNode instances.

        Uses _compute_fusion_groups() as the single source of truth for
        group detection, then rebuilds the Sequential with fused wrappers.
        """
        if not isinstance(self.graph, Sequential):
            return

        groups = self._compute_fusion_groups()
        if not groups:
            return

        # Map first-node-id → group for O(1) lookup during the walk
        group_map: dict[int, list[Node]] = {id(g[0]): g for g in groups}
        members: set[int] = {id(n) for g in groups for n in g}

        new_nodes: list[Node] = []
        for node in self.graph.nodes:
            if id(node) in group_map:
                new_nodes.append(FusedOperatorNode(group_map[id(node)]))
            elif id(node) not in members:
                # Node is not part of any fusion group — keep as-is
                new_nodes.append(node)
            # else: node is a non-first member of a group — skip (already fused)

        self.graph = Sequential(new_nodes)

    def add(
        self,
        node: (
            Node | DataSourceModule | OperatorModule | BatcherModule | SamplerModule | SharderModule
        ),
    ) -> "DAGExecutor":
        """Add a node to the pipeline.

        Extends the current graph sequentially. Accepts Node instances
        or module types (DataSourceModule, OperatorModule, etc.) which
        are auto-wrapped into the appropriate Node.

        Args:
            node: Node or module to add

        Returns:
            Self for chaining
        """
        normalized_node = self._normalize_added_node(node)
        if normalized_node is None:
            return self
        self._enforce_batch_order(normalized_node)
        self._append_node_to_graph(normalized_node)
        self._invalidate_execution_caches()
        return self

    def _normalize_added_node(self, node: Any) -> Node | None:
        """Normalize supported module/container types to a DAG node."""
        if isinstance(node, DataSourceModule):
            self._set_source_node(DataSourceNode(node))
            return None
        if isinstance(node, DataSourceNode):
            self._set_source_node(node)
            return None
        if isinstance(node, OperatorModule):
            return OperatorNode(node)
        if isinstance(node, BatcherModule):
            return OperatorNode(node)
        if isinstance(node, SamplerModule):
            return SamplerNode(node)
        if isinstance(node, SharderModule):
            return SharderNode(node)
        if isinstance(node, Node):
            return node
        raise ValueError(f"Unsupported type: {type(node)}")

    def _set_source_node(self, node: DataSourceNode) -> None:
        """Set source node outside the graph execution chain."""
        self._source_node = nnx.data(node)

    def _enforce_batch_order(self, node: Node) -> None:
        """Enforce batch-first operator ordering when configured."""
        if not self.enforce_batch:
            return
        if isinstance(node, BatchNode):
            self._batch_node = nnx.data(node)
            return
        if isinstance(node, OperatorNode) and self._batch_node is None:
            raise ValueError(
                "Batch-first enforcement: Add BatchNode before transforms. "
                "Use executor.add(BatchNode(batch_size)) first."
            )

    def _append_node_to_graph(self, node: Node) -> None:
        """Append a node sequentially to the current graph."""
        if isinstance(self.graph, Identity):
            self.graph = node
            return
        self.graph = self.graph >> node

    def _invalidate_execution_caches(self) -> None:
        """Invalidate derived execution caches after graph mutation."""
        self._runtime_state.jit_execute = None
        self._runtime_state.fused_step_cache.clear()
        self._needs_rng = None
        self._has_stateful_ops = None

    def parallel(self, nodes: list[Node]) -> "DAGExecutor":
        """Add parallel branches to the pipeline.

        Args:
            nodes: List of nodes to execute in parallel (can be Node or OperatorModule)

        Returns:
            Self for chaining
        """
        # Auto-wrap OperatorModule instances in OperatorNode
        wrapped_nodes = []
        for node in nodes:
            if isinstance(node, OperatorModule) and not isinstance(node, Node):
                wrapped_nodes.append(OperatorNode(node))
            else:
                wrapped_nodes.append(node)

        parallel_node: Parallel = Parallel(wrapped_nodes)
        return self.add(parallel_node)

    def branch(
        self,
        condition: Callable[[Any], bool],
        true_path: Node,
        false_path: Node,
    ) -> "DAGExecutor":
        """Add conditional branching.

        Args:
            condition: Function that returns True/False
            true_path: Path if condition is True
            false_path: Path if condition is False

        Returns:
            Self for chaining
        """
        # Convert modules to nodes
        if isinstance(true_path, DataSourceModule):
            true_path = DataSourceNode(true_path)
        elif isinstance(true_path, OperatorModule):
            true_path = OperatorNode(true_path)

        if isinstance(false_path, DataSourceModule):
            false_path = DataSourceNode(false_path)
        elif isinstance(false_path, OperatorModule):
            false_path = OperatorNode(false_path)

        branch_node: Branch = Branch(condition, true_path, false_path)
        return self.add(branch_node)

    def merge(
        self, strategy: Literal["concat", "sum", "mean", "stack"] = "concat", axis: int = -1
    ) -> "DAGExecutor":
        """Merge parallel branches.

        Args:
            strategy: How to merge parallel results
            axis: Axis for concatenation/stacking

        Returns:
            Self for chaining
        """
        merge_node: Merge = Merge(strategy=strategy, axis=axis)
        return self.add(merge_node)

    def cache(self, cache_size: int = 100) -> "DAGExecutor":
        """Add caching to the current graph.

        Args:
            cache_size: Maximum cache entries

        Returns:
            Self for chaining
        """
        self.graph: Node = CacheNode(self.graph, cache_size=cache_size)
        return self

    def shuffle(self, buffer_size: int) -> "DAGExecutor":
        """Add shuffling to the pipeline.

        Args:
            buffer_size: Size of shuffle buffer

        Returns:
            Self for chaining
        """
        shuffle_node: ShuffleNode = ShuffleNode(buffer_size=buffer_size)
        return self.add(shuffle_node)

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "DAGExecutor":
        """Add batching to the pipeline.

        Args:
            batch_size: Number of elements per batch
            drop_remainder: Whether to drop incomplete final batch

        Returns:
            Self for chaining
        """
        batch_node: BatchNode = BatchNode(batch_size=batch_size, drop_remainder=drop_remainder)
        return self.add(batch_node)

    def operate(self, operator: OperatorModule) -> "DAGExecutor":
        """Add an operator to the pipeline.

        Args:
            operator: OperatorModule to add

        Returns:
            Self for chaining
        """
        return self.add(OperatorNode(operator))

    def __rshift__(self, other: Node | OperatorModule) -> "DAGExecutor":
        """Alias for add() to support >> operator.

        Examples:
            Basic piping:

            ```python
            pipeline = from_source(source) >> normalize >> augment
            ```
        """
        # _normalize_added_node handles OperatorModule → OperatorNode conversion
        return self.add(other)

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Execute the DAG on a single data element.

        Args:
            data: Input data to process
            key: Optional RNG key

        Returns:
            Processed data
        """
        # Increment iteration count for each pipeline execution
        self._iteration_count += 1

        # If no RNG provided, create one if we have rngs
        if key is None and hasattr(self, "rngs") and self.rngs is not None:
            key = self.rngs.default()

        return self._execute(self.graph, data, key)

    def __iter__(self) -> Iterator[Any]:
        """Create iterator for the pipeline.

        Returns:
            Iterator that yields batches
        """
        if self._source_node is None:
            raise ValueError("No data source in pipeline. Add a DataSourceNode first.")

        if self.enforce_batch and self._batch_node is None:
            raise ValueError(
                "Batch-first enforcement: No BatchNode in pipeline. "
                "Add batch(batch_size) to the pipeline."
            )

        self._iterator = nnx.data(self._create_iterator())
        self._epoch_count += 1
        return self

    def __next__(self) -> Any:
        """Get next batch from the pipeline.

        Returns:
            Next processed batch

        Raises:
            StopIteration: When pipeline is exhausted
        """
        if self._iterator is None:
            raise ValueError("Call iter() first or use in for loop")

        return next(self._iterator)

    def close(self) -> None:
        """Close the active iterator and release streaming resources."""
        if self._iterator is None:
            return
        close_fn = getattr(self._iterator, "close", None)
        if callable(close_fn):
            close_fn()
        self._iterator = None

    def _can_use_batch_first(self) -> bool:
        """Check if batch-first execution path can be used.

        Batch-first execution retrieves batches directly from the source
        instead of iterating element-by-element. This is significantly
        faster (10-100x) for sources that support direct batch retrieval.

        Returns:
            True if batch-first path is available.
        """
        if self._source_node is None or self._batch_node is None:
            return False

        # Check if source has get_batch method
        source = self._source_node.source
        if not hasattr(source, "get_batch"):
            return False

        # Check if source has a known length (required for batch iteration)
        # Note: Some sources (e.g., HFStreamingSource) have __len__
        # but return None, so we must verify it returns a valid integer
        if not hasattr(source, "__len__"):
            return False

        try:
            length = len(source)
            if length is None or not isinstance(length, int):
                return False
        except (TypeError, NotImplementedError):
            return False

        return True

    def _get_post_batch_operators(self) -> list[OperatorNode]:
        """Get list of operators that come after BatchNode in the pipeline.

        Returns:
            List of OperatorNode instances to apply after batching.
        """
        operators = []

        def collect_operators(node: Node) -> None:
            """Recursively collect operators from node graph."""
            if isinstance(node, OperatorNode):
                operators.append(node)
            elif isinstance(node, Sequential):
                for child in node.nodes:
                    collect_operators(child)

        # Get the graph portion after BatchNode
        post_batch = self._get_post_batch_graph()
        if post_batch is not None:
            collect_operators(post_batch)

        return operators

    def _convert_to_jax_arrays(self, batch_data: Any) -> dict:
        """Convert batch data to JAX arrays.

        Uses jax.device_put for efficient async host-to-device transfer.
        This handles dicts natively (pytree support), batches the transfers,
        and can overlap with Python execution.

        Handles multiple formats from source.get_batch():
        1. Dict with array values: {"image": array, "label": array}
        2. List of element dicts: [{"image": arr1, "label": 0}, ...]
        3. List of raw values: [1, 2, 3, ...] -> {"data": array}
        4. Raw array: array -> {"data": array}

        Args:
            batch_data: Data from source's get_batch() (dict, list, or array).

        Returns:
            Dictionary with batched JAX arrays.
        """
        import numpy as np

        # Handle list input — stack into arrays first
        if isinstance(batch_data, list):
            if not batch_data:
                return {}
            if isinstance(batch_data[0], dict):
                keys = batch_data[0].keys()
                stacked = {k: np.stack([elem[k] for elem in batch_data]) for k in keys}
                return jax.device_put(stacked)
            return {"data": jax.device_put(np.asarray(batch_data))}

        # Handle dict — device_put handles the entire pytree in one call
        if isinstance(batch_data, dict):
            return jax.device_put(batch_data)

        # Handle raw array/non-dict data
        return {"data": jax.device_put(batch_data)}

    def _make_fused_step(
        self, operators: list[OperatorNode]
    ) -> Callable[[dict, dict], tuple[dict, dict]]:
        """Create a fused step function for the operator chain.

        Chains all operators' _vmap_apply calls into a single nnx.jit-compiled
        function. Uses two strategies based on operator types:

        Deterministic chains: operators are closure-captured so they become
        trace constants in the compiled XLA kernel — no per-call
        to_tree/from_tree overhead.

        Stochastic chains: operators are passed as explicit arguments so
        nnx.jit properly manages their NNX state (including RNG) through
        the to_tree/from_tree cycle on every call.

        Args:
            operators: List of OperatorNode instances to fuse.

        Returns:
            Function: (data, states) -> (data, states)
        """
        # In fused steps, all operators are OperatorModule (not BatcherModule)
        ops = tuple(cast(OperatorModule, op_node.operator) for op_node in operators)
        has_stochastic = any(op.stochastic for op in ops)

        if has_stochastic:
            # Stochastic: pass operators as explicit arguments so nnx.jit
            # manages RNG state through to_tree/from_tree on every call.
            # Use nnx.cached_partial to cache graph traversal of operators —
            # only data/states are re-traced on each call, not the operator graph.
            @nnx.jit
            def fused_step_stochastic(operators_tuple: Any, data: Any, states: Any) -> Any:
                for op in operators_tuple:
                    data, states = op._vmap_apply(data, states)
                return data, states

            return nnx.cached_partial(fused_step_stochastic, ops)
        else:
            # Deterministic: closure-capture for zero overhead.
            # No mutable NNX state (RNG) to manage, so closure capture is
            # safe and faster (operators become trace constants).
            @nnx.jit
            def fused_step_deterministic(data: Any, states: Any) -> Any:
                for op in ops:
                    data, states = op._vmap_apply(data, states)
                return data, states

            return fused_step_deterministic

    def _fused_step_cache_key(self, operators: list[OperatorNode]) -> tuple[Any, ...]:
        """Build a stable operator-chain signature for fused-step caching."""
        return tuple(
            (
                operator_node.operator.__class__.__qualname__,
                id(operator_node.operator),
                repr(getattr(operator_node.operator, "config", None)),
            )
            for operator_node in operators
        )

    def _get_or_create_fused_step(
        self, operators: list[OperatorNode]
    ) -> Callable[[dict, dict], tuple[dict, dict]] | None:
        """Return cached fused-step for operator chain, compiling on first use."""
        if not operators:
            return None
        cache_key = self._fused_step_cache_key(operators)
        cached = self._runtime_state.fused_step_cache.get(cache_key)
        if cached is None:
            cached = self._make_fused_step(operators)
            self._runtime_state.fused_step_cache[cache_key] = cached
        return cached

    def _process_batch_from_source(
        self,
        batch_data: Any,
        operators: list[OperatorNode],
        fused_step: Callable | None = None,
    ) -> BatchView:
        """Convert raw batch data to Batch and apply operators.

        Uses a fused operator chain: passes raw dicts through all operators
        via _apply_on_raw(), creating only ONE Batch object at the end.
        This eliminates N-1 intermediate Batch objects and their associated
        nnx.Variable overhead.

        When fused_step is provided (JIT-compiled), the entire chain runs as
        a single compiled XLA kernel for maximum throughput.

        When no operators are present, keeps data as-is (numpy arrays stay
        numpy) to avoid the memory overhead of JAX's XLA allocator.

        Args:
            batch_data: Raw data from source's get_batch() (dict or list of dicts).
            operators: List of OperatorNode instances to apply.
            fused_step: Optional JIT-compiled step function from _make_fused_step.

        Returns:
            Processed lightweight BatchView after applying all operators.
        """
        if operators:
            data = self._convert_to_jax_arrays(batch_data)
            states: dict = {}
            if fused_step is not None:
                data, states = fused_step(data, states)
            else:
                for op_node in operators:
                    op = cast(OperatorModule, op_node.operator)
                    data, states = op._apply_on_raw(data, states)
        else:
            # No operators — keep as numpy for zero-copy memory efficiency.
            data = self._normalize_batch_data(batch_data)
            states = {}

        if data:
            batch_size = int(jax.tree.leaves(data)[0].shape[0])
        else:
            batch_size = 0

        return BatchView(data=data, states=states, batch_size=batch_size)

    @staticmethod
    def _normalize_batch_data(batch_data: Any) -> dict:
        """Normalize batch data to dict format without JAX conversion.

        Unlike _convert_to_jax_arrays, this keeps numpy arrays as numpy
        to avoid XLA allocator overhead when no JAX operations are needed.

        Args:
            batch_data: Data from source's get_batch().

        Returns:
            Dictionary with array values (numpy or JAX, whatever the source provides).
        """
        import numpy as np

        if isinstance(batch_data, dict):
            return batch_data

        if isinstance(batch_data, list):
            if not batch_data:
                return {}
            if isinstance(batch_data[0], dict):
                keys = batch_data[0].keys()
                return {k: np.stack([elem[k] for elem in batch_data]) for k in keys}
            return {"data": np.asarray(batch_data)}

        return {"data": batch_data}

    def _create_batch_first_iterator(self) -> Iterator[BatchView]:
        """Create batch-first iterator for optimal performance.

        This method retrieves batches directly from the source using
        get_batch(), bypassing element-by-element iteration. This is
        10-100x faster for sources that support it.

        When operators are present, creates a JIT-compiled fused step
        function that chains all operators into a single XLA kernel.
        The first batch triggers compilation; subsequent batches execute
        the compiled kernel with near-zero Python overhead.

        Yields:
            Batches created directly from source data.
        """
        assert self._source_node is not None, "No source node configured"
        assert self._batch_node is not None, "No batch node configured"
        source = self._source_node.source
        batch_size = self._batch_node.batch_size
        drop_remainder = self._batch_node.drop_remainder
        total_samples = len(source)
        operators = self._get_post_batch_operators()

        reset_fn = getattr(source, "reset", None)
        if reset_fn is not None:
            reset_fn()

        # Reuse compiled fused steps across epochs for unchanged operator chains.
        fused_step = self._get_or_create_fused_step(operators)

        # DataSourceModule subclasses provide get_batch at runtime
        get_batch: Callable[..., Any] = getattr(source, "get_batch")

        num_complete_batches = total_samples // batch_size
        remainder = total_samples % batch_size

        for _ in range(num_complete_batches):
            yield self._process_batch_from_source(get_batch(batch_size), operators, fused_step)

        if remainder > 0 and not drop_remainder:
            yield self._process_batch_from_source(get_batch(remainder), operators, fused_step)

    def _wrap_with_prefetch(self, raw_iter: Iterator[Any]) -> Iterator[Any]:
        """Wrap an iterator with prefetch buffers if configured.

        Returns:
            The original iterator if prefetch is disabled, otherwise a
            prefetch-wrapped iterator (optionally with device prefetch).
        """
        if self.prefetch_size <= 0:
            return raw_iter

        from datarax.control.prefetcher import DevicePrefetcher, Prefetcher

        source_iter = Prefetcher(buffer_size=self.prefetch_size).prefetch(raw_iter)
        if self.device_prefetch:
            source_iter = DevicePrefetcher(buffer_size=max(self.prefetch_size // 2, 1)).prefetch(
                source_iter
            )
        return source_iter

    def _create_iterator(self) -> Iterator[Any]:
        """Create the actual iterator.

        Returns:
            Iterator over processed batches

        Note: Uses batch-first path when available for optimal performance.
        Falls back to element-by-element iteration for complex pipelines.
        Wraps with Prefetcher when prefetch_size > 0 for background loading.
        """
        if self._can_use_batch_first():
            yield from self._iterate_batch_first()
            return

        yield from self._iterate_element_by_element()

    def _iterate_batch_first(self) -> Iterator[Any]:
        """Batch-first iteration path for optimal performance."""
        raw_iter = self._create_batch_first_iterator()
        source_iter = self._wrap_with_prefetch(raw_iter)
        try:
            # Increment _iteration_count on the consumer side (not in
            # _process_batch_from_source) so prefetcher background thread
            # doesn't race ahead of the actual consumption count.
            for batch in source_iter:
                self._iteration_count += 1
                yield batch
        finally:
            close_fn = getattr(source_iter, "close", None)
            if callable(close_fn):
                close_fn()

    def _iterate_element_by_element(self) -> Iterator[Any]:
        """Element-by-element fallback for complex pipelines."""
        if self._source_node is None:
            raise RuntimeError("No data source node found in DAG")
        source_iter = iter(self._source_node)

        # Temporarily disable caching for iteration
        # Caching during iteration can cause issues with buffering nodes
        saved_cache = self._cache
        self._cache = None

        try:
            for element in source_iter:
                key = None
                if self.rngs is not None:
                    self._iteration_count += 1
                    key = self.rngs()

                result = self._execute(self.graph, element, key)
                if result is not None:
                    yield result

            yield from self._flush_all_buffers()
        finally:
            self._cache = saved_cache
            close_fn = getattr(source_iter, "close", None)
            if callable(close_fn):
                close_fn()

    def _execute(self, node: Node, data: Any, key: jax.Array | None = None) -> Any:
        """Execute a node in the graph.

        Args:
            node: Node to execute
            data: Input data
            key: Optional RNG key

        Returns:
            Processed data or None
        """
        # Check cache if enabled and pipeline is cacheable
        # Skip caching for:
        # 1. Stochastic ops (use RNG key)
        # 2. Stateful ops (NNX modules with mutable state that affects output)
        cache_key = None
        can_cache = self._cache is not None and key is None

        if can_cache:
            # Lazy detection of stateful ops
            if self._has_stateful_ops is None:
                self._has_stateful_ops = self._detect_stateful_ops()
            can_cache = not self._has_stateful_ops

        if can_cache and self._cache is not None:
            cache_key = self._compute_cache_key(node, data)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Execute node
        if self._runtime_state.jit_execute is not None:
            result = self._runtime_state.jit_execute(node, data, key)
        else:
            result = node(data, key=key)

        # Cache result if deterministic and stateless
        if can_cache and result is not None and cache_key is not None and self._cache is not None:
            self._cache[cache_key] = result

        return result

    def _flush_all_buffers(self) -> Iterator[Batch]:
        """Flush all buffered nodes in the pipeline.

        This method properly drains all buffering nodes (BatchNode, ShuffleNode, etc.)
        by flushing them in order and passing results through the rest of the pipeline.

        Yields:
            Batches from flushed buffers
        """
        # Collect all nodes that need flushing
        flushable_nodes = self._collect_flushable_nodes(self.graph)

        # First, flush BatchNode if present and pass through rest of pipeline
        if self._batch_node is not None:
            batch_flush = self._batch_node.flush()
            if batch_flush is not None:
                # Find the portion of the graph after BatchNode
                post_batch_graph = self._get_post_batch_graph()
                if post_batch_graph is not None:
                    # Execute the flushed batch through the rest of the pipeline
                    key = self.rngs() if self.rngs is not None else None
                    result = self._execute(post_batch_graph, batch_flush, key)
                    if result is not None:
                        yield result
                else:
                    # No nodes after BatchNode, yield directly
                    yield batch_flush

        # Now drain all remaining flushable nodes (like ShuffleNode)
        # Keep flushing until all nodes return None
        while True:
            flushed_any = False
            for node in flushable_nodes:
                flush_fn = getattr(node, "flush", None)
                if flush_fn is not None:
                    flushed = flush_fn()
                    if flushed is not None:
                        flushed_any = True
                        yield flushed

            if not flushed_any:
                break

    def _collect_flushable_nodes(self, node: Node) -> list[Node]:
        """Collect all nodes in the graph that have a flush method.

        Args:
            node: Root node to search from

        Returns:
            List of flushable nodes in execution order
        """
        flushable = []

        def collect(n: Node) -> None:
            if n is None:
                return

            # Check if this node has flush (but skip BatchNode - handled separately)
            if hasattr(n, "flush") and n is not self._batch_node:
                flushable.append(n)

            # Recurse into composite nodes
            if isinstance(n, Sequential):
                for child in n.nodes:
                    collect(child)
            elif hasattr(n, "inner_node"):
                collect(n.inner_node)  # type: ignore[attr-defined]
            elif hasattr(n, "nodes"):
                for child in n.nodes:  # type: ignore[attr-defined]
                    collect(child)

        collect(node)
        return flushable

    def _get_post_batch_graph(self) -> Node | None:
        """Get the portion of the graph after BatchNode.

        Returns:
            Node representing the graph after BatchNode, or None if BatchNode is last
        """
        if not isinstance(self.graph, Sequential):
            # If graph is not sequential, check if it's just BatchNode
            if self.graph is self._batch_node:
                return None
            return self.graph

        # Find BatchNode in sequence and return everything after it
        nodes = list(self.graph.nodes)
        for i, node in enumerate(nodes):
            if node is self._batch_node:
                if i == len(nodes) - 1:
                    return None
                # Return remaining nodes as a new Sequential (or single node)
                remaining = nodes[i + 1 :]
                if len(remaining) == 1:
                    return remaining[0]
                return Sequential(remaining)

        # BatchNode not in sequence - return whole graph
        return self.graph

    def reset(self) -> None:
        """Reset the pipeline to initial state.

        This method resets:

        - Iteration and epoch counts
        - Cache if enabled
        - Data source and all stateful nodes
        - Iterator state
        """
        # Reset counters
        self._iteration_count = 0
        self._epoch_count = 0

        # Clear cache
        if self._cache is not None:
            self._cache.clear()

        # Reset the source if it exists
        if self._source_node is not None:
            if hasattr(self._source_node.source, "reset"):
                self._source_node.source.reset()  # type: ignore[attr-defined]
            # Iterator will be recreated on next iteration

        # Reset stateful nodes in the graph
        self._reset_node(self.graph)

        # Clear iterator
        self._iterator = nnx.data(None)

    def _reset_node(self, node: Node) -> None:
        """Recursively reset a node and its children.

        Args:
            node: Node to reset
        """
        if node is None:
            return

        # Reset the node if it has a reset method
        if hasattr(node, "reset"):
            node.reset()  # type: ignore[attr-defined]

        # Handle composite nodes
        if hasattr(node, "inner_node"):
            # For nodes like CacheNode that wrap another node
            self._reset_node(node.inner_node)  # type: ignore[attr-defined]
        elif hasattr(node, "left") and hasattr(node, "right"):
            # For Sequential nodes
            self._reset_node(node.left)  # type: ignore[attr-defined]
            self._reset_node(node.right)  # type: ignore[attr-defined]
        elif hasattr(node, "nodes"):
            # For Parallel nodes
            for n in node.nodes:  # type: ignore[attr-defined]
                self._reset_node(n)
        elif hasattr(node, "true_path") and hasattr(node, "false_path"):
            # For Branch nodes
            self._reset_node(node.true_path)  # type: ignore[attr-defined]
            self._reset_node(node.false_path)  # type: ignore[attr-defined]

    def _compile(self) -> None:
        """JIT compile the pipeline execution.

        Applies fusion first (replaces consecutive fusible nodes with
        FusedOperatorNode wrappers), then creates the JIT-compiled
        execution function with donate_argnums for memory reuse.
        """
        self._apply_fusion()

        def execute_fn(node: Any, data: Any, key: Any) -> Any:
            return node(data, key=key)

        # donate_argnums=(1,) donates the `data` argument — XLA reuses its
        # buffer for output since each pipeline step consumes its input.
        # This reduces peak memory by ~1 batch_size.
        self._runtime_state.jit_execute = nnx.jit(execute_fn, donate_argnums=(1,))

    def _compute_cache_key(self, node: Node, data: Any) -> int:
        """Compute cache key for node and data.

        Args:
            node: Node being executed
            data: Input data

        Returns:
            Hash key for caching
        """
        import hashlib

        node_id = id(node)

        if isinstance(data, jax.Array):
            # Use shape/dtype + object identity — no device-to-host transfer
            data_hash = hashlib.md5(
                f"{data.shape}:{data.dtype}:{id(data)}".encode(), usedforsecurity=False
            ).hexdigest()
        elif isinstance(data, dict):
            parts = []
            for k, v in sorted(data.items()):
                if hasattr(v, "shape"):
                    parts.append(f"{k}:{v.shape}:{v.dtype}")
                else:
                    parts.append(f"{k}:{hash(v)}")
            data_hash = hashlib.md5("|".join(parts).encode(), usedforsecurity=False).hexdigest()
        else:
            data_hash = str(hash(data))

        return hash((node_id, data_hash))

    def _validate_graph(self) -> None:
        """Validate the graph structure."""
        # Check for cycles (simplified check)
        visited = set()

        def visit(node: Any) -> None:
            if id(node) in visited:
                raise ValueError("Cycle detected in graph")
            visited.add(id(node))

            if hasattr(node, "nodes"):
                for n in node.nodes:
                    visit(n)
            elif hasattr(node, "inner_node"):
                visit(node.inner_node)

        if not isinstance(self.graph, Identity):
            visit(self.graph)

    def get_state(self) -> dict[str, Any]:
        """Get complete pipeline state for checkpointing.

        Returns:
            State dictionary
        """
        # Get NNX state
        nnx_state = nnx.state(self)

        # Get graph state
        graph_state = None
        if hasattr(self.graph, "get_state"):
            graph_state = self.graph.get_state()  # type: ignore[attr-defined]

        return {
            "nnx_state": nnx.to_pure_dict(nnx_state),
            "graph_state": graph_state,
            "iteration_count": self._iteration_count,
            "epoch_count": self._epoch_count,
            "cache": self._cache,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore pipeline state from checkpoint.

        Args:
            state: State dictionary from get_state()
        """
        if "nnx_state" in state:
            self._restore_nnx_state(state["nnx_state"])
        self._restore_graph_state(state.get("graph_state"))
        self._restore_counters_and_cache(state)

    def _restore_nnx_state(self, saved_state: Any) -> None:
        """Restore NNX state strictly.

        Raises:
            ValueError: If the saved state structure does not match current module structure.
        """
        if not isinstance(saved_state, dict):
            raise ValueError(f"Invalid NNX checkpoint state type: {type(saved_state).__name__}")
        saved_state = nnx.restore_int_paths(saved_state)
        current_state = nnx.state(self)
        nnx.replace_by_pure_dict(current_state, saved_state)
        nnx.update(self, current_state)

    def _restore_graph_state(self, graph_state: Any) -> None:
        """Restore graph-level checkpoint state when supported."""
        if graph_state is None:
            return
        if hasattr(self.graph, "set_state"):
            self.graph.set_state(graph_state)  # type: ignore[attr-defined]

    def _restore_counters_and_cache(self, state: dict[str, Any]) -> None:
        """Restore iteration counters and cache payload from checkpoint."""
        if "iteration_count" in state:
            self._iteration_count = state["iteration_count"]
        if "epoch_count" in state:
            self._epoch_count = state["epoch_count"]
        if "cache" in state and state["cache"] is not None:
            self._cache = state["cache"]

    def _collect_to_array(
        self,
        key: str = "image",
        device: jax.Device | None = None,  # type: ignore[name-defined]
    ) -> jax.Array:
        """Collect pipeline output to a single staged array on device.

        This method iterates through the entire pipeline, extracts the specified
        key from each batch, and concatenates all results into a single JAX array
        placed on the specified device.

        This is useful for:

        - Preparing data for JAX training loops that expect a single array
        - Converting pipeline output for use with lax.fori_loop
        - Pre-staging data on accelerator memory

        Args:
            key: The key to extract from each batch's data dictionary.
                Defaults to "image".
            device: Target device for the collected array. If None, uses
                the first available GPU, or CPU if no GPU is available.

        Returns:
            A single JAX array containing all collected data from the pipeline,
            with shape (total_samples, ...) where the first dimension is the
            concatenated batch dimension.

        Raises:
            KeyError: If the specified key is not found in a batch.
            ValueError: If the pipeline yields no batches.

        Example:
            ```python
            from datarax import from_source

            pipeline = from_source(mnist_source, batch_size=64)
            images = pipeline._collect_to_array(key="image")
            labels = pipeline._collect_to_array(key="label")

            # Use with JAX training loop
            def train_step(i, state):
                batch_images = lax.dynamic_slice(images, [i * 64, 0, 0, 0], [64, 28, 28, 1])
                return update(state, batch_images)

            final_state = lax.fori_loop(0, num_batches, train_step, init_state)
            ```
        """
        import jax.numpy as jnp

        batches = []
        for batch in self:
            # Use dict-like interface (works with both Batch and BatchView)
            if key not in batch:
                available = list(batch)
                raise KeyError(f"Key '{key}' not found in batch. Available keys: {available}")
            batches.append(batch[key])

        if not batches:
            raise ValueError("Pipeline yielded no batches")

        data = jnp.concatenate(batches, axis=0)

        if device is None:
            try:
                device = jax.devices("gpu")[0]
            except RuntimeError:
                device = jax.devices()[0]

        return jax.device_put(data, device)

    def clear_cache(self) -> None:
        """Clear all caches in the pipeline."""
        if self._cache is not None:
            self._cache.clear()

        # Clear node caches
        def clear_node_cache(node: Any) -> None:
            if isinstance(node, CacheNode):
                node.clear_cache()
            elif hasattr(node, "nodes"):
                for n in node.nodes:
                    clear_node_cache(n)

        clear_node_cache(self.graph)

    def _visualize(self) -> str:
        """Get text visualization of the pipeline.

        Returns:
            String representation of the graph
        """

        def visualize_node(node: Node, indent: int = 0) -> str:
            prefix = "  " * indent

            if isinstance(node, Sequential):
                parts = [f"{prefix}Sequential("]
                for n in node.nodes:
                    parts.append(visualize_node(n, indent + 1))
                parts.append(f"{prefix})")
                return "\n".join(parts)

            elif isinstance(node, Parallel):
                parts = [f"{prefix}Parallel("]
                for n in node.nodes:
                    parts.append(visualize_node(n, indent + 1))
                parts.append(f"{prefix})")
                return "\n".join(parts)

            else:
                return f"{prefix}{node}"

        return f"DAGExecutor(\n{visualize_node(self.graph, 1)}\n)"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DAGExecutor("
            f"name={self.name}, "
            f"iterations={self._iteration_count}, "
            f"epochs={self._epoch_count}, "
            f"cached={self.enable_caching}, "
            f"jit={self.jit_compile})"
        )


# Convenience functions for pipeline construction


def pipeline(*nodes: Node) -> DAGExecutor:
    """Create a pipeline from a sequence of nodes.

    Args:
        *nodes: Nodes to apply in sequence (first node must be a DataLoader)

    Returns:
        DAGExecutor instance

    Raises:
        ValueError: If no nodes provided or first node is not a DataLoader
    """
    if not nodes:
        raise ValueError("Pipeline must have at least one node")

    # Check that first node is a DataLoader
    first_node = nodes[0]
    if not isinstance(first_node, DataLoader):
        raise ValueError(f"First node must be a DataLoader, got {type(first_node).__name__}")

    # DataLoader already handles batching, so disable batch enforcement
    executor = DAGExecutor(enforce_batch=False)

    # Add all nodes to the executor
    for node in nodes:
        executor.add(node)
    return executor


def from_source(
    source: DataSourceModule,
    batch_size: int = 32,
    enforce_batch: bool = True,
    jit_compile: bool = False,
    prefetch_size: int = 2,
    device_prefetch: bool = False,
) -> DAGExecutor:
    """Create a pipeline starting from a data source.

    Args:
        source: Data source module
        batch_size: Size of batches to use (if enforce_batch is True)
        enforce_batch: Whether to enforce batch-first processing
        jit_compile: Whether to JIT compile the pipeline
        prefetch_size: Number of batches to prefetch ahead (0 = disabled)
        device_prefetch: Enable two-stage prefetch with async device transfer

    Returns:
        DAGExecutor with source
    """
    executor = DAGExecutor(
        enforce_batch=enforce_batch,
        jit_compile=jit_compile,
        prefetch_size=prefetch_size,
        device_prefetch=device_prefetch,
    )
    executor.add(source)

    # Add a batch node if batch enforcement is enabled
    if enforce_batch:
        executor.batch(batch_size)

    return executor
