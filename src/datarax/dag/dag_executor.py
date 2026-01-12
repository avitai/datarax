"""
Complete DAG pipeline executor for Datarax.

This module provides the main DAGExecutor class for executing complex
data processing workflows as directed acyclic graphs (DAGs).

COMPLETE IMPLEMENTATION following TDD principles.
"""

from typing import Any, Callable, Literal, Iterator
import jax
import flax.nnx as nnx
from flax.nnx.transforms.compilation import JitFn

from datarax.dag.nodes import (
    Node,
    Identity,
    Sequential,
    Parallel,
    Branch,
    Merge,
    DataSourceNode,
    BatchNode,
    OperatorNode,
    ShuffleNode,
    CacheNode,
    DataLoader,
    SamplerNode,
    SharderNode,
)
from datarax.core.module import DataraxModule
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.batcher import BatcherModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule
from datarax.typing import Batch


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
        name: str | None = None,
    ):
        """Initialize DAG executor.

        Args:
            graph: Initial graph node or None for empty
            rngs: RNG state for the pipeline (lazy - only used if stochastic ops exist)
            enable_caching: Whether to cache intermediate results
            jit_compile: Whether to JIT compile the pipeline
            enforce_batch: Whether to enforce batch-first processing
            name: Optional name for the executor
        """
        super().__init__()

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

        # Configuration - lazy RNG (only created when needed)
        # Store rngs parameter but defer creation until pipeline needs it
        self._rngs_seed = rngs  # Store for lazy initialization
        self._rngs: nnx.Rngs | None = None  # Lazily initialized
        self._needs_rng: bool | None = None  # Cached stochastic detection
        self._has_stateful_ops: bool | None = None  # Cached stateful detection
        self.enable_caching = enable_caching
        self.jit_compile = jit_compile
        self.enforce_batch = enforce_batch
        self.name = name or "DAGExecutor"

        # Pipeline state - plain Python (not nnx.Variable to avoid TraceContextError)
        # These are used for iteration tracking, not gradient computation
        self._iteration_count: int = 0
        self._epoch_count: int = 0

        # Execution state - plain Python attributes (not tracked by NNX)
        self._source_node: DataSourceNode | None = None
        self._batch_node: BatchNode | None = None
        self._iterator: Iterator[Batch] | None = None
        self._cache: dict[int, Any] | None = {} if enable_caching else None

        # JIT compiled execution function
        self._jit_execute: JitFn | None = None
        if jit_compile:
            self._compile()

        # Validate graph
        self._validate_graph()

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
            # Use object.__setattr__ to bypass NNX checking for plain Python attribute
            object.__setattr__(self, "_rngs", self._rngs_seed)
            return self._rngs

        # Check if pipeline needs RNG (cache result)
        if self._needs_rng is None:
            self._needs_rng = self._detect_stochastic_ops()

        # Only create default RNG if pipeline has stochastic ops
        if self._needs_rng:
            new_rngs = nnx.Rngs(0)
            object.__setattr__(self, "_rngs", new_rngs)
            return self._rngs

        # Pipeline is deterministic and user didn't provide rngs - return None
        return None

    def _detect_stochastic_ops(self) -> bool:
        """Detect if pipeline contains any stochastic operations.

        Traverses the DAG to check for stochastic nodes/operators.
        This follows Grain's pattern of separating deterministic (map)
        from stochastic (random_map) operations.

        Returns:
            True if any stochastic operations exist, False otherwise.
        """

        def check_node(node: Node) -> bool:
            """Recursively check if node or children are stochastic."""
            # Check OperatorNode for stochastic config
            if isinstance(node, OperatorNode):
                operator = node.operator
                if hasattr(operator, "config"):
                    if getattr(operator.config, "stochastic", False):
                        return True

            # Check ShuffleNode (always stochastic)
            if isinstance(node, ShuffleNode):
                return True

            # Check SamplerNode with shuffle
            if isinstance(node, SamplerNode):
                sampler = node.sampler
                if hasattr(sampler, "config"):
                    if getattr(sampler.config, "stochastic", False):
                        return True

            # Check DataSourceNode with shuffle
            if isinstance(node, DataSourceNode):
                source = node.source
                if hasattr(source, "config"):
                    if getattr(source.config, "stochastic", False):
                        return True

            # Recursively check composite nodes
            if isinstance(node, Sequential):
                return any(check_node(n) for n in node.nodes)
            if isinstance(node, Parallel):
                return any(check_node(n) for n in node.nodes)
            if isinstance(node, Branch):
                return check_node(node.true_path) or check_node(node.false_path)
            # Merge doesn't have child nodes that need checking

            return False

        # Check main graph
        return check_node(self.graph)

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

        def check_node(node: Node) -> bool:
            """Recursively check if node or children are stateful."""
            # Check OperatorNode for explicit stateful config
            if isinstance(node, OperatorNode):
                operator = node.operator
                if hasattr(operator, "config"):
                    # Explicit stateful flag takes precedence
                    if getattr(operator.config, "stateful", False):
                        return True

            # Recursively check composite nodes
            if isinstance(node, Sequential):
                return any(check_node(n) for n in node.nodes)
            if isinstance(node, Parallel):
                return any(check_node(n) for n in node.nodes)
            if isinstance(node, Branch):
                return check_node(node.true_path) or check_node(node.false_path)
            # Merge doesn't have child nodes that need checking

            return False

        # Check main graph
        return check_node(self.graph)

    def add(self, node: Node) -> "DAGExecutor":
        """Add a node to the pipeline.

        Extends the current graph sequentially.

        Args:
            node: Node or module to add

        Returns:
            Self for chaining
        """
        # Convert module to node if needed
        if isinstance(node, DataSourceModule):
            node = DataSourceNode(node)
            self._source_node = nnx.data(node)
            # Don't add source to the execution graph, just store it
            return self
        elif isinstance(node, OperatorModule):
            # Check if stochastic for OperatorNode vs OperatorNode
            if hasattr(node, "config") and getattr(node.config, "stochastic", False):
                node = OperatorNode(node)
            else:
                node = OperatorNode(node)
        elif isinstance(node, BatcherModule):
            node = OperatorNode(node)
        elif isinstance(node, SamplerModule):
            node = SamplerNode(node)
        elif isinstance(node, SharderModule):
            node = SharderNode(node)
        elif isinstance(node, DataSourceNode):
            # Handle DataSourceNode directly
            self._source_node = nnx.data(node)
            # Don't add source to the execution graph, just store it
            return self
        elif not isinstance(node, Node):
            raise ValueError(f"Unsupported type: {type(node)}")

        # Check for batch enforcement
        if self.enforce_batch:
            if isinstance(node, BatchNode):
                self._batch_node = nnx.data(node)
            elif isinstance(node, OperatorNode | OperatorNode):
                if self._batch_node is None:
                    raise ValueError(
                        "Batch-first enforcement: Add BatchNode before transforms. "
                        "Use executor.add(BatchNode(batch_size)) first."
                    )

        # Update graph
        if isinstance(self.graph, Identity):
            self.graph: Node = node
        else:
            self.graph: Node = self.graph >> node

        # Invalidate caches when graph structure changes
        self._jit_execute = None
        self._needs_rng = None  # Force re-detection of stochastic ops
        self._has_stateful_ops = None  # Force re-detection of stateful ops

        return self

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

    def __iter__(self) -> Iterator[Batch]:
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

    def __next__(self) -> Batch:
        """Get next batch from the pipeline.

        Returns:
            Next processed batch

        Raises:
            StopIteration: When pipeline is exhausted
        """
        if self._iterator is None:
            raise ValueError("Call iter() first or use in for loop")

        return next(self._iterator)

    def _create_iterator(self) -> Iterator[Batch]:
        """Create the actual iterator.

        Returns:
            Iterator over processed batches

        Note: Caching is disabled during iteration because:
        1. Each element may have unique data that should not be cached
        2. Buffering nodes (BatchNode) aggregate multiple elements
        3. The cache key computation is designed for single-call, not streaming
        """
        # Get source iterator
        if self._source_node is None:
            raise RuntimeError("No data source node found in DAG")
        source_iter = iter(self._source_node)

        # Temporarily disable caching for iteration
        # Caching during iteration can cause issues with buffering nodes
        saved_cache = self._cache
        self._cache = None

        try:
            # Process through pipeline
            for element in source_iter:
                # Get RNG key for this iteration
                key = None
                if self.rngs is not None:
                    self._iteration_count += 1
                    key = self.rngs()

                # Execute pipeline
                result = self._execute(self.graph, element, key)

                # Yield if we got a result (batching might buffer)
                if result is not None:
                    yield result

            # Flush any remaining data from all buffered nodes
            yield from self._flush_all_buffers()
        finally:
            # Restore cache for single-call operations
            self._cache = saved_cache

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

        if can_cache:
            cache_key = self._compute_cache_key(node, data)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Execute node
        if self._jit_execute is not None:
            result = self._jit_execute(node, data, key)
        else:
            result = node(data, key=key)

        # Cache result if deterministic and stateless
        if can_cache and result is not None and cache_key is not None:
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
                if hasattr(node, "flush"):
                    flushed = node.flush()
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
        """JIT compile the pipeline execution."""

        def execute_fn(node, data, key):
            return node(data, key=key)

        self._jit_execute = nnx.jit(execute_fn)

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
            data_hash = hashlib.md5(
                f"{data.shape}:{data.dtype}:{float(data.sum())}".encode(), usedforsecurity=False
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

        def visit(node):
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
        # Restore NNX state
        if "nnx_state" in state:
            current_state = nnx.state(self)
            try:
                nnx.replace_by_pure_dict(current_state, state["nnx_state"])
                nnx.update(self, current_state)
            except ValueError as e:
                # If structures don't match exactly, try a more selective update
                # This can happen when loading state from slightly different pipeline versions
                import warnings

                warnings.warn(
                    f"State structure mismatch during restore: {e}. Attempting selective restore."
                )

                # Only update matching keys
                saved_dict = state["nnx_state"]
                current_dict = nnx.to_pure_dict(current_state)

                def selective_update(current: Any, saved: Any, path: str = "") -> Any:
                    """Recursively update only matching paths."""
                    if isinstance(current, dict) and isinstance(saved, dict):
                        result = {}
                        for key in current:
                            if key in saved:
                                new_path = f"{path}.{key}" if path else key
                                result[key] = selective_update(current[key], saved[key], new_path)
                            else:
                                # Keep current value if not in saved
                                result[key] = current[key]
                        return result
                    elif isinstance(current, list | tuple) and isinstance(saved, list | tuple):
                        result = []
                        for i, (c_item, s_item) in enumerate(zip(current, saved)):
                            result.append(selective_update(c_item, s_item, f"{path}[{i}]"))
                        # Return same type as current
                        return type(current)(result)
                    else:
                        # Leaf node - update if types match
                        if type(current) == type(saved):
                            return saved
                        return current

                updated_dict = selective_update(current_dict, saved_dict)
                assert isinstance(updated_dict, dict)
                nnx.replace_by_pure_dict(current_state, updated_dict)
                nnx.update(self, current_state)

        # Restore graph state
        if "graph_state" in state and state["graph_state"] is not None:
            if hasattr(self.graph, "set_state"):
                self.graph.set_state(state["graph_state"])  # type: ignore[attr-defined]

        # Restore counters
        if "iteration_count" in state:
            self._iteration_count = state["iteration_count"]
        if "epoch_count" in state:
            self._epoch_count = state["epoch_count"]

        # Restore cache
        if "cache" in state and state["cache"] is not None:
            self._cache = state["cache"]

    def clear_cache(self) -> None:
        """Clear all caches in the pipeline."""
        if self._cache is not None:
            self._cache.clear()

        # Clear node caches
        def clear_node_cache(node):
            if isinstance(node, CacheNode):
                node.clear_cache()
            elif hasattr(node, "nodes"):
                for n in node.nodes:
                    clear_node_cache(n)

        clear_node_cache(self.graph)

    def visualize(self) -> str:
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
) -> DAGExecutor:
    """Create a pipeline starting from a data source.

    Args:
        source: Data source module
        batch_size: Size of batches to use (if enforce_batch is True)
        enforce_batch: Whether to enforce batch-first processing
        jit_compile: Whether to JIT compile the pipeline

    Returns:
        DAGExecutor with source
    """
    executor = DAGExecutor(enforce_batch=enforce_batch, jit_compile=jit_compile)
    executor.add(source)

    # Add a batch node if batch enforcement is enabled
    if enforce_batch:
        executor.batch(batch_size)

    return executor
