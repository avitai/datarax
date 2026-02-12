"""
Complete DAG pipeline executor for Datarax.

This module provides the main DAGExecutor class for executing complex
data processing workflows as directed acyclic graphs (DAGs).

COMPLETE IMPLEMENTATION following TDD principles.
"""

from typing import Any, Literal
from collections.abc import Callable, Iterator
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
    FusedOperatorNode,
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
        prefetch_size: int = 2,
        name: str | None = None,
    ):
        """Initialize DAG executor.

        Args:
            graph: Initial graph node or None for empty
            rngs: RNG state for the pipeline (lazy - only used if stochastic ops exist)
            enable_caching: Whether to cache intermediate results
            jit_compile: Whether to JIT compile the pipeline
            enforce_batch: Whether to enforce batch-first processing
            prefetch_size: Number of batches to prefetch ahead (0 = disabled)
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
        self.prefetch_size = prefetch_size
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

        def collect_operators(node: Node):
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
        ops = tuple(op_node.operator for op_node in operators)
        has_stochastic = any(op.stochastic for op in ops)

        if has_stochastic:
            # Stochastic: pass operators as explicit arguments so nnx.jit
            # manages RNG state through to_tree/from_tree on every call.
            @nnx.jit
            def fused_step_stochastic(operators_tuple, data, states):
                for op in operators_tuple:
                    data, states = op._vmap_apply(data, states)
                return data, states

            def call_fused(data, states):
                return fused_step_stochastic(ops, data, states)

            return call_fused
        else:
            # Deterministic: closure-capture for zero overhead.
            # No mutable NNX state (RNG) to manage, so closure capture is
            # safe and faster (operators become trace constants).
            @nnx.jit
            def fused_step_deterministic(data, states):
                for op in ops:
                    data, states = op._vmap_apply(data, states)
                return data, states

            return fused_step_deterministic

    def _process_batch_from_source(
        self,
        batch_data: Any,
        operators: list[OperatorNode],
        fused_step: Callable | None = None,
    ) -> Batch:
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
            Processed Batch after applying all operators.
        """
        if operators:
            data = self._convert_to_jax_arrays(batch_data)
            states: dict = {}
            if fused_step is not None:
                data, states = fused_step(data, states)
            else:
                for op_node in operators:
                    data, states = op_node.operator._apply_on_raw(data, states)
        else:
            # No operators — keep as numpy for zero-copy memory efficiency.
            data = self._normalize_batch_data(batch_data)
            states = {}

        # Single Batch creation at the end (eliminates N-1 intermediates)
        return Batch.from_parts(data=data, states=states, validate=False)

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

    def _create_batch_first_iterator(self) -> Iterator[Batch]:
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
        source = self._source_node.source
        batch_size = self._batch_node.batch_size
        drop_remainder = self._batch_node.drop_remainder
        total_samples = len(source)
        operators = self._get_post_batch_operators()

        if hasattr(source, "reset"):
            source.reset()

        # Create JIT-compiled fused step for operator chains
        fused_step = self._make_fused_step(operators) if operators else None

        num_complete_batches = total_samples // batch_size
        remainder = total_samples % batch_size

        for _ in range(num_complete_batches):
            yield self._process_batch_from_source(
                source.get_batch(batch_size), operators, fused_step
            )

        if remainder > 0 and not drop_remainder:
            yield self._process_batch_from_source(
                source.get_batch(remainder), operators, fused_step
            )

    def _create_iterator(self) -> Iterator[Batch]:
        """Create the actual iterator.

        Returns:
            Iterator over processed batches

        Note: Uses batch-first path when available for optimal performance.
        Falls back to element-by-element iteration for complex pipelines.
        Wraps with Prefetcher when prefetch_size > 0 for background loading.
        """
        # Try batch-first path for optimal performance
        if self._can_use_batch_first():
            raw_iter = self._create_batch_first_iterator()
            if self.prefetch_size > 0:
                from datarax.control.prefetcher import Prefetcher

                source_iter = Prefetcher(buffer_size=self.prefetch_size).prefetch(raw_iter)
            else:
                source_iter = raw_iter
            # Increment _iteration_count on the consumer side (not in
            # _process_batch_from_source) so prefetcher background thread
            # doesn't race ahead of the actual consumption count.
            for batch in source_iter:
                self._iteration_count += 1
                yield batch
            return

        # Fallback to element-by-element iteration
        # (needed for complex pipelines with shuffle nodes, etc.)
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
        """JIT compile the pipeline execution.

        Applies fusion first (replaces consecutive fusible nodes with
        FusedOperatorNode wrappers), then creates the JIT-compiled
        execution function with donate_argnums for memory reuse.
        """
        self._apply_fusion()

        def execute_fn(node, data, key):
            return node(data, key=key)

        # donate_argnums=(1,) donates the `data` argument — XLA reuses its
        # buffer for output since each pipeline step consumes its input.
        # This reduces peak memory by ~1 batch_size.
        self._jit_execute = nnx.jit(execute_fn, donate_argnums=(1,))

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

    def collect_to_array(
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
            images = pipeline.collect_to_array(key="image")
            labels = pipeline.collect_to_array(key="label")

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
    prefetch_size: int = 2,
) -> DAGExecutor:
    """Create a pipeline starting from a data source.

    Args:
        source: Data source module
        batch_size: Size of batches to use (if enforce_batch is True)
        enforce_batch: Whether to enforce batch-first processing
        jit_compile: Whether to JIT compile the pipeline
        prefetch_size: Number of batches to prefetch ahead (0 = disabled)

    Returns:
        DAGExecutor with source
    """
    executor = DAGExecutor(
        enforce_batch=enforce_batch, jit_compile=jit_compile, prefetch_size=prefetch_size
    )
    executor.add(source)

    # Add a batch node if batch enforcement is enabled
    if enforce_batch:
        executor.batch(batch_size)

    return executor
