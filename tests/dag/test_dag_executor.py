"""
Comprehensive test suite for DAGExecutor.

This test suite provides extensive coverage of the DAGExecutor class and related
functionality, including initialization, pipeline construction, execution,
caching, state management, and integration with various node types.
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any, Iterator

from datarax.dag.dag_executor import DAGExecutor, OperatorNode, pipeline
from datarax.typing import Element
from datarax.core.element_batch import Batch
from datarax.dag.nodes import (
    Identity,
    Sequential,
    Parallel,
    Branch,
    Merge,
    SplitFields,
    parallel,
    branch,
    DataLoader,
    DataSourceNode,
)
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule


def batch_allclose(a: Batch | Any, b: Batch | Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare two Batch objects or arrays using jax.tree.map.

    Works with both Batch objects (compares data.value PyTrees) and raw arrays.
    """
    # Extract data if Batch objects
    a_data = a.data.get_value() if isinstance(a, Batch) else a
    b_data = b.data.get_value() if isinstance(b, Batch) else b

    # Use tree.map to compare all leaves, then reduce with all()
    comparisons = jax.tree.map(
        lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol), a_data, b_data
    )
    return all(jax.tree.leaves(comparisons))


def batch_mul(batch: Batch, scalar: float) -> Batch:
    """Multiply all arrays in a Batch by a scalar."""
    new_data = jax.tree.map(lambda x: x * scalar, batch.data.get_value())
    return Batch.from_parts(
        data=new_data,
        states=batch.states.get_value(),
        metadata_list=batch._metadata_list,
        validate=False,
    )


def batch_add(batch1: Batch, batch2: Batch) -> Batch:
    """Add corresponding arrays from two Batches."""
    new_data = jax.tree.map(lambda x, y: x + y, batch1.data.get_value(), batch2.data.get_value())
    return Batch.from_parts(
        data=new_data,
        states=batch1.states.get_value(),
        metadata_list=batch1._metadata_list,
        validate=False,
    )


class MockDataSource(DataSourceModule):
    """Mock data source for testing."""

    # REQUIRED: Annotate data attribute with nnx.data() to prevent NNX container errors
    data: list = nnx.data()

    def __init__(self, size=10, name="mock_source"):
        config = StructuralConfig(stochastic=False)
        super().__init__(config, name=name)
        self.size = size
        self.data = [jnp.array([i, i + 1, i + 2]) for i in range(size)]
        self.index = 0

    def __iter__(self) -> Iterator[Element]:
        self.index = 0
        return self

    def __next__(self) -> Element:
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result


# Test fixtures and helper classes
class MockOperator(OperatorModule):
    """Mock operator for testing."""

    def __init__(
        self, multiplier: float = 2.0, *, rngs: nnx.Rngs | None = None, name: str = "mock_operator"
    ):
        config = OperatorConfig(stochastic=False)
        super().__init__(config, rngs=rngs, name=name)
        self.multiplier = multiplier

    @property
    def call_count(self):
        """Get the call count from the parent's _iteration_count."""
        return self._iteration_count

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: Any,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], Any]:
        """Apply transformation to data."""
        if isinstance(data, dict):
            result = {k: v * self.multiplier for k, v in data.items()}
        else:
            result = data * self.multiplier
        return result, state, metadata


class IterationAwareOperator(OperatorModule):
    """Operator that uses iteration count to produce different output per call.

    Uses OperatorModule's _iteration_count (incremented in __call__ before apply_batch)
    to add iteration-dependent offset. The count is passed via stats to keep apply() pure.
    """

    def __init__(self, *, rngs: nnx.Rngs | None = None, name: str = "iteration_aware_operator"):
        config = OperatorConfig(stochastic=False)
        super().__init__(config, rngs=rngs, name=name)

    @property
    def call_count(self):
        """Get the call count from the parent's _iteration_count."""
        return self._iteration_count

    @property
    def state(self):
        """Alias for iteration count (for test compatibility)."""
        return self._iteration_count

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: Any,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], Any]:
        """Apply transformation with iteration-dependent offset (pure function)."""
        iter_count = stats.get("iteration_count", 0.0) if stats else 0.0
        result = jax.tree.map(lambda x: x + iter_count, data)
        return result, state, metadata

    def apply_batch(self, batch, stats=None):
        """Pass iteration count to apply via stats (outside traced context)."""
        if stats is None:
            stats = {}
        stats = {**stats, "iteration_count": float(self._iteration_count)}
        return super().apply_batch(batch, stats=stats)


@pytest.fixture
def sample_data():
    """Sample data for testing as Batch."""
    data = {"_array": jnp.ones((4, 8, 8, 3))}
    states = {"_array": jnp.zeros((4,))}
    return Batch.from_parts(data=data, states=states, validate=False)


@pytest.fixture
def sample_dict_data():
    """Sample dictionary data for testing as Batch."""
    data = {
        "image": jnp.ones((4, 8, 8, 3)),
        "label": jnp.array([0, 1, 2, 3]),
        "metadata": jnp.ones((4, 2)),
    }
    states = {
        "image": jnp.zeros((4,)),
        "label": jnp.zeros((4,)),
        "metadata": jnp.zeros((4,)),
    }
    return Batch.from_parts(data=data, states=states, validate=False)


@pytest.fixture
def mock_operator():
    """Mock operator fixture."""
    return MockOperator(multiplier=2.0)


class RandomOperator(OperatorModule):
    """Operator that uses randomness for testing."""

    def __init__(
        self,
        noise_scale: float = 0.1,
        *,
        rngs: nnx.Rngs | None = None,
        name: str = "random_operator",
    ):
        config = OperatorConfig(stochastic=True, stream_name="default")
        # Ensure rngs is provided for stochastic operator
        if rngs is None:
            rngs = nnx.Rngs(0)
        super().__init__(config, rngs=rngs, name=name)
        self.noise_scale = noise_scale

    @property
    def call_count(self):
        """Get the call count from the parent's _iteration_count."""
        return self._iteration_count

    def generate_random_params(self, rng: jax.Array, data_shapes: Any) -> jax.Array:
        """Generate random keys for batch elements."""
        batch_sizes = jax.tree.leaves(
            jax.tree.map(lambda s: s[0], data_shapes, is_leaf=lambda x: isinstance(x, tuple))
        )
        batch_size = batch_sizes[0] if batch_sizes else 1
        return jax.random.split(rng, batch_size)

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: Any,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], Any]:
        """Apply random noise to data."""
        if random_params is not None:

            def add_noise(x):
                noise = jax.random.normal(random_params, x.shape) * self.noise_scale
                return x + noise

            result = jax.tree.map(add_noise, data)
        else:
            result = data
        return result, state, metadata


class TestDAGExecutorInitialization:
    """Test DAGExecutor initialization and configuration."""

    def test_default_initialization(self):
        """Test default DAGExecutor initialization."""
        executor = DAGExecutor(enforce_batch=False)

        assert isinstance(executor.graph, Identity)
        # rngs is lazy - None for deterministic pipelines (no stochastic ops)
        assert executor.rngs is None  # No stochastic ops = no RNG needed
        assert executor.enable_caching is True
        assert executor.jit_compile is False
        assert executor._iteration_count == 0
        assert executor._cache == {}
        assert executor._jit_execute is None

    def test_initialization_with_graph(self, mock_operator):
        """Test initialization with initial graph."""
        executor = DAGExecutor(mock_operator, enforce_batch=False)

        assert isinstance(executor.graph, OperatorNode)
        assert executor.graph.operator is mock_operator

    def test_initialization_with_node(self):
        """Test initialization with Node instance."""
        identity = Identity()
        executor = DAGExecutor(identity, enforce_batch=False)

        assert executor.graph is identity

    def test_initialization_with_rngs(self):
        """Test initialization with custom RNGs."""
        rngs = nnx.Rngs(42)
        executor = DAGExecutor(rngs=rngs, enforce_batch=False)

        assert executor.rngs is rngs

    def test_initialization_with_caching_disabled(self):
        """Test initialization with caching disabled."""
        executor = DAGExecutor(enable_caching=False, enforce_batch=False)

        assert executor.enable_caching is False
        assert executor._cache is None

    def test_initialization_with_jit_enabled(self):
        """Test initialization with JIT compilation enabled."""
        executor = DAGExecutor(jit_compile=True, enforce_batch=False)

        assert executor.jit_compile is True
        assert executor._jit_execute is not None

    def test_initialization_all_options(self, mock_operator):
        """Test initialization with all options specified."""
        rngs = nnx.Rngs(123)
        executor = DAGExecutor(
            graph=mock_operator,
            rngs=rngs,
            enable_caching=False,
            jit_compile=True,
            enforce_batch=False,
        )

        assert isinstance(executor.graph, OperatorNode)
        assert executor.rngs is rngs
        assert executor.enable_caching is False
        assert executor.jit_compile is True


class TestDAGExecutorPipelineConstruction:
    """Test pipeline construction methods."""

    def test_add_transform_to_empty(self, mock_operator):
        """Test adding transform to empty executor."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(enforce_batch=False)
        # Add BatchNode first as required by batch-first principle
        executor.add(BatchNode(batch_size=4))
        result = executor.add(mock_operator)

        assert result is executor  # Returns self for chaining
        # Graph should now be Sequential with BatchNode and OperatorNode
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 2
        assert isinstance(executor.graph.nodes[0], BatchNode)
        assert isinstance(executor.graph.nodes[1], OperatorNode)
        assert executor.graph.nodes[1].operator is mock_operator

    def test_add_node_to_empty(self):
        """Test adding node to empty executor."""
        executor = DAGExecutor(enforce_batch=False)
        identity = Identity()
        result = executor.add(identity)

        assert result is executor
        assert executor.graph is identity

    def test_add_sequential_transforms(self):
        """Test adding multiple transforms sequentially."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Add BatchNode first, then transforms
        executor.add(BatchNode(batch_size=4)).add(OperatorNode(t1)).add(OperatorNode(t2))

        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 3  # BatchNode + 2 transforms

    def test_add_invalidates_jit(self):
        """Test that adding transforms invalidates JIT compilation."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(jit_compile=True, enforce_batch=False)

        # Add BatchNode first, then transform
        executor.add(BatchNode(batch_size=4)).add(OperatorNode(MockOperator()))

        assert executor._jit_execute is None

    def test_parallel_construction(self):
        """Test parallel pipeline construction."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")
        t3 = MockOperator(multiplier=4.0, name="t3")

        # Add BatchNode first, then parallel transforms
        executor.add(BatchNode(batch_size=4))
        result = executor.parallel([OperatorNode(t1), OperatorNode(t2), OperatorNode(t3)])

        assert result is executor
        # Graph should be Sequential with BatchNode followed by Parallel
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 2
        assert isinstance(executor.graph.nodes[0], BatchNode)
        assert isinstance(executor.graph.nodes[1], Parallel)
        assert len(executor.graph.nodes[1].nodes) == 3

    def test_branch_construction(self):
        """Test branch construction."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(enforce_batch=False)
        # Condition must access Batch data: x.data.get_value()["_array"].shape[0]
        condition = lambda x: x.data.get_value()["_array"].shape[0] > 2
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        # Add BatchNode first, then branch
        executor.add(BatchNode(batch_size=4))
        result = executor.branch(condition, OperatorNode(true_path), OperatorNode(false_path))

        assert result is executor
        # Graph should be Sequential with BatchNode followed by Branch
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 2
        assert isinstance(executor.graph.nodes[0], BatchNode)
        assert isinstance(executor.graph.nodes[1], Branch)
        assert executor.graph.nodes[1].condition is condition

    def test_merge_construction(self):
        """Test merge construction."""
        executor = DAGExecutor(enforce_batch=False)

        result = executor.merge(strategy="sum", axis=0)

        assert result is executor
        assert isinstance(executor.graph, Merge)
        assert executor.graph.strategy == "sum"
        assert executor.graph.axis == 0

    def test_cache_construction(self):
        """Test cache construction."""
        from datarax.dag.nodes import BatchNode, CacheNode

        executor = DAGExecutor(enforce_batch=False)
        # Add BatchNode first, then transform
        executor.add(BatchNode(batch_size=4)).add(OperatorNode(MockOperator()))

        result = executor.cache(cache_size=50)

        assert result is executor
        # Graph should now be a CacheNode wrapping the Sequential
        assert isinstance(executor.graph, CacheNode)
        # The CacheNode should wrap the previous Sequential
        assert isinstance(executor.graph.inner_node, Sequential)
        assert executor.graph.cache_size == 50

    def test_complex_pipeline_construction(self):
        """Test complex pipeline construction."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(enforce_batch=False)

        # Build: batch -> normalize -> parallel(aug1, aug2) -> merge -> crop
        normalize = MockOperator(multiplier=0.5, name="normalize")
        aug1 = MockOperator(multiplier=1.1, name="aug1")
        aug2 = MockOperator(multiplier=0.9, name="aug2")
        crop = MockOperator(multiplier=1.0, name="crop")

        # Add BatchNode first, then build pipeline
        executor.add(BatchNode(batch_size=4)).add(OperatorNode(normalize)).parallel(
            [OperatorNode(aug1), OperatorNode(aug2)]
        ).merge("mean").add(OperatorNode(crop))

        # Should be Sequential with 5 nodes (batch + 4 transform nodes)
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 5

    def test_operator_overloading_construction(self):
        """Test pipeline construction using >> operator."""
        from datarax.dag.nodes import BatchNode

        # Test: from_source(source) >> op1 >> op2
        # Note: from_source returns a DAGExecutor. We verify DAGExecutor >> Node behavior.

        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Initial setup: DAGExecutor with just BatchNode (simulating from_source)
        executor.add(BatchNode(batch_size=4))

        # Use >> operator to add nodes
        # This calls DAGExecutor.__rshift__
        result = executor >> OperatorNode(t1) >> OperatorNode(t2)

        assert result is executor
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 3
        # Node 0 is BatchNode
        assert isinstance(executor.graph.nodes[1], OperatorNode)
        assert executor.graph.nodes[1].operator is t1
        assert isinstance(executor.graph.nodes[2], OperatorNode)
        assert executor.graph.nodes[2].operator is t2


class TestDAGExecutorExecution:
    """Test DAG execution functionality."""

    def test_basic_execution(self, sample_dict_data):
        """Test basic pipeline execution on pre-batched data."""
        # For testing transform execution directly on batched data,
        # we don't use BatchNode since data is already batched
        executor = DAGExecutor(enforce_batch=False)
        transform = MockOperator(multiplier=3.0)
        executor.add(OperatorNode(transform))

        result = executor(sample_dict_data)

        # Check each field in the result
        assert "image" in result
        assert "label" in result
        assert batch_allclose(result["image"], sample_dict_data["image"] * 3.0)
        assert batch_allclose(result["label"], sample_dict_data["label"] * 3.0)
        # Now tracking at batch level, so transform is called once per batch
        assert transform.call_count == 1

    def test_sequential_execution(self, sample_dict_data):
        """Test sequential pipeline execution on pre-batched data."""
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        executor.add(OperatorNode(t1)).add(OperatorNode(t2))
        result = executor(sample_dict_data)

        # Check transformations were applied sequentially
        assert batch_allclose(result["image"], sample_dict_data["image"] * 2.0 * 3.0)
        assert batch_allclose(result["label"], sample_dict_data["label"] * 2.0 * 3.0)
        # Now tracking at batch level, so each transform is called once per batch
        assert t1.call_count == 1
        assert t2.call_count == 1

    def test_parallel_execution(self, sample_dict_data):
        """Test parallel pipeline execution on pre-batched data."""
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        executor.parallel([OperatorNode(t1), OperatorNode(t2)])
        result = executor(sample_dict_data)

        assert isinstance(result, list)
        assert len(result) == 2
        # First parallel branch
        assert batch_allclose(result[0]["image"], sample_dict_data["image"] * 2.0)
        assert batch_allclose(result[0]["label"], sample_dict_data["label"] * 2.0)
        # Second parallel branch
        assert batch_allclose(result[1]["image"], sample_dict_data["image"] * 3.0)
        assert batch_allclose(result[1]["label"], sample_dict_data["label"] * 3.0)

    def test_branch_execution_true_path(self, sample_dict_data):
        """Test branch execution taking true path."""
        executor = DAGExecutor(enforce_batch=False)
        # Condition checks the batch size of the image field (access via Batch API)
        condition = lambda x: x.data.get_value()["image"].shape[0] >= 4  # True for our sample data
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        executor.branch(condition, OperatorNode(true_path), OperatorNode(false_path))
        result = executor(sample_dict_data)

        assert batch_allclose(result["image"], sample_dict_data["image"] * 2.0)
        assert batch_allclose(result["label"], sample_dict_data["label"] * 2.0)
        assert true_path.call_count == 1
        assert false_path.call_count == 0

    def test_branch_execution_false_path(self, sample_dict_data):
        """Test branch execution taking false path."""
        executor = DAGExecutor(enforce_batch=False)
        # Condition checks the batch size of the image field (access via Batch API)
        condition = lambda x: x.data.get_value()["image"].shape[0] > 10  # False for our sample data
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        executor.branch(condition, true_path, false_path)
        result = executor(sample_dict_data)

        assert batch_allclose(result["image"], sample_dict_data["image"] * 0.5)
        assert batch_allclose(result["label"], sample_dict_data["label"] * 0.5)
        assert true_path.call_count == 0
        assert false_path.call_count == 1

    def test_merge_execution(self, sample_dict_data):
        """Test merge execution."""
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=1.0, name="t1")
        t2 = MockOperator(multiplier=2.0, name="t2")

        executor.parallel([t1, t2]).merge("sum")
        result = executor(sample_dict_data)

        # After merge with "sum", the results from parallel branches are summed
        expected_image = sample_dict_data["image"] * 1.0 + sample_dict_data["image"] * 2.0
        expected_label = sample_dict_data["label"] * 1.0 + sample_dict_data["label"] * 2.0
        assert batch_allclose(result["image"], expected_image)
        assert batch_allclose(result["label"], expected_label)

    def test_iteration_count_tracking(self, sample_dict_data):
        """Test iteration count tracking."""
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator()))

        assert executor._iteration_count == 0

        executor(sample_dict_data)
        assert executor._iteration_count == 1

        executor(sample_dict_data)
        assert executor._iteration_count == 2

    def test_empty_pipeline_execution(self, sample_data):
        """Test execution of empty pipeline (Identity)."""
        executor = DAGExecutor(enforce_batch=False)
        result = executor(sample_data)

        # Compare Batch data - empty pipeline returns input unchanged
        # Compare Batch data - empty pipeline returns input unchanged
        assert batch_allclose(
            result.data.get_value()["_array"], sample_data.data.get_value()["_array"]
        )


class TestDAGExecutorRNGHandling:
    """Test RNG handling and key distribution."""

    def test_rng_key_generation_with_rngs(self, sample_data):
        """Test RNG key generation when RNGs are provided."""
        rngs = nnx.Rngs(42)
        executor = DAGExecutor(rngs=rngs, enforce_batch=False)
        transform = RandomOperator(noise_scale=0.1, rngs=rngs)
        executor.add(transform)

        # RNG keys are always provided when rngs is available
        result1 = executor(sample_data)
        result2 = executor(sample_data)

        # Results should be different due to randomness
        assert not batch_allclose(result1, result2)

    def test_rng_key_with_default_rngs(self, sample_data):
        """Test RNG key generation with stochastic operators.

        When stochastic operators are added, the executor lazily creates RNGs
        and provides keys during iteration. Operators manage their own RNG state.
        """
        # Create shared RNGs for executor and operator
        rngs = nnx.Rngs(0)
        executor = DAGExecutor(rngs=rngs, enforce_batch=False)
        # Transform uses the same rngs as executor for coordinated randomness
        transform = RandomOperator(noise_scale=0.1, rngs=rngs)
        executor.add(transform)

        # RNGs are provided to stochastic ops, so results should be different
        result1 = executor(sample_data)
        result2 = executor(sample_data)

        # Results should be different due to randomness from RNGs
        assert not batch_allclose(result1, result2)

    def test_rng_key_distribution_parallel(self, sample_data):
        """Test RNG key distribution in parallel execution."""
        rngs = nnx.Rngs(42)
        executor = DAGExecutor(rngs=rngs, enforce_batch=False)
        t1 = RandomOperator(noise_scale=0.1, name="t1", rngs=rngs)
        t2 = RandomOperator(noise_scale=0.1, name="t2", rngs=rngs)

        executor.parallel([OperatorNode(t1), OperatorNode(t2)])

        # RNG keys are always provided when rngs is available
        result = executor(sample_data)

        # Should get different results from each parallel branch
        assert isinstance(result, list)
        assert len(result) == 2
        assert not batch_allclose(result[0], result[1])

    def test_no_rngs_provided(self, sample_data):
        """Test execution when no RNGs are provided."""
        executor = DAGExecutor(rngs=None, enforce_batch=False)
        transform = RandomOperator(noise_scale=0.1)
        executor.add(OperatorNode(transform))

        # Even with rngs=None, default RNGs are created, so transform gets key
        result = executor(sample_data)

        # Should work without error, but transform gets RNG key from default RNGs
        assert not batch_allclose(result, sample_data)  # Noise is added due to default RNGs


class TestDAGExecutorLazyRNG:
    """Test lazy RNG optimization - RNG only created when stochastic ops exist."""

    def test_lazy_rng_no_stochastic_ops(self):
        """Verify RNG is None for fully deterministic pipelines."""
        executor = DAGExecutor(enforce_batch=False)

        # No stochastic operators added - RNG should be None
        assert executor.rngs is None

        # Add deterministic operator
        transform = MockOperator(multiplier=2.0)
        executor.add(OperatorNode(transform))

        # Still no stochastic ops - RNG should still be None
        assert executor.rngs is None

    def test_lazy_rng_with_stochastic_ops(self):
        """Verify RNG is created lazily when stochastic ops detected."""
        executor = DAGExecutor(enforce_batch=False)

        # Initially no RNG
        assert executor.rngs is None

        # Add stochastic operator
        stochastic_op = RandomOperator(noise_scale=0.1)
        executor.add(OperatorNode(stochastic_op))

        # Now RNG should be created lazily
        assert executor.rngs is not None
        assert isinstance(executor.rngs, nnx.Rngs)

    def test_lazy_rng_explicit_rngs_always_used(self):
        """Verify user-provided rngs are always returned."""
        explicit_rngs = nnx.Rngs(42)
        executor = DAGExecutor(rngs=explicit_rngs, enforce_batch=False)

        # User provided explicit RNGs - should be returned even without stochastic ops
        assert executor.rngs is explicit_rngs

        # Still the same after adding deterministic op
        executor.add(OperatorNode(MockOperator(multiplier=2.0)))
        assert executor.rngs is explicit_rngs

    def test_lazy_rng_caching_deterministic_pipeline(self, sample_data):
        """Verify deterministic pipelines enable caching."""
        executor = DAGExecutor(enable_caching=True, enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        executor.add(OperatorNode(transform))

        # Deterministic pipeline - RNG is None
        assert executor.rngs is None

        # Execute twice
        result1 = executor(sample_data)
        result2 = executor(sample_data)

        # Caching should work - same result, operator called once
        assert transform.call_count == 1  # Cached!
        assert batch_allclose(result1, result2)

    def test_lazy_rng_stochastic_detection_shuffle_node(self):
        """Verify ShuffleNode is detected as stochastic."""
        from datarax.dag.nodes import ShuffleNode

        executor = DAGExecutor(enforce_batch=False)
        assert executor.rngs is None

        # Add ShuffleNode - should trigger RNG creation
        shuffle_node = ShuffleNode(buffer_size=100)
        executor.add(shuffle_node)

        # ShuffleNode is stochastic - RNG should now exist
        assert executor.rngs is not None

    def test_lazy_rng_stochastic_detection_nested_sequential(self):
        """Verify stochastic ops in nested Sequential are detected."""
        from datarax.dag.nodes import Sequential

        executor = DAGExecutor(enforce_batch=False)
        assert executor.rngs is None

        # Create nested sequential with stochastic op
        inner_seq = Sequential(
            [
                OperatorNode(MockOperator(multiplier=1.0)),
                OperatorNode(RandomOperator(noise_scale=0.1)),
            ]
        )
        executor.graph = inner_seq

        # Reset detection cache to force redetection
        executor._needs_rng = None

        # Should detect stochastic op in nested structure
        assert executor.rngs is not None

    def test_lazy_rng_performance_benefit(self, sample_data):
        """Verify lazy RNG actually improves performance for deterministic pipelines.

        This test ensures the optimization is working by checking that
        deterministic pipelines don't generate RNG keys during iteration.
        """
        import time

        # Create deterministic pipeline
        executor = DAGExecutor(enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        executor.add(OperatorNode(transform))

        # Verify no RNG overhead
        assert executor.rngs is None

        # Time multiple executions - should be fast without RNG generation
        start = time.perf_counter()
        for _ in range(100):
            executor(sample_data)
        elapsed = time.perf_counter() - start

        # Basic sanity check - should complete reasonably fast
        # (The actual performance gain was 4x+ in benchmarks)
        assert elapsed < 5.0, f"Deterministic pipeline too slow: {elapsed:.2f}s"


class TestDAGExecutorCaching:
    """Test caching functionality."""

    def test_caching_enabled_by_default(self):
        """Test that caching is enabled by default."""
        executor = DAGExecutor(enforce_batch=False)
        assert executor.enable_caching is True
        assert executor._cache is not None

    def test_caching_disabled(self):
        """Test caching disabled."""
        executor = DAGExecutor(enable_caching=False, enforce_batch=False)
        assert executor.enable_caching is False
        assert executor._cache is None

    def test_cache_with_deterministic_pipeline(self, sample_data):
        """Test caching works for deterministic pipelines (no stochastic ops).

        With lazy RNG optimization, deterministic pipelines (no stochastic operators)
        don't create RNGs at all. This enables caching for deterministic transforms.
        """
        executor = DAGExecutor(enable_caching=True, enforce_batch=False)
        # MockOperator is deterministic - no stochastic config
        transform = MockOperator(multiplier=2.0)
        executor.add(OperatorNode(transform))

        # Deterministic pipeline = no RNG = caching enabled
        # First call
        result1 = executor(sample_data)
        assert transform.call_count == 1

        # Second call SHOULD use cache since pipeline is deterministic
        result2 = executor(sample_data)
        assert transform.call_count == 1  # Not incremented - cached!
        assert batch_allclose(result1, result2)

    def test_no_cache_with_explicit_rngs(self, sample_data):
        """Test no caching in training mode (with RNG key)."""
        rngs = nnx.Rngs(42)
        executor = DAGExecutor(rngs=rngs, enable_caching=True, enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        executor.add(transform)

        # With RNGs, each call gets different key, so no caching
        # Both calls should execute transform
        executor(sample_data, key=None)
        assert transform.call_count == 1

        executor(sample_data)
        assert transform.call_count == 2  # Incremented

    def test_clear_cache(self, sample_data):
        """Test cache clearing."""
        # Create executor without rngs - lazy RNG means None for deterministic ops
        executor = DAGExecutor(enable_caching=True, enforce_batch=False, rngs=None)
        # MockOperator is deterministic, so rngs will be None (lazy optimization)
        transform = MockOperator(multiplier=2.0)
        executor.add(transform)

        # Populate cache - no key will be generated since rngs is None
        executor(sample_data)
        assert len(executor._cache) > 0

        # Clear cache
        executor.clear_cache()
        assert len(executor._cache) == 0

    def test_cache_with_cached_node(self, sample_data):
        """Test interaction with Cache node."""
        executor = DAGExecutor(enable_caching=True, enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        executor.add(OperatorNode(transform)).cache(cache_size=10)

        # First call
        result1 = executor(sample_data)

        # Clear executor cache but not node cache
        executor.clear_cache()

        # Second call should still benefit from node-level caching
        result2 = executor(sample_data)
        assert batch_allclose(result1, result2)


class TestDAGExecutorJITCompilation:
    """Test JIT compilation functionality."""

    def test_jit_compilation_disabled_by_default(self):
        """Test JIT compilation is disabled by default."""
        executor = DAGExecutor(enforce_batch=False)
        assert executor.jit_compile is False
        assert executor._jit_execute is None

    def test_jit_compilation_enabled(self):
        """Test JIT compilation enabled."""
        executor = DAGExecutor(jit_compile=True, enforce_batch=False)
        assert executor.jit_compile is True
        assert executor._jit_execute is not None

    def test_jit_execution(self, sample_data):
        """Test JIT-compiled execution."""
        executor = DAGExecutor(jit_compile=True, enforce_batch=False)
        transform = MockOperator(multiplier=3.0)
        executor.add(OperatorNode(transform))

        # Need to recompile after adding transform
        executor._compile()

        result = executor(sample_data)
        expected = batch_mul(sample_data, 3.0)
        assert batch_allclose(result, expected)

    def test_jit_invalidation_on_add(self):
        """Test JIT compilation invalidation when adding transforms."""
        executor = DAGExecutor(jit_compile=True, enforce_batch=False)

        executor.add(OperatorNode(MockOperator()))

        assert executor._jit_execute is None

    @pytest.mark.skip(reason="JIT compilation with complex graphs needs more testing")
    def test_jit_with_complex_graph(self, sample_data):
        """Test JIT compilation with complex graph."""
        executor = DAGExecutor(jit_compile=True, enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        executor.parallel([OperatorNode(t1), OperatorNode(t2)]).merge("sum")
        executor._compile()

        result = executor(sample_data)
        expected = batch_add(batch_mul(sample_data, 2.0), batch_mul(sample_data, 3.0))
        assert batch_allclose(result, expected)


class TestDAGExecutorStateManagement:
    """Test state management and checkpointing."""

    def test_get_state_basic(self):
        """Test basic state retrieval."""
        executor = DAGExecutor(enforce_batch=False)
        transform = MockOperator()
        executor.add(OperatorNode(transform))

        state = executor.get_state()

        assert isinstance(state, dict)
        assert "nnx_state" in state
        assert "graph_state" in state
        assert "iteration_count" in state
        assert "cache" in state

    def test_get_state_with_execution(self, sample_data):
        """Test state after execution."""
        executor = DAGExecutor(enforce_batch=False)
        transform = MockOperator()
        executor.add(OperatorNode(transform))

        # Execute a few times
        executor(sample_data)
        executor(sample_data)

        state = executor.get_state()

        assert state["iteration_count"] == 2

    def test_set_state_basic(self, sample_data):
        """Test basic state restoration.

        Note: Caching is disabled to avoid NNX state serialization issues
        with JAX arrays in cache. This test focuses on iteration/epoch counts.
        """
        # Create and execute original executor
        # Disable caching to avoid NNX state serialization issues
        executor1 = DAGExecutor(enable_caching=False, enforce_batch=False)
        transform1 = MockOperator()
        executor1.add(OperatorNode(transform1))

        executor1(sample_data)
        executor1(sample_data)
        state = executor1.get_state()

        # Create new executor and restore state
        executor2 = DAGExecutor(enable_caching=False, enforce_batch=False)
        transform2 = MockOperator()
        executor2.add(OperatorNode(transform2))

        executor2.set_state(state)

        assert executor2._iteration_count == 2

    @pytest.mark.skip(reason="Cache serialization with JAX arrays needs separate fix")
    def test_state_with_cache(self, sample_data):
        """Test state management with cache."""
        executor = DAGExecutor(enable_caching=True, enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        executor.add(OperatorNode(transform))

        executor(sample_data)  # Populate cache

        state = executor.get_state()
        assert state["cache"] is not None

        # Restore to new executor
        new_executor = DAGExecutor(enable_caching=True, enforce_batch=False)
        new_executor.add(OperatorNode(MockOperator(multiplier=2.0)))
        new_executor.set_state(state)

        assert new_executor._cache is not None

    def test_state_without_cache(self):
        """Test state management without cache."""
        executor = DAGExecutor(enable_caching=False, enforce_batch=False)

        state = executor.get_state()
        assert state["cache"] is None

        new_executor = DAGExecutor(enable_caching=False, enforce_batch=False)
        new_executor.set_state(state)

        assert new_executor._cache is None


class TestOperatorNode:
    """Test OperatorNode wrapper functionality."""

    def test_transform_node_creation(self):
        """Test OperatorNode creation."""
        transform = MockOperator(multiplier=2.0, name="test")
        node = OperatorNode(transform, name="test")

        assert node.operator is transform
        assert node.name == "test"

    def test_transform_node_execution(self, sample_data):
        """Test OperatorNode execution."""
        transform = MockOperator(multiplier=3.0)
        node = OperatorNode(transform, name="test")

        result = node(sample_data, key=None)
        expected = batch_mul(sample_data, 3.0)

        assert batch_allclose(result, expected)
        assert transform.call_count == 1

    def test_transform_node_with_key(self, sample_data):
        """Test OperatorNode execution with RNG key."""
        transform = RandomOperator(noise_scale=0.1)
        node = OperatorNode(transform, name="test")

        key = jax.random.key(42)
        result = node(sample_data, key=key)

        # Should not be identical to input due to noise
        assert not batch_allclose(result, sample_data)

    def test_transform_node_state_management(self, sample_data):
        """Test OperatorNode state management."""
        transform = MockOperator()
        node = OperatorNode(transform, name="test")

        # Execute and check state
        node(sample_data)
        state = node.get_state()

        assert isinstance(state, dict)

        # Create new node and restore state
        new_transform = MockOperator()
        new_node = OperatorNode(new_transform, name="test")
        new_node.set_state(state)

        # States should match
        assert new_transform.call_count == transform.call_count.get_value()

    def test_transform_node_repr(self):
        """Test OperatorNode string representation."""
        transform = MockOperator(name="test_transform")
        node = OperatorNode(transform, name="test")

        repr_str = repr(node)
        # OperatorNode repr includes node name (from Node.__repr__)
        assert "OperatorNode" in repr_str and "test" in repr_str


class TestDAGExecutorOperatorComposition:
    """Test operator-based composition (>>, |)."""

    def test_rshift_operator_with_transforms(self, sample_data):
        """Test >> operator with transforms."""
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Use >> operator
        composed = OperatorNode(t1) >> OperatorNode(t2)

        assert isinstance(composed, Sequential)
        assert len(composed.nodes) == 2

        # Test execution
        result = composed(sample_data, key=None)
        expected = batch_mul(batch_mul(sample_data, 2.0), 3.0)
        assert batch_allclose(result, expected)

    def test_or_operator_with_transforms(self, sample_data):
        """Test | operator with transforms."""
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Use | operator
        composed = OperatorNode(t1) | OperatorNode(t2)

        assert isinstance(composed, Parallel)
        assert len(composed.nodes) == 2

        # Test execution
        result = composed(sample_data, key=None)
        assert isinstance(result, list)
        assert len(result) == 2
        assert batch_allclose(result[0], batch_mul(sample_data, 2.0))
        assert batch_allclose(result[1], batch_mul(sample_data, 3.0))

    def test_mixed_operator_composition(self, sample_data):
        """Test mixed operator composition."""
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")
        t3 = MockOperator(multiplier=4.0, name="t3")

        # Create: t1 >> (t2 | t3) using OperatorNode wrappers
        parallel_part = OperatorNode(t2) | OperatorNode(t3)
        composed = OperatorNode(t1) >> parallel_part

        assert isinstance(composed, Sequential)
        assert len(composed.nodes) == 2
        assert isinstance(composed.nodes[1], Parallel)

        # Test execution
        result = composed(sample_data, key=None)
        assert isinstance(result, list)
        assert len(result) == 2

        # First transform applied, then parallel
        expected_base = batch_mul(sample_data, 2.0)
        assert batch_allclose(result[0], batch_mul(expected_base, 3.0))
        assert batch_allclose(result[1], batch_mul(expected_base, 4.0))


class TestDAGExecutorComplexScenarios:
    """Test complex DAG scenarios and edge cases."""

    def test_deep_sequential_pipeline(self, sample_data):
        """Test deep sequential pipeline."""
        executor = DAGExecutor(enforce_batch=False)

        # Add many sequential transforms
        for i in range(10):
            executor.add(OperatorNode(MockOperator(multiplier=1.1, name=f"t{i}")))

        result = executor(sample_data)
        expected = batch_mul(sample_data, 1.1**10)

        assert batch_allclose(result, expected, rtol=1e-5)

    def test_wide_parallel_pipeline(self, sample_data):
        """Test wide parallel pipeline."""
        executor = DAGExecutor(enforce_batch=False)

        # Create many parallel transforms
        transforms = [MockOperator(multiplier=i + 1, name=f"t{i}") for i in range(10)]
        executor.parallel([OperatorNode(t) for t in transforms])

        result = executor(sample_data)

        assert isinstance(result, list)
        assert len(result) == 10

        for i, res in enumerate(result):
            expected = batch_mul(sample_data, i + 1)
            assert batch_allclose(res, expected)

    def test_nested_dag_structure(self, sample_data):
        """Test nested DAG structure."""
        executor = DAGExecutor(enforce_batch=False)

        # Create nested structure: normalize -> parallel(aug1 -> crop1, aug2 -> crop2) -> merge
        normalize = MockOperator(multiplier=0.5, name="normalize")
        aug1 = MockOperator(multiplier=1.1, name="aug1")
        crop1 = MockOperator(multiplier=0.9, name="crop1")
        aug2 = MockOperator(multiplier=1.2, name="aug2")
        crop2 = MockOperator(multiplier=0.8, name="crop2")

        # Build nested structure
        branch1 = OperatorNode(aug1) >> OperatorNode(crop1)
        branch2 = OperatorNode(aug2) >> OperatorNode(crop2)

        executor.add(OperatorNode(normalize)).parallel([branch1, branch2]).merge("mean")

        result = executor(sample_data)

        # Calculate expected result
        normalized = batch_mul(sample_data, 0.5)
        branch1_result = batch_mul(batch_mul(normalized, 1.1), 0.9)
        branch2_result = batch_mul(batch_mul(normalized, 1.2), 0.8)
        expected = batch_mul(batch_add(branch1_result, branch2_result), 0.5)  # mean = (a+b)/2

        assert batch_allclose(result, expected)

    def test_conditional_complex_paths(self, sample_data):
        """Test complex conditional paths."""
        executor = DAGExecutor(enforce_batch=False)

        # Create condition based on Batch data properties
        def condition(batch):
            # Extract data from Batch and compute mean of first leaf
            data = batch.data.get_value()
            first_leaf = jax.tree.leaves(data)[0]
            return jnp.mean(first_leaf) > 0.5

        # Complex true path: parallel processing
        true_t1 = MockOperator(multiplier=2.0, name="true_t1")
        true_t2 = MockOperator(multiplier=3.0, name="true_t2")
        true_path = (OperatorNode(true_t1) | OperatorNode(true_t2)) >> Merge("sum")

        # Simple false path
        false_path = MockOperator(multiplier=0.1, name="false")

        executor.branch(condition, true_path, OperatorNode(false_path))

        result = executor(sample_data)

        # Our sample data has mean=1.0, so should take true path
        expected = batch_add(
            batch_mul(sample_data, 2.0), batch_mul(sample_data, 3.0)
        )  # sum of parallel
        assert batch_allclose(result, expected)

    def test_dictionary_data_processing(self, sample_dict_data):
        """Test processing dictionary data."""
        executor = DAGExecutor(enforce_batch=False)

        # Transform that works with dict data
        def dict_transform(data, *, key=None):
            return {
                "image": data["image"] * 2.0,
                "label": data["label"] + 1,
                "metadata": data["metadata"] * 0.5,
            }

        class DictTransform(OperatorModule):
            def __init__(self):
                config = OperatorConfig(stochastic=False)
                super().__init__(config, name="dict_transform")

            def apply(self, data, state, metadata, random_params=None, stats=None):
                result = dict_transform(data, key=None)
                return result, state, metadata

        executor.add(OperatorNode(DictTransform()))
        result = executor(sample_dict_data)

        # Result is a Batch with transformed data
        assert isinstance(result, Batch)
        result_data = result.data.get_value()
        input_data = sample_dict_data.data.get_value()
        assert jnp.allclose(result_data["image"], input_data["image"] * 2.0)
        assert jnp.allclose(result_data["label"], input_data["label"] + 1)
        assert jnp.allclose(result_data["metadata"], input_data["metadata"] * 0.5)

    def test_split_fields_integration(self, sample_dict_data):
        """Test integration with SplitFields node."""
        executor = DAGExecutor(enforce_batch=False)

        # Create field-specific transforms wrapped in OperatorNode
        image_transform = MockOperator(multiplier=2.0, name="image_proc")
        label_transform = MockOperator(multiplier=1.0, name="label_proc")  # Identity-like

        split_node = SplitFields(
            {"image": OperatorNode(image_transform), "label": OperatorNode(label_transform)}
        )

        executor.add(split_node)
        result = executor(sample_dict_data)

        # Result is a Batch with transformed fields
        assert isinstance(result, Batch)
        result_data = result.data.get_value()
        input_data = sample_dict_data.data.get_value()
        assert jnp.allclose(result_data["image"], input_data["image"] * 2.0)
        assert jnp.allclose(result_data["label"], input_data["label"] * 1.0)
        # metadata should pass through unchanged
        assert jnp.allclose(result_data["metadata"], input_data["metadata"])


class TestDAGExecutorErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_merge_strategy(self, sample_data):
        """Test invalid merge strategy handling."""
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0)
        t2 = MockOperator(multiplier=3.0)

        executor.parallel([OperatorNode(t1), OperatorNode(t2)])

        # Create merge with invalid strategy
        merge_node = Merge(strategy="invalid_strategy")
        executor.add(merge_node)

        with pytest.raises(ValueError, match="Unknown merge strategy"):
            executor(sample_data)

    def test_empty_parallel_execution(self, sample_data):
        """Test parallel execution with empty node list."""
        executor = DAGExecutor(enforce_batch=False)
        executor.parallel([])  # Empty list

        result = executor(sample_data)
        # Should still work, returning empty list from parallel
        assert isinstance(result, list)
        assert len(result) == 0

    def test_none_data_handling(self):
        """Test handling of zero-filled data."""
        executor = DAGExecutor(enforce_batch=False)

        # Operator that handles zero-filled data
        class ZeroHandlingOperator(OperatorModule):
            def __init__(self):
                config = OperatorConfig(stochastic=False)
                super().__init__(config, name="zero_handling")

            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Return zeros of same shape
                result = jax.tree.map(jnp.zeros_like, data)
                return result, state, metadata

        executor.add(OperatorNode(ZeroHandlingOperator()))

        # Pass batch data as Batch object
        batch = Batch.from_parts(
            data={"arr": jnp.zeros((1, 2, 2))},
            states={"arr": jnp.zeros((1,))},
            validate=False,
        )
        result = executor(batch)

        expected = Batch.from_parts(
            data={"arr": jnp.zeros((1, 2, 2))},
            states={"arr": jnp.zeros((1,))},
            validate=False,
        )
        assert batch_allclose(result, expected)

    def test_mismatched_data_shapes_in_merge(self, sample_data):
        """Test merge with mismatched data shapes."""
        executor = DAGExecutor(enforce_batch=False)

        # Operators that produce different shapes
        class ReshapeOperator1(OperatorModule):
            def __init__(self):
                config = OperatorConfig(stochastic=False)
                super().__init__(config, name="reshape1")

            def apply(self, data, state, metadata, random_params=None, stats=None):
                result = data.reshape(-1)  # Flatten
                return result, state, metadata

        class ReshapeOperator2(OperatorModule):
            def __init__(self):
                config = OperatorConfig(stochastic=False)
                super().__init__(config, name="reshape2")

            def apply(self, data, state, metadata, random_params=None, stats=None):
                return data, state, metadata  # Keep original shape

        executor.parallel([OperatorNode(ReshapeOperator1()), OperatorNode(ReshapeOperator2())])
        executor.merge("concat")  # This should fail due to shape mismatch

        with pytest.raises(Exception):  # JAX will raise an error for incompatible shapes
            executor(sample_data)


class TestDAGExecutorVisualization:
    """Test DAG visualization functionality."""

    def test_visualize_empty_dag(self):
        """Test visualization of empty DAG."""
        executor = DAGExecutor(enforce_batch=False)

        viz = executor.visualize()

        assert isinstance(viz, str)
        assert "DAGExecutor" in viz
        assert "Identity" in viz

    def test_visualize_simple_dag(self):
        """Test visualization of simple DAG."""
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator(name="test_transform")))

        viz = executor.visualize()

        assert isinstance(viz, str)
        assert "DAGExecutor" in viz
        assert "test_transform" in viz or "OperatorNode" in viz

    def test_visualize_complex_dag(self):
        """Test visualization of complex DAG."""
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(name="transform1")
        t2 = MockOperator(name="transform2")
        t3 = MockOperator(name="transform3")

        executor.add(OperatorNode(t1)).parallel([OperatorNode(t2), OperatorNode(t3)]).merge(
            "concat"
        )

        viz = executor.visualize()

        assert isinstance(viz, str)
        assert "DAGExecutor" in viz
        assert "Sequential" in viz or "Parallel" in viz

    def test_repr_string(self):
        """Test string representation."""
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator()))

        # Execute once to increment iteration count with Batch
        batch = Batch.from_parts(
            data={"arr": jnp.ones((2, 2))},
            states={"arr": jnp.zeros((2,))},
            validate=False,
        )
        executor(batch)

        repr_str = repr(executor)

        assert "DAGExecutor" in repr_str
        assert "iterations=1" in repr_str
        assert "cached=True" in repr_str
        assert "jit=False" in repr_str


class TestConvenienceFunctions:
    """Test convenience functions for pipeline construction."""

    def test_pipeline_function(self, sample_data):
        """Test pipeline() convenience function with DataLoader first."""
        from datarax.dag.nodes import DataLoader
        from datarax.core.data_source import DataSourceModule

        # Create a simple data source
        class SimpleSource(DataSourceModule):
            # REQUIRED: Annotate data with nnx.data()
            data: list = nnx.data()

            def __init__(self):
                config = StructuralConfig(stochastic=False)
                super().__init__(config, name="simple_source")
                self.data = [sample_data]
                self.index = 0

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                result = self.data[self.index]
                self.index += 1
                return result

        # Create a DataLoader as required by new API
        source = SimpleSource()
        loader = DataLoader(source, batch_size=1)

        # Create transforms
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Pipeline must start with DataLoader
        p = pipeline(loader, t1, t2)

        assert isinstance(p, DAGExecutor)
        # Note: Result will be batched due to DataLoader

    def test_parallel_function(self, sample_data):
        """Test creating and using Parallel node directly."""
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Create Parallel node directly with OperatorNodes
        p = Parallel([OperatorNode(t1, name="t1"), OperatorNode(t2, name="t2")])

        assert isinstance(p, Parallel)
        result = p(sample_data, key=None)

        assert isinstance(result, list)
        assert len(result) == 2
        assert batch_allclose(result[0], batch_mul(sample_data, 2.0))
        assert batch_allclose(result[1], batch_mul(sample_data, 3.0))

    def test_branch_function(self, sample_data):
        """Test creating and using Branch node directly."""
        # Condition must access Batch data: x.data.get_value()["_array"].shape[0]
        condition = lambda x: x.data.get_value()["_array"].shape[0] > 2
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        # Create Branch node directly with OperatorNodes
        b = Branch(condition, OperatorNode(true_path), OperatorNode(false_path))

        assert isinstance(b, Branch)
        result = b(sample_data, key=None)

        # Sample data has shape[0] = 4 > 2, so should take true path
        expected = batch_mul(sample_data, 2.0)
        assert batch_allclose(result, expected)

    def test_convenience_functions_with_nodes(self, sample_data):
        """Test creating Parallel with Node instances."""

        # Create Parallel node directly with Identity nodes
        p = Parallel([Identity(name="identity1"), Identity(name="identity2")])

        assert isinstance(p, Parallel)
        result = p(sample_data, key=None)

        assert isinstance(result, list)
        assert len(result) == 2
        assert batch_allclose(result[0], sample_data)
        assert batch_allclose(result[1], sample_data)


class TestDAGExecutorIntegration:
    """Integration tests with real Datarax components."""

    def test_integration_with_nnx_modules(self, sample_data):
        """Test integration with real NNX modules."""
        # Get the array shape from the Batch for Linear layer initialization
        input_shape = sample_data.data.get_value()["_array"].shape[-1]

        # Create a simple NNX module
        class SimpleNNXModule(OperatorModule):
            def __init__(self):
                config = OperatorConfig(stochastic=False)
                super().__init__(config, name="simple_nnx")
                self.linear = nnx.Linear(
                    in_features=input_shape,
                    out_features=input_shape,
                    rngs=nnx.Rngs(42),
                )

            def apply(self, data, state, metadata, random_params=None, stats=None):
                # data is a dict with "_array" key
                arr = data["_array"]
                # Apply linear transformation to last dimension
                original_shape = arr.shape
                flat_data = arr.reshape(-1, arr.shape[-1])
                transformed = self.linear(flat_data)
                result = {"_array": transformed.reshape(original_shape)}
                return result, state, metadata

        executor = DAGExecutor(enforce_batch=False)
        nnx_module = SimpleNNXModule()
        executor.add(OperatorNode(nnx_module))

        result = executor(sample_data)

        # Should have same shape but different values
        assert (
            result.data.get_value()["_array"].shape == sample_data.data.get_value()["_array"].shape
        )
        assert not batch_allclose(result, sample_data)

    def test_mixed_transform_types(self, sample_data):
        """Test mixing different types of transforms."""
        # Mix DataraxModule and Node types
        mock_operator = MockOperator(multiplier=2.0)
        identity_node = Identity()

        executor = DAGExecutor(enforce_batch=False)
        executor.add(mock_operator).add(identity_node)

        result = executor(sample_data)
        expected = batch_mul(sample_data, 2.0)  # Identity doesn't change the result

        assert batch_allclose(result, expected)

    def test_stateful_execution_across_calls(self, sample_data):
        """Test iteration-aware operator produces different output per call.

        Note: Caching is disabled because IterationAwareOperator uses internal
        state to modify output. With lazy RNG optimization, deterministic pipelines
        enable caching by default, so stateful operators need caching disabled.
        """
        # Disable caching for stateful operators that depend on internal state
        executor = DAGExecutor(enable_caching=False, enforce_batch=False)
        operator = IterationAwareOperator()
        executor.add(operator)

        # Multiple executions should show iteration-dependent changes
        result1 = executor(sample_data)
        result2 = executor(sample_data)
        result3 = executor(sample_data)

        # Each result should be different (adds iteration count to data)
        assert not batch_allclose(result1, result2)
        assert not batch_allclose(result2, result3)

        # Iteration count should have incremented
        assert operator.state.get_value() == 3
        assert operator.call_count == 3


# Performance and stress tests
class TestDAGExecutorPerformance:
    """Performance and stress tests."""

    @pytest.mark.slow
    def test_large_data_processing(self):
        """Test processing of large data."""
        # Create large Batch with image data
        large_data = Batch.from_parts(
            data={"image": jnp.ones((100, 224, 224, 3))},
            states={"image": jnp.zeros((100,))},
            validate=False,
        )

        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator(multiplier=1.1)))

        result = executor(large_data)

        expected = batch_mul(large_data, 1.1)
        assert batch_allclose(result, expected)

    @pytest.mark.slow
    def test_many_sequential_transforms(self):
        """Test many sequential transforms."""
        data = Batch.from_parts(
            data={"image": jnp.ones((10, 32, 32, 3))},
            states={"image": jnp.zeros((10,))},
            validate=False,
        )

        executor = DAGExecutor(enforce_batch=False)

        # Add 100 sequential transforms
        for i in range(100):
            executor.add(OperatorNode(MockOperator(multiplier=1.001, name=f"t{i}")))

        result = executor(data)
        expected = batch_mul(data, 1.001**100)

        assert batch_allclose(result, expected, rtol=1e-3)

    @pytest.mark.slow
    def test_wide_parallel_processing(self):
        """Test wide parallel processing."""
        data = Batch.from_parts(
            data={"image": jnp.ones((10, 16, 16, 3))},
            states={"image": jnp.zeros((10,))},
            validate=False,
        )

        executor = DAGExecutor(enforce_batch=False)

        # Create 50 parallel transforms
        transforms = [MockOperator(multiplier=1.0, name=f"t{i}") for i in range(50)]
        executor.parallel(transforms)

        result = executor(data)

        assert isinstance(result, list)
        assert len(result) == 50

        for res in result:
            assert batch_allclose(res, data)


class TestDAGExecutorBasics:
    """Test basic DAGExecutor functionality."""

    def test_default_initialization(self):
        """Test default DAGExecutor initialization."""
        executor = DAGExecutor()

        assert isinstance(executor.graph, Identity)
        # rngs is lazy - None for deterministic pipelines (no stochastic ops)
        assert executor.rngs is None  # No stochastic ops = no RNG needed
        assert executor.enable_caching is True
        assert executor.jit_compile is False
        assert executor._iteration_count == 0
        assert executor._cache == {}
        assert executor._jit_execute is None

    def test_initialization_with_transform(self):
        """Test initialization with initial transform."""
        # Create a OperatorNode instead of a raw MockOperator
        transform = MockOperator(multiplier=2.0)
        transform_node = OperatorNode(transform)
        executor = DAGExecutor(transform_node)

        assert isinstance(executor.graph, OperatorNode)
        assert executor.graph.operator is transform

    def test_initialization_with_node(self):
        """Test initialization with Node instance."""
        identity = Identity()
        executor = DAGExecutor(identity)

        assert executor.graph is identity

    def test_initialization_options(self):
        """Test initialization with various options."""
        rngs = nnx.Rngs(42)
        executor = DAGExecutor(rngs=rngs, enable_caching=False, jit_compile=False)

        assert executor.rngs is rngs
        assert executor.enable_caching is False
        assert executor._cache is None

    def test_add_transform_to_empty(self):
        """Test adding transform to empty executor."""
        executor = DAGExecutor(enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        result = executor.add(transform)

        assert result is executor  # Returns self for chaining
        assert isinstance(executor.graph, OperatorNode)
        assert executor.graph.operator is transform

    def test_add_sequential_transforms(self):
        """Test adding multiple transforms sequentially."""
        executor = DAGExecutor()
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Add batch node first to satisfy batch-first enforcement
        executor.batch(32).add(OperatorNode(t1)).add(OperatorNode(t2))

        assert isinstance(executor.graph, Sequential)
        # The graph now contains BatchNode + 2 transforms = 3 nodes
        assert len(executor.graph.nodes) == 3


class TestDAGExecutorExecutionExtended:
    """Test DAG execution functionality - extended tests."""

    def test_basic_execution(self, sample_data):
        """Test basic pipeline execution."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        transform = MockOperator(multiplier=3.0)
        executor.add(transform)

        result = executor(sample_data)

        expected = batch_mul(sample_data, 3.0)
        assert batch_allclose(result, expected)
        assert transform.call_count == 1

    def test_sequential_execution(self, sample_data):
        """Test sequential pipeline execution."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        executor.add(OperatorNode(t1)).add(OperatorNode(t2))
        result = executor(sample_data)

        expected = batch_mul(batch_mul(sample_data, 2.0), 3.0)
        assert batch_allclose(result, expected)
        assert t1.call_count == 1
        assert t2.call_count == 1

    def test_parallel_execution(self, sample_data):
        """Test parallel pipeline execution."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        executor.parallel([t1, t2])
        result = executor(sample_data)

        assert isinstance(result, list)
        assert len(result) == 2
        assert batch_allclose(result[0], batch_mul(sample_data, 2.0))
        assert batch_allclose(result[1], batch_mul(sample_data, 3.0))

    def test_branch_execution_true_path(self, sample_data):
        """Test branch execution taking true path."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        # Condition must access Batch data: x.data.get_value()["_array"].shape[0]
        condition = lambda x: x.data.get_value()["_array"].shape[0] >= 4  # True for our sample data
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        executor.branch(condition, true_path, false_path)
        result = executor(sample_data)

        expected = batch_mul(sample_data, 2.0)
        assert batch_allclose(result, expected)
        assert true_path.call_count == 1
        assert false_path.call_count == 0

    def test_branch_execution_false_path(self, sample_data):
        """Test branch execution taking false path."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        # Condition must access Batch data: x.data.get_value()["_array"].shape[0]
        condition = (
            lambda x: x.data.get_value()["_array"].shape[0] > 10
        )  # False for our sample data
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        executor.branch(condition, true_path, false_path)
        result = executor(sample_data)

        expected = batch_mul(sample_data, 0.5)
        assert batch_allclose(result, expected)
        assert true_path.call_count == 0
        assert false_path.call_count == 1

    def test_merge_execution(self, sample_data):
        """Test merge execution."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        t1 = MockOperator(multiplier=1.0, name="t1")
        t2 = MockOperator(multiplier=2.0, name="t2")

        executor.parallel([t1, t2]).merge("sum")
        result = executor(sample_data)

        expected = batch_add(
            batch_mul(sample_data, 1.0), batch_mul(sample_data, 2.0)
        )  # sum of parallel results
        assert batch_allclose(result, expected)

    def test_iteration_count_tracking(self, sample_data):
        """Test iteration count tracking."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator()))

        assert executor._iteration_count == 0

        executor(sample_data)
        assert executor._iteration_count == 1

        executor(sample_data)
        assert executor._iteration_count == 2

    def test_empty_pipeline_execution(self, sample_data):
        """Test execution of empty pipeline (Identity)."""
        executor = DAGExecutor()
        result = executor(sample_data)

        assert batch_allclose(result, sample_data)


class TestDAGExecutorPipelineConstructionExtended:
    """Test pipeline construction methods - extended tests."""

    def test_parallel_construction(self):
        """Test parallel pipeline construction."""
        executor = DAGExecutor()
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")
        t3 = MockOperator(multiplier=4.0, name="t3")

        result = executor.parallel([t1, t2, t3])

        assert result is executor
        assert isinstance(executor.graph, Parallel)
        assert len(executor.graph.nodes) == 3

    def test_branch_construction(self):
        """Test branch construction."""
        executor = DAGExecutor()
        # Condition must access Batch data: x.data.get_value()["_array"].shape[0]
        condition = lambda x: x.data.get_value()["_array"].shape[0] > 2
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        result = executor.branch(condition, true_path, false_path)

        assert result is executor
        assert isinstance(executor.graph, Branch)
        assert executor.graph.condition is condition

    def test_merge_construction(self):
        """Test merge construction."""
        executor = DAGExecutor()

        result = executor.merge(strategy="sum", axis=0)

        assert result is executor
        assert isinstance(executor.graph, Merge)
        assert executor.graph.strategy == "sum"
        assert executor.graph.axis == 0

    def test_cache_construction(self):
        """Test cache construction."""
        # Disable batch enforcement for this test
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator()))

        result = executor.cache(cache_size=50)

        assert result is executor
        # The DAGExecutor uses CacheNode, not Cache
        from datarax.dag.nodes import CacheNode

        assert isinstance(executor.graph, CacheNode)
        assert executor.graph.cache_size == 50

    def test_complex_pipeline_construction(self):
        """Test complex pipeline construction."""
        # Disable batch enforcement for this test
        executor = DAGExecutor(enforce_batch=False)

        # Build: normalize -> parallel(aug1, aug2) -> merge -> crop
        normalize = MockOperator(multiplier=0.5, name="normalize")
        aug1 = MockOperator(multiplier=1.1, name="aug1")
        aug2 = MockOperator(multiplier=0.9, name="aug2")
        crop = MockOperator(multiplier=1.0, name="crop")

        executor.add(OperatorNode(normalize)).parallel(
            [OperatorNode(aug1), OperatorNode(aug2)]
        ).merge("mean").add(OperatorNode(crop))

        # Should be Sequential with 4 nodes
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 4


class TestDAGExecutorCachingExtended:
    """Test caching functionality - extended tests."""

    def test_caching_enabled_by_default(self):
        """Test that caching is enabled by default."""
        executor = DAGExecutor()
        assert executor.enable_caching is True
        assert executor._cache is not None

    def test_caching_disabled(self):
        """Test caching disabled."""
        executor = DAGExecutor(enable_caching=False)
        assert executor.enable_caching is False
        assert executor._cache is None

    def test_clear_cache(self, sample_data):
        """Test cache clearing."""
        # Disable batch enforcement for this test
        executor = DAGExecutor(enable_caching=True, enforce_batch=False)
        transform = MockOperator(multiplier=2.0)
        executor.add(transform)

        # Execute to potentially populate cache
        executor(sample_data)

        # Clear cache
        executor.clear_cache()
        assert len(executor._cache) == 0


class TestOperatorNodeExtended:
    """Test OperatorNode wrapper functionality - extended tests."""

    def test_transform_node_creation(self):
        """Test OperatorNode creation."""
        transform = MockOperator(multiplier=2.0, name="test")
        node = OperatorNode(transform)

        assert node.operator is transform
        assert node.name == "test"

    def test_transform_node_execution(self, sample_data):
        """Test OperatorNode execution."""
        transform = MockOperator(multiplier=3.0)
        node = OperatorNode(transform)

        result = node(sample_data, key=None)
        expected = batch_mul(sample_data, 3.0)

        assert batch_allclose(result, expected)
        assert transform.call_count == 1

    def test_transform_node_state_management(self, sample_data):
        """Test OperatorNode state management."""
        transform = MockOperator()
        node = OperatorNode(transform)

        # Execute and check state
        node(sample_data)
        state = node.get_state()

        assert isinstance(state, dict)

        # Create new node and restore state
        new_transform = MockOperator()
        new_node = OperatorNode(new_transform)
        new_node.set_state(state)

        # States should match
        assert new_transform.call_count == transform.call_count.get_value()


class TestDAGExecutorStateManagementExtended:
    """Test state management and checkpointing - extended tests."""

    def test_get_state_basic(self):
        """Test basic state retrieval."""
        executor = DAGExecutor()
        transform = MockOperator()
        executor.batch(32).add(transform)

        state = executor.get_state()

        assert isinstance(state, dict)
        assert "nnx_state" in state
        assert "graph_state" in state
        assert "iteration_count" in state
        assert "cache" in state

    def test_get_state_with_execution(self, sample_data):
        """Test state after execution."""
        executor = DAGExecutor()
        transform = MockOperator()
        executor.batch(32).add(transform)

        # Execute a few times
        executor(sample_data)
        executor(sample_data)

        state = executor.get_state()

        assert state["iteration_count"] == 2

    def test_set_state_basic(self, sample_data):
        """Test basic state restoration."""
        # Create and execute original executor
        # Disable batch enforcement for simpler state management
        executor1 = DAGExecutor(enable_caching=False, enforce_batch=False)

        # Don't add transforms - just test basic state
        executor1(sample_data)
        executor1(sample_data)

        # Get state
        state = executor1.get_state()
        assert state["iteration_count"] == 2
        assert state["epoch_count"] == 0

        # Create new executor with same configuration
        executor2 = DAGExecutor(enable_caching=False, enforce_batch=False)

        # Restore state
        executor2.set_state(state)

        # Check that basic state values were restored
        assert executor2._iteration_count == 2
        assert executor2._epoch_count == 0

        # Verify the restored executor can still execute
        result = executor2(sample_data)
        assert batch_allclose(result, sample_data)  # Identity transform
        assert executor2._iteration_count == 3  # Incremented


class TestConvenienceFunctionsExtended:
    """Test convenience functions for pipeline construction - extended tests."""

    def test_pipeline_function(self, sample_data):
        """Test pipeline() convenience function with DataLoader first."""
        # Create a DataLoader as required by new API
        source = MockDataSource(size=5)
        loader = DataLoader(source, batch_size=2)

        # Create transforms
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Pipeline must start with DataLoader
        p = pipeline(loader, t1, t2)

        assert isinstance(p, DAGExecutor)
        # Note: Result will be batched due to DataLoader

    def test_pipeline_function_validation(self):
        """Test pipeline() function validation."""
        t1 = MockOperator(multiplier=2.0, name="t1")

        # Should raise error when not starting with DataLoader
        with pytest.raises(ValueError, match="First node must be a DataLoader"):
            pipeline(t1)

        # Should raise error for empty pipeline
        with pytest.raises(ValueError, match="Pipeline must have at least one node"):
            pipeline()

    def test_parallel_function(self, sample_data):
        """Test parallel() convenience function."""
        t1 = MockOperator(multiplier=2.0, name="t1")
        t2 = MockOperator(multiplier=3.0, name="t2")

        # Wrap operators in OperatorNode for parallel composition
        p = parallel(OperatorNode(t1), OperatorNode(t2))

        assert isinstance(p, Parallel)
        result = p(sample_data, key=None)

        assert isinstance(result, list)
        assert len(result) == 2
        assert batch_allclose(result[0], batch_mul(sample_data, 2.0))
        assert batch_allclose(result[1], batch_mul(sample_data, 3.0))

    def test_branch_function(self, sample_data):
        """Test branch() convenience function."""
        # Condition must access Batch data: x.data.get_value()["_array"].shape[0]
        condition = lambda x: x.data.get_value()["_array"].shape[0] > 2
        true_path = MockOperator(multiplier=2.0, name="true")
        false_path = MockOperator(multiplier=0.5, name="false")

        # Wrap operators in OperatorNode for branch composition
        b = branch(condition, OperatorNode(true_path), OperatorNode(false_path))

        assert isinstance(b, Branch)
        result = b(sample_data, key=None)

        # Sample data has shape[0] = 4 > 2, so should take true path
        expected = batch_mul(sample_data, 2.0)
        assert batch_allclose(result, expected)


class TestDAGExecutorComplexScenariosExtended:
    """Test complex DAG scenarios - extended tests."""

    def test_nested_dag_structure(self, sample_data):
        """Test nested DAG structure."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)

        # Create nested structure: normalize -> parallel(aug1 -> crop1, aug2 -> crop2) -> merge
        normalize = MockOperator(multiplier=0.5, name="normalize")
        aug1 = MockOperator(multiplier=1.1, name="aug1")
        crop1 = MockOperator(multiplier=0.9, name="crop1")
        aug2 = MockOperator(multiplier=1.2, name="aug2")
        crop2 = MockOperator(multiplier=0.8, name="crop2")

        # Build nested structure - wrap operators in OperatorNode for >> composition
        branch1 = OperatorNode(aug1) >> OperatorNode(crop1)
        branch2 = OperatorNode(aug2) >> OperatorNode(crop2)

        executor.add(OperatorNode(normalize)).parallel([branch1, branch2]).merge("mean")

        result = executor(sample_data)

        # Calculate expected result
        normalized = batch_mul(sample_data, 0.5)
        branch1_result = batch_mul(batch_mul(normalized, 1.1), 0.9)
        branch2_result = batch_mul(batch_mul(normalized, 1.2), 0.8)
        expected = batch_mul(batch_add(branch1_result, branch2_result), 0.5)  # mean = (a+b)/2

        assert batch_allclose(result, expected)

    def test_dictionary_data_processing(self, sample_dict_data):
        """Test processing dictionary data."""
        # Disable batch enforcement for direct execution tests
        executor = DAGExecutor(enforce_batch=False)

        # Operator that works with dict data
        class DictOperator(OperatorModule):
            def __init__(self):
                config = OperatorConfig(stochastic=False)
                super().__init__(config, name="dict_operator")

            def apply(self, data, state, metadata, random_params=None, stats=None):
                result = {
                    "image": data["image"] * 2.0,
                    "label": data["label"] + 1,
                    "metadata": data["metadata"] * 0.5,
                }
                return result, state, metadata

        executor.add(OperatorNode(DictOperator()))
        result = executor(sample_dict_data)

        # Result is a Batch with transformed data
        assert isinstance(result, Batch)
        result_data = result.data.get_value()
        input_data = sample_dict_data.data.get_value()
        assert jnp.allclose(result_data["image"], input_data["image"] * 2.0)
        assert jnp.allclose(result_data["label"], input_data["label"] + 1)
        assert jnp.allclose(result_data["metadata"], input_data["metadata"] * 0.5)


class TestDAGExecutorVisualizationExtended:
    """Test DAG visualization functionality - extended tests."""

    def test_visualize_empty_dag(self):
        """Test visualization of empty DAG."""
        executor = DAGExecutor()

        viz = executor.visualize()

        assert isinstance(viz, str)
        assert "DAGExecutor" in viz
        assert "Identity" in viz

    def test_visualize_simple_dag(self):
        """Test visualization of simple DAG."""
        # Disable batch enforcement for visualization test
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator(name="test_transform")))

        viz = executor.visualize()

        assert isinstance(viz, str)
        assert "DAGExecutor" in viz

    def test_repr_string(self):
        """Test string representation."""
        # Disable batch enforcement for this test
        executor = DAGExecutor(enforce_batch=False)
        executor.add(OperatorNode(MockOperator()))

        # Execute once to increment iteration count with Batch
        batch = Batch.from_parts(
            data={"arr": jnp.ones((2, 2))},
            states={"arr": jnp.zeros((2,))},
            validate=False,
        )
        executor(batch)

        repr_str = repr(executor)

        assert "DAGExecutor" in repr_str
        assert "iterations=1" in repr_str
        assert "cached=True" in repr_str
        assert "jit=False" in repr_str


class TestDAGExecutorModuleConversions:
    """Test automatic module-to-node conversions in DAGExecutor.add()."""

    def test_data_source_module_conversion(self):
        """Test DataSourceModule is converted to DataSourceNode."""

        # Create a simple data source
        source = MockDataSource(size=10)
        executor = DAGExecutor()

        # Add the module directly
        executor.add(source)

        # Verify it was converted to DataSourceNode
        assert executor._source_node is not None
        assert isinstance(executor._source_node, DataSourceNode)
        assert executor._source_node.source is source

    def test_operator_module_conversion(self):
        """Test OperatorModule is converted to OperatorNode."""
        executor = DAGExecutor(enforce_batch=False)
        operator = MockOperator(multiplier=2.0)

        # Add the module directly
        executor.add(operator)

        # Verify it was converted to OperatorNode
        assert isinstance(executor.graph, OperatorNode)
        assert executor.graph.operator is operator

    def test_batcher_module_conversion(self):
        """Test BatcherModule is converted to OperatorNode."""
        from datarax.core.batcher import BatcherModule

        # Create a minimal batcher for testing
        class TestBatcher(BatcherModule):
            def __init__(self, batch_size=4):
                config = StructuralConfig(stochastic=False)
                super().__init__(config)
                self.batch_size = batch_size

            def __call__(self, data):
                # Simple pass-through for testing
                return data

        executor = DAGExecutor(enforce_batch=False)
        batcher = TestBatcher(batch_size=4)

        # Add the module directly
        executor.add(batcher)

        # Verify it was converted to OperatorNode (not a BatcherNode)
        assert isinstance(executor.graph, OperatorNode)
        assert executor.graph.operator is batcher

    def test_sampler_module_conversion(self):
        """Test SamplerModule is converted to SamplerNode."""
        from datarax.core.sampler import SamplerModule
        from datarax.dag.nodes import SamplerNode

        # Create a minimal sampler for testing
        class TestSampler(SamplerModule):
            def __init__(self):
                config = StructuralConfig(stochastic=False)
                super().__init__(config)
                self.index = 0

            def __call__(self, idx):
                return list(range(idx))

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= 10:
                    raise StopIteration
                result = self.index
                self.index += 1
                return result

        executor = DAGExecutor(enforce_batch=False)
        sampler = TestSampler()

        # Add the module directly
        executor.add(sampler)

        # Verify it was converted to SamplerNode
        assert isinstance(executor.graph, SamplerNode)
        assert executor.graph.sampler is sampler

    def test_sharder_module_conversion(self):
        """Test SharderModule is converted to SharderNode."""
        from datarax.core.sharder import SharderModule
        from datarax.dag.nodes import SharderNode

        # Create a minimal sharder for testing
        class TestSharder(SharderModule):
            def shard(self, data):
                return data  # Simple pass-through

        executor = DAGExecutor(enforce_batch=False)
        sharder = TestSharder()

        # Add the module directly
        executor.add(sharder)

        # Verify it was converted to SharderNode
        assert isinstance(executor.graph, SharderNode)
        assert executor.graph.sharder is sharder

    def test_mixed_module_conversions_in_pipeline(self):
        """Test that multiple module types convert correctly in sequence."""
        from datarax.core.sampler import SamplerModule
        from datarax.core.sharder import SharderModule
        from datarax.dag.nodes import Sequential, SamplerNode, SharderNode

        # Create minimal test modules
        class TestSampler(SamplerModule):
            def __init__(self):
                config = StructuralConfig(stochastic=False)
                super().__init__(config)

            def __call__(self, idx):
                return list(range(idx))

        class TestSharder(SharderModule):
            def shard(self, data):
                return data

        # Build pipeline with multiple module types
        executor = DAGExecutor(enforce_batch=False)

        # Add different module types
        executor.add(OperatorNode(MockOperator(multiplier=2.0)))  # OperatorModule
        executor.add(TestSampler())  # SamplerModule
        executor.add(TestSharder())  # SharderModule

        # Verify the pipeline structure
        assert isinstance(executor.graph, Sequential)
        assert len(executor.graph.nodes) == 3
        assert isinstance(executor.graph.nodes[0], OperatorNode)
        assert isinstance(executor.graph.nodes[1], SamplerNode)
        assert isinstance(executor.graph.nodes[2], SharderNode)

    def test_node_passed_directly_not_converted(self):
        """Test that nodes passed directly are not double-wrapped."""
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor(enforce_batch=False)
        batch_node = BatchNode(batch_size=32)

        # Add node directly (not a module)
        executor.add(batch_node)

        # Verify it wasn't wrapped again
        assert executor.graph is batch_node

    def test_unsupported_type_raises_error(self):
        """Test that unsupported types raise an error."""
        executor = DAGExecutor(enforce_batch=False)

        # Try to add an unsupported type
        with pytest.raises(ValueError, match="Unsupported type"):
            executor.add("not a module or node")

        with pytest.raises(ValueError, match="Unsupported type"):
            executor.add(42)

        with pytest.raises(ValueError, match="Unsupported type"):
            executor.add(lambda x: x * 2)  # Raw function not wrapped


class TestCollectToArray:
    """Tests for DAGExecutor.collect_to_array() method."""

    @pytest.fixture
    def mock_source_with_get_batch(self):
        """Create a mock data source that supports get_batch()."""

        class BatchableSource(DataSourceModule):
            """Mock source with get_batch support for batch-first execution."""

            data: list = nnx.data()
            index: int = nnx.data()

            def __init__(self, size: int = 100, feature_dim: int = 10):
                config = StructuralConfig(stochastic=False)
                super().__init__(config, name="batchable_source")
                self.size = size
                self.feature_dim = feature_dim
                # Create data with predictable values for testing
                self.data = [
                    {
                        "image": jnp.ones((feature_dim,)) * i,
                        "label": jnp.array(i % 10),
                    }
                    for i in range(size)
                ]
                self.index = 0

            def __len__(self) -> int:
                return self.size

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                result = self.data[self.index]
                self.index += 1
                return result

            def get_batch(self, batch_size: int) -> list:
                """Get a batch of elements."""
                start = self.index
                end = min(self.index + batch_size, len(self.data))
                batch = self.data[start:end]
                self.index = end
                return batch

            def reset(self):
                """Reset the data source index."""
                self.index = 0

        return BatchableSource

    def test_collect_single_key(self, mock_source_with_get_batch):
        """Test collecting single key to array."""
        from datarax.dag.dag_executor import from_source

        source = mock_source_with_get_batch(size=50, feature_dim=8)
        pipeline = from_source(source, batch_size=10)

        result = pipeline.collect_to_array(key="image")

        assert isinstance(result, jax.Array)
        assert result.shape[0] == 50  # Total samples
        assert result.shape[1] == 8  # Feature dimension

    def test_collect_default_key(self, mock_source_with_get_batch):
        """Test collecting with default key='image'."""
        from datarax.dag.dag_executor import from_source

        source = mock_source_with_get_batch(size=20, feature_dim=4)
        pipeline = from_source(source, batch_size=5)

        result = pipeline.collect_to_array()  # Uses default key="image"

        assert isinstance(result, jax.Array)
        assert result.shape == (20, 4)

    def test_collect_to_specific_device(self, mock_source_with_get_batch):
        """Test collecting to specific device."""
        from datarax.dag.dag_executor import from_source

        source = mock_source_with_get_batch(size=10, feature_dim=4)
        pipeline = from_source(source, batch_size=5)

        device = jax.devices()[0]
        result = pipeline.collect_to_array(key="image", device=device)

        assert isinstance(result, jax.Array)
        # Check that result is on the specified device
        assert device in result.devices()

    def test_collect_missing_key_raises(self, mock_source_with_get_batch):
        """Test KeyError for missing key."""
        from datarax.dag.dag_executor import from_source

        source = mock_source_with_get_batch(size=10, feature_dim=4)
        pipeline = from_source(source, batch_size=5)

        with pytest.raises(KeyError, match="Key 'nonexistent' not found"):
            pipeline.collect_to_array(key="nonexistent")

    def test_collect_empty_pipeline_raises(self):
        """Test ValueError for empty pipeline."""
        from datarax.dag.dag_executor import from_source

        class EmptySource(DataSourceModule):
            """A source that yields no elements."""

            data: list = nnx.data()
            index: int = nnx.data()

            def __init__(self):
                config = StructuralConfig(stochastic=False)
                super().__init__(config, name="empty_source")
                self.data = []
                self.index = 0

            def __len__(self) -> int:
                return 0

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                raise StopIteration

            def get_batch(self, batch_size: int) -> list:
                return []

            def reset(self):
                self.index = 0

        source = EmptySource()
        pipeline = from_source(source, batch_size=5)

        with pytest.raises(ValueError, match="no batches"):
            pipeline.collect_to_array()

    def test_collect_preserves_dtype(self, mock_source_with_get_batch):
        """Test dtype preservation during collection."""
        from datarax.dag.dag_executor import from_source

        class Float16Source(DataSourceModule):
            """Source with float16 data."""

            data: list = nnx.data()
            index: int = nnx.data()

            def __init__(self, size: int = 20):
                config = StructuralConfig(stochastic=False)
                super().__init__(config, name="float16_source")
                self.size = size
                self.data = [{"image": jnp.ones((4,), dtype=jnp.float16) * i} for i in range(size)]
                self.index = 0

            def __len__(self) -> int:
                return self.size

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                if self.index >= len(self.data):
                    raise StopIteration
                result = self.data[self.index]
                self.index += 1
                return result

            def get_batch(self, batch_size: int) -> list:
                start = self.index
                end = min(self.index + batch_size, len(self.data))
                batch = self.data[start:end]
                self.index = end
                return batch

            def reset(self):
                self.index = 0

        source = Float16Source(size=10)
        pipeline = from_source(source, batch_size=5)

        result = pipeline.collect_to_array(key="image")

        assert result.dtype == jnp.float16

    def test_collect_with_label_key(self, mock_source_with_get_batch):
        """Test collecting a different key (label)."""
        from datarax.dag.dag_executor import from_source

        source = mock_source_with_get_batch(size=30, feature_dim=4)
        pipeline = from_source(source, batch_size=10)

        result = pipeline.collect_to_array(key="label")

        assert isinstance(result, jax.Array)
        assert result.shape == (30,)  # Labels are scalars per element

    def test_collect_values_correctness(self, mock_source_with_get_batch):
        """Test that collected values are correct."""
        from datarax.dag.dag_executor import from_source

        source = mock_source_with_get_batch(size=5, feature_dim=3)
        pipeline = from_source(source, batch_size=2)

        result = pipeline.collect_to_array(key="image")

        # Each element i has image values all equal to i
        # So result[0] should be [0, 0, 0], result[1] should be [1, 1, 1], etc.
        expected = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
        assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
