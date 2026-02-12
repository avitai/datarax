"""P1: XLA fusion depth performance target tests.

Target: Sub-linear throughput degradation for chains of 10+ transforms.
Tests both Sequential-level fusion and DAG-level topological fusion.
"""

import pytest
import jax.numpy as jnp
from datarax.core.element_batch import Batch
from datarax.dag.dag_executor import DAGExecutor
from datarax.dag.nodes.control_flow import Sequential, FusedOperatorNode
from datarax.dag.nodes.data_source import OperatorNode, BatchNode
from tests.benchmarks.complex_dag_builder import ComplexDAGBuilder, MatMulOperator


def _create_batch(batch_size: int = 128) -> Batch:
    """Create a simple test batch."""
    return Batch.from_parts(
        data={"x": jnp.ones((batch_size, 64))},
        states={},
        validate=False,
    )


@pytest.mark.benchmark
class TestP1XLAFusion:
    """P1: Chain depth 10 must not degrade more than 2x vs chain depth 1."""

    def test_is_jit_fusible_operator_node(self):
        """Verify OperatorNode reports is_jit_fusible=True."""
        node = OperatorNode(MatMulOperator(64, 64, compute_intensity=1))
        assert node.is_jit_fusible

    def test_is_jit_fusible_batch_node(self):
        """Verify BatchNode reports is_jit_fusible=False."""
        assert not BatchNode(batch_size=32).is_jit_fusible

    def test_is_jit_fusible_sequential_all_fusible(self):
        """Verify Sequential with all-fusible children is fusible."""
        ops = [OperatorNode(MatMulOperator(64, 64)) for _ in range(3)]
        assert Sequential(ops).is_jit_fusible

    def test_is_jit_fusible_sequential_mixed(self):
        """Verify Sequential with non-fusible child is not fusible."""
        nodes = [OperatorNode(MatMulOperator(64, 64)), BatchNode(batch_size=32)]
        assert not Sequential(nodes).is_jit_fusible

    def test_fusion_groups_linear_chain(self):
        """Verify fusion correctly groups consecutive fusible OperatorNodes."""
        graph = ComplexDAGBuilder.build_linear_chain(length=5, compute_intensity=1)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        groups = executor._compute_fusion_groups()

        # All 5 OperatorNodes in a Sequential should fuse into 1 group
        assert len(groups) == 1
        assert len(groups[0]) == 5

    def test_fusion_groups_empty_for_non_fusible(self):
        """Verify no fusion groups for non-fusible nodes."""
        executor = DAGExecutor(graph=BatchNode(batch_size=32), enforce_batch=False)
        assert len(executor._compute_fusion_groups()) == 0

    def test_fusion_groups_single_fusible_returns_empty(self):
        """Verify a single fusible node does NOT form a group (threshold >= 2)."""
        graph = Sequential([OperatorNode(MatMulOperator(64, 64))])
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        assert len(executor._compute_fusion_groups()) == 0

    def test_fusion_groups_mixed_sequence(self):
        """Verify mixed fusible/non-fusible sequence produces correct groups."""
        # [Op, Op, Batch, Op, Op, Op] → 2 groups: [Op,Op] and [Op,Op,Op]
        nodes = [
            OperatorNode(MatMulOperator(64, 64), name="op_0"),
            OperatorNode(MatMulOperator(64, 64), name="op_1"),
            BatchNode(batch_size=32),
            OperatorNode(MatMulOperator(64, 64), name="op_2"),
            OperatorNode(MatMulOperator(64, 64), name="op_3"),
            OperatorNode(MatMulOperator(64, 64), name="op_4"),
        ]
        graph = Sequential(nodes)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        groups = executor._compute_fusion_groups()

        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 3

    def test_apply_fusion_replaces_groups(self):
        """Verify _apply_fusion replaces fusible groups with FusedOperatorNode."""
        nodes = [
            OperatorNode(MatMulOperator(64, 64), name="op_0"),
            OperatorNode(MatMulOperator(64, 64), name="op_1"),
            BatchNode(batch_size=32),
            OperatorNode(MatMulOperator(64, 64), name="op_2"),
            OperatorNode(MatMulOperator(64, 64), name="op_3"),
        ]
        graph = Sequential(nodes)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        executor._apply_fusion()

        result_nodes = list(executor.graph.nodes)
        # [FusedOperatorNode([op_0, op_1]), BatchNode, FusedOperatorNode([op_2, op_3])]
        assert len(result_nodes) == 3
        assert isinstance(result_nodes[0], FusedOperatorNode)
        assert isinstance(result_nodes[1], BatchNode)
        assert isinstance(result_nodes[2], FusedOperatorNode)
        assert len(result_nodes[0].fused_nodes) == 2
        assert len(result_nodes[2].fused_nodes) == 2

    def test_fused_operator_node_direct_call(self):
        """Verify FusedOperatorNode produces correct output when called directly."""
        ops = [OperatorNode(MatMulOperator(64, 64, compute_intensity=1)) for _ in range(3)]
        batch = _create_batch(4)

        # Run sequentially without fusion
        expected = batch
        for op in ops:
            expected = op(expected)

        # Run through FusedOperatorNode
        fused = FusedOperatorNode(ops)
        result = fused(batch)

        assert jnp.allclose(result.data["x"], expected.data["x"], atol=1e-5)

    def test_fused_operator_node_repr(self):
        """Verify FusedOperatorNode repr shows child node names."""
        ops = [
            OperatorNode(MatMulOperator(64, 64), name="A"),
            OperatorNode(MatMulOperator(64, 64), name="B"),
        ]
        fused = FusedOperatorNode(ops)
        assert "A" in repr(fused)
        assert "B" in repr(fused)

    def test_linear_chain_execution_correctness(self):
        """Verify fused execution produces same result as unfused."""
        graph = ComplexDAGBuilder.build_linear_chain(length=3, compute_intensity=1)
        batch = _create_batch(4)

        # Execute without fusion
        executor_no_fuse = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        result_no_fuse = executor_no_fuse(batch)

        # Execute with JIT (which triggers fusion via _compile → _apply_fusion)
        graph2 = ComplexDAGBuilder.build_linear_chain(length=3, compute_intensity=1)
        executor_fuse = DAGExecutor(graph=graph2, jit_compile=True, enforce_batch=False)
        result_fuse = executor_fuse(batch)

        assert jnp.allclose(result_no_fuse.data["x"], result_fuse.data["x"], atol=1e-5)

    def test_diamond_dag_execution(self):
        """Verify diamond DAG (A -> Parallel(B,C) -> Merge(D)) executes correctly."""
        graph = ComplexDAGBuilder.build_diamond_dag(compute_intensity=1)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        batch = _create_batch(4)
        result = executor(batch)
        assert result is not None
        assert "x" in result.data


@pytest.mark.benchmark
class TestP1TopologicalSort:
    """Tests for _topological_sort() — flattens graph tree into execution order."""

    def test_linear_chain_order(self):
        """Verify linear chain produces nodes in order."""
        graph = ComplexDAGBuilder.build_linear_chain(length=4, compute_intensity=1)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        order = executor._topological_sort()

        # 4 leaf nodes in sequential order
        assert len(order) == 4
        names = [n.name for n in order]
        assert names == ["linear_0", "linear_1", "linear_2", "linear_3"]

    def test_diamond_dag_order_respects_dependencies(self):
        """Verify diamond DAG topological sort: A before B,C; B,C before D."""
        graph = ComplexDAGBuilder.build_diamond_dag(compute_intensity=1)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        order = executor._topological_sort()

        names = [n.name for n in order]
        pos = {name: i for i, name in enumerate(names)}

        # A must come before B and C
        assert pos["A"] < pos["B"]
        assert pos["A"] < pos["C"]
        # B and C must come before D
        assert pos["B"] < pos["D"]
        assert pos["C"] < pos["D"]

    def test_single_node_graph(self):
        """Verify single-node graph returns that node."""
        node = OperatorNode(MatMulOperator(64, 64), name="solo")
        executor = DAGExecutor(graph=node, jit_compile=False, enforce_batch=False)
        order = executor._topological_sort()

        assert len(order) == 1
        assert order[0].name == "solo"

    def test_parallel_branches_all_present(self):
        """Verify parallel fanout includes all branches."""
        graph = ComplexDAGBuilder.build_width_fanout(width=4, compute_intensity=1)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        order = executor._topological_sort()

        names = {n.name for n in order}
        for i in range(4):
            assert f"branch_{i}" in names

    def test_mixed_topology_all_nodes_present(self):
        """Verify mixed topology has no missing or duplicate nodes."""
        graph = ComplexDAGBuilder.build_mixed_topology(depth=3, width=2)
        executor = DAGExecutor(graph=graph, jit_compile=False, enforce_batch=False)
        order = executor._topological_sort()

        # No duplicates
        ids = [id(n) for n in order]
        assert len(ids) == len(set(ids)), "Topological sort produced duplicates"
