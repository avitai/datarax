import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any
from datarax.dag.nodes import Node, Sequential, Parallel, OperatorNode, MergeBatchNode
from datarax.core.operator import OperatorModule

from datarax.core.config import OperatorConfig


class MatMulOperator(OperatorModule):
    """An operator that performs adjustable compute (matmul) to simulate load."""

    def __init__(self, input_dim: int, output_dim: int, compute_intensity: int = 1):
        config = OperatorConfig(stochastic=False)
        super().__init__(config, name="matmul_op")
        self.w = nnx.Param(jax.random.normal(jax.random.PRNGKey(0), (input_dim, output_dim)))
        self.compute_intensity = compute_intensity

    def apply(
        self,
        data: Any,
        state: Any,
        metadata: Any,
        random_params: Any = None,
        stats: Any = None,
    ) -> tuple[Any, Any, Any]:
        def transform(x):
            y = x @ self.w.get_value()
            for _ in range(self.compute_intensity):
                y = jnp.sin(y) @ jnp.eye(y.shape[-1])  # Artificial load
            return y

        new_data = jax.tree.map(transform, data)
        return new_data, state, metadata


class ConvOperator(OperatorModule):
    """An operator that simulates a Convolutional layer load."""

    def __init__(self, features: int, kernel_size: tuple[int, int] = (3, 3)):
        config = OperatorConfig(stochastic=False)
        super().__init__(config, name="conv_op")
        # Initialize a standard NNX Conv layer
        # Note: We need a dummy RNG key for initialization in __init__?
        # NNX creates Params lazily or needs a key.
        # We'll stick to manual Param creation for simplicity or use nnx.Conv if initialized
        # carefully.
        # Let's use manual weights to strictly control overhead.
        self.features = features
        self.kernel = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), (*kernel_size, 1, features))
        )  # Depthwise-ish or simplified

    def apply(
        self,
        data: Any,
        state: Any,
        metadata: Any,
        random_params: Any = None,
        stats: Any = None,
    ) -> tuple[Any, Any, Any]:
        def transform(x):
            # Expect x to be (Batch, H, W, C) or similar.
            # If 2D (Batch, Dim), reshape to fake image
            # If 1D or scalar, skip convolution (can't reshape)
            if x.ndim < 2:
                return x
            original_shape = x.shape
            if x.ndim == 2:
                # B, D -> B, 8, D//8, 1 (Approx)
                h = 8
                w = x.shape[1] // h
                if w == 0:
                    return x  # Too small to reshape
                x_img = x.reshape(x.shape[0], h, w, 1)
            else:
                x_img = x

            # Simple manual conv or lax.conv
            # dn = lax.conv(lhs=x_img, rhs=self.kernel.value, window_strides=(1,1), padding='SAME')
            # Using jax.lax directly for benchmark purity
            dn = jax.lax.conv_general_dilated(
                lhs=x_img,
                rhs=self.kernel.get_value(),
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

            if x.ndim == 2:
                return dn.reshape(original_shape[0], -1)[:, : original_shape[1]]
            return dn

        new_data = jax.tree.map(transform, data)
        return new_data, state, metadata


class ActivationOperator(OperatorModule):
    """Simple element-wise activation."""

    def __init__(self, func_name: str = "relu"):
        super().__init__(OperatorConfig(), name=f"{func_name}_op")
        self.func = getattr(jax.nn, func_name, jax.nn.relu)

    def apply(
        self,
        data: Any,
        state: Any,
        metadata: Any,
        random_params: Any = None,
        stats: Any = None,
    ) -> tuple[Any, Any, Any]:
        new_data = jax.tree.map(self.func, data)
        return new_data, state, metadata


class ComplexDAGBuilder:
    """Helper to generate DAG configurations for benchmarking."""

    @staticmethod
    def build_diamond_dag(compute_intensity: int = 1) -> Node:
        """Build a diamond DAG: A -> B, A -> C, B -> D, C -> D.

        This topology tests topological sort correctness with multiple
        paths converging on a single node.
        """
        a = OperatorNode(MatMulOperator(64, 64, compute_intensity), name="A")
        b = OperatorNode(MatMulOperator(64, 64, compute_intensity), name="B")
        c = OperatorNode(MatMulOperator(64, 64, compute_intensity), name="C")
        merge = MergeBatchNode(strategy="mean", name="D")

        # A → Parallel(B, C) → MergeBatch(D)
        return Sequential([a, Parallel([b, c]), merge])

    @staticmethod
    def build_linear_chain(length: int, compute_intensity: int = 1) -> Node:
        """Builds Op -> Op -> Op chain."""
        nodes = []
        for i in range(length):
            op = MatMulOperator(64, 64, compute_intensity)
            node = OperatorNode(op, name=f"linear_{i}")
            nodes.append(node)

        return Sequential(nodes)

    @staticmethod
    def build_width_fanout(width: int, compute_intensity: int = 1) -> Node:
        """Builds Fan-out -> Parallel Ops -> Fan-in Aggregation."""
        # 1. Parallel Branch
        branches = []
        for i in range(width):
            op = MatMulOperator(64, 64, compute_intensity)
            branches.append(OperatorNode(op, name=f"branch_{i}"))

        parallel_block = Parallel(branches)
        # for width scaling, just measuring the overhead of parallel dispatch is enough
        return Sequential([parallel_block])

    @staticmethod
    def build_mixed_topology(depth: int, width: int) -> Node:
        """Builds a mixed topology: Linear -> Split(Width) -> Concat -> Linear (Repeat 'depth').

        This mimics a ResNet block or Inception module repeated 'depth' times.
        """
        stages = []
        for d in range(depth):
            # 1. Pre-process (Conv)
            conv_op = ConvOperator(features=32)
            stages.append(OperatorNode(conv_op, name=f"stage_{d}_conv"))

            # 2. Parallel block (MatMul branches)
            branches = []
            for w in range(width):
                # Mixed operators in branches?
                # Let's alternate MatMul and Activation
                if w % 2 == 0:
                    op = MatMulOperator(64, 64, compute_intensity=2)
                else:
                    op = ActivationOperator("silu")
                branches.append(OperatorNode(op, name=f"stage_{d}_branch_{w}"))

            stages.append(Parallel(branches))

            # 3. Merge parallel outputs back into a single Batch
            stages.append(MergeBatchNode(strategy="mean", name=f"stage_{d}_merge"))

            # 4. Post-process (Activation)
            act_op = ActivationOperator("relu")
            stages.append(OperatorNode(act_op, name=f"stage_{d}_act"))

        return Sequential(stages)
