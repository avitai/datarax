from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable, Literal, Any

from datarax.dag.nodes.base import Node


class Identity(Node):
    """Identity node that passes data through unchanged.

    Useful as a placeholder or for graph construction.
    """

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Pass data through unchanged."""
        return data


class Sequential(Node):
    """Sequential composition of nodes.

    Executes nodes in sequence, passing the output of each
    as input to the next.

    Examples:
        Basic sequence:

        ```python
        from datarax.nodes import Sequential, Identity
        op1 = Identity(name="op1")
        op2 = Identity(name="op2")
        seq = Sequential([op1, op2])
        # Or using operators:
        seq = op1 >> op2
        ```
    """

    def __init__(self, nodes: list[Node]):
        """Initialize sequential node.

        Args:
            nodes: List of nodes to execute in sequence
        """
        super().__init__()
        # Use nnx.List for proper NNX state tracking
        self.nodes = nnx.List(nodes)

        # Connect nodes in sequence
        for i in range(len(self.nodes) - 1):
            if isinstance(self.nodes[i], Node) and isinstance(self.nodes[i + 1], Node):
                self.nodes[i].connect_to(self.nodes[i + 1])

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Execute nodes sequentially.

        Args:
            data: Input data
            key: Optional RNG key (shared across all nodes)

        Returns:
            Output from the last node
        """
        result = data
        for node in self.nodes:
            # Skip processing if we have None (e.g., from BatchNode buffering)
            if result is None:
                return None

            result = node(result, key=key)
        return result

    def __repr__(self) -> str:
        """String representation."""
        node_names = [n.name if hasattr(n, "name") else str(n) for n in self.nodes]
        return f"Sequential({' >> '.join(node_names)})"


class Parallel(Node):
    """Parallel execution of nodes.

    Executes multiple nodes in parallel on the same input,
    returning a list of outputs.

    Examples:
        Parallel execution:

        ```python
        from datarax.nodes import Parallel, Identity
        op_a = Identity(name="A")
        op_b = Identity(name="B")
        parallel = Parallel([op_a, op_b])
        # Or using operators:
        parallel = op_a | op_b
        ```
    """

    def __init__(self, nodes: list[Node]):
        """Initialize parallel node.

        Args:
            nodes: List of nodes to execute in parallel
        """
        super().__init__()
        # Use nnx.List for proper NNX state tracking
        self.nodes = nnx.List(nodes)

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> list[Any]:
        """Execute nodes in parallel.

        Args:
            data: Input data (sent to all nodes)
            key: Optional RNG key (split across nodes)

        Returns:
            List of outputs from each node
        """
        # Split RNG key if provided
        if key is not None:
            keys = jax.random.split(key, len(self.nodes))
        else:
            keys = [None] * len(self.nodes)

        results = []
        for node, k in zip(self.nodes, keys):
            results.append(node(data, key=k))

        return results

    def __repr__(self) -> str:
        """String representation."""
        node_names = [n.name if hasattr(n, "name") else str(n) for n in self.nodes]
        return f"Parallel({' | '.join(node_names)})"


class Branch(Node):
    """Conditional branching node.

    Executes different paths based on a condition function.

    Note: For JAX JIT compatibility, consider using jax.lax.cond
    instead of Python if/else in the condition function.

    Examples:
        Conditional branching:

        ```python
        from datarax.nodes import Branch, Identity
        branch = Branch(
            condition=lambda x: True,
            true_path=Identity(name="true"),
            false_path=Identity(name="false")
        )
        ```
    """

    def __init__(
        self,
        condition: Callable[[Any], bool],
        true_path: Node,
        false_path: Node,
    ):
        """Initialize branch node.

        Args:
            condition: Function that returns True/False based on input
            true_path: Node to execute if condition is True
            false_path: Node to execute if condition is False
        """
        super().__init__()
        self.condition = condition
        self.true_path = true_path
        self.false_path = false_path

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Execute branch based on condition.

        For NNX modules, uses nnx.cond with modules passed as operands (not closures)
        to enable proper state propagation. For plain callables, uses jax.lax.cond.

        Args:
            data: Input data
            key: Optional RNG key

        Returns:
            Output from selected path
        """
        condition_result = self.condition(data)

        # Use nnx.cond for NNX modules with modules passed as operands
        # This allows NNX to track state mutations (closures don't work for mutable state)
        if isinstance(self.true_path, nnx.Module):
            return nnx.cond(
                condition_result,
                lambda tp, fp, d, k: tp(d, key=k),
                lambda tp, fp, d, k: fp(d, key=k),
                self.true_path,
                self.false_path,
                data,
                key,
            )
        else:
            # Plain callables: capture in closure, use jax.lax.cond
            return jax.lax.cond(
                condition_result,
                lambda d: self.true_path(d, key=key),
                lambda d: self.false_path(d, key=key),
                data,
            )

    def __repr__(self) -> str:
        """String representation."""
        cond_name = self.condition.__name__ if hasattr(self.condition, "__name__") else "lambda"
        true_name = self.true_path.name if hasattr(self.true_path, "name") else str(self.true_path)
        false_name = (
            self.false_path.name if hasattr(self.false_path, "name") else str(self.false_path)
        )
        return f"Branch(condition={cond_name}, true={true_name}, false={false_name})"


class Merge(Node):
    """Merge multiple inputs using various strategies.

    Combines outputs from parallel branches back into a single output.

    Strategies:
    - 'concat': Concatenate along last axis
    - 'sum': Element-wise sum
    - 'mean': Element-wise mean
    - 'stack': Stack into new dimension

    Examples:
        Merge parallel outputs:

        ```python
        from datarax.nodes import Merge, Parallel, Identity
        op_a = Identity(name="A")
        op_b = Identity(name="B")
        pipeline = (Identity(name="input") >>
                   Parallel([op_a, op_b]) >>
                   Merge(strategy='concat'))
        ```
    """

    def __init__(
        self, strategy: Literal["concat", "sum", "mean", "stack"] = "concat", axis: int = -1
    ):
        """Initialize merge node.

        Args:
            strategy: How to merge the inputs
            axis: Axis for concatenation/stacking (default: -1)
        """
        super().__init__()
        self.strategy = strategy
        self.axis = axis

    def __call__(self, inputs: list[Any] | Any, *, key: jax.Array | None = None) -> Any:
        """Merge inputs according to strategy.

        Args:
            inputs: List of inputs to merge (or single input if not from Parallel)
            key: Optional RNG key (unused)

        Returns:
            Merged output
        """
        # Handle single input (pass through)
        if not isinstance(inputs, list):
            return inputs

        # Apply merge strategy
        if self.strategy == "concat":
            # For PyTrees, concatenate each leaf
            return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=self.axis), *inputs)
        elif self.strategy == "sum":
            # Sum all PyTree leaves element-wise
            return jax.tree.map(lambda *xs: sum(xs), *inputs)
        elif self.strategy == "mean":
            # Average all PyTree leaves
            return jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs, axis=0), axis=0), *inputs)
        elif self.strategy == "stack":
            # Stack all PyTree leaves
            return jax.tree.map(lambda *xs: jnp.stack(xs, axis=self.axis), *inputs)
        else:
            raise ValueError(f"Unknown merge strategy: {self.strategy}")

    def __repr__(self) -> str:
        """String representation."""
        return f"Merge(strategy={self.strategy}, axis={self.axis})"


# Convenience functions


def parallel(*nodes: Node) -> Parallel:
    """Create a parallel node from multiple nodes.

    Args:
        nodes: Nodes to execute in parallel

    Returns:
        Parallel node
    """
    return Parallel(nodes)


def branch(
    condition: Callable[[Any], bool],
    true_path: Node,
    false_path: Node,
) -> Branch:
    """Create a branch node with condition and paths.

    Args:
        condition: Function that returns True/False
        true_path: Path if condition is True
        false_path: Path if condition is False

    Returns:
        Branch node
    """
    return Branch(condition, true_path, false_path)
