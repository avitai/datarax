from __future__ import annotations
import flax.nnx as nnx
import jax
from typing import Any


class Node(nnx.Module):
    """Base class for DAG nodes.

    All pipeline components inherit from this base class, providing
    a consistent interface for DAG construction and execution.

    Supports operator-based composition:
    - ``>>`` for sequential composition (e.g., ``node1 >> node2``)
    - ``|`` for parallel composition (e.g., ``node1 | node2``)

    Attributes:
        name: Unique identifier for the node.
        inputs: List of input nodes feeding into this node.
        outputs: List of output nodes this node feeds into.
    """

    def __init__(self, name: str | None = None):
        """Initialize node.

        Args:
            name: Optional name for the node. If not provided, defaults to class name.
        """
        super().__init__()
        self.name = name or self.__class__.__name__
        # Use nnx.List for proper NNX state tracking
        self.inputs: nnx.List[Node] = nnx.List([])
        self.outputs: nnx.List[Node] = nnx.List([])

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Execute the node on input data.

        This method defines the core logic of the node. It transforms the input
        ``data`` and returns the result. Use ``key`` for stochastic operations.

        Args:
            data: Input data to process. Can be any PyTree.
            key: Optional PRNG key for stochastic operations.

        Returns:
            Processed data structure (PyTree).

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def __rshift__(self, other: Node) -> Node:
        """Sequential composition using >> operator.

        Args:
            other: Node to execute after this one

        Returns:
            Sequential node combining both
        """
        # Circular import handling
        from datarax.dag.nodes.control_flow import Sequential

        if isinstance(self, Sequential):
            # Extend existing sequential IN PLACE for performance
            # This avoids O(n²) complexity when building long chains
            if isinstance(other, Sequential):
                # Merge two Sequential nodes
                self.nodes.extend(other.nodes)
            else:
                # Add single node
                self.nodes.append(other)
            return self
        elif isinstance(other, Sequential):
            # Prepend to existing sequential
            return Sequential([self, *other.nodes])
        else:
            # Create new sequential
            return Sequential([self, other])

    def __or__(self, other: Node) -> Node:
        """Parallel composition using | operator.

        Args:
            other: Node to execute in parallel

        Returns:
            Parallel node combining both
        """
        # Circular import handling
        from datarax.dag.nodes.control_flow import Parallel

        if isinstance(self, Parallel):
            # Extend existing parallel IN PLACE for performance
            if isinstance(other, Parallel):
                # Merge two Parallel nodes
                self.nodes.extend(other.nodes)
            else:
                # Add single node
                self.nodes.append(other)
            return self
        elif isinstance(other, Parallel):
            # Prepend to existing parallel
            return Parallel([self, *other.nodes])
        else:
            # Create new parallel
            return Parallel([self, other])

    def connect_to(self, other: Node) -> None:
        """Connect this node to another in the graph.

        Args:
            other: Node to connect to
        """
        self.outputs.append(other)
        other.inputs.append(self)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
