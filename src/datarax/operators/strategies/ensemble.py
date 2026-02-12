"""Ensemble composition strategies."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from datarax.operators.strategies.base import CompositionStrategyImpl, StrategyContext
from datarax.core.operator import OperatorModule


class EnsembleStrategy(CompositionStrategyImpl):
    """Applies operators in parallel and reduces outputs (mean, sum, etc)."""

    def __init__(self, mode: str):
        """Initialize ensemble strategy.

        Args:
            mode: Reduction mode ("mean", "sum", "max", "min")
        """
        self.mode = mode

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operators in parallel and reduce outputs element-wise.

        Args:
            operators: Operators to execute on identical input.
            context: Execution context with input data, state, and RNG params.

        Returns:
            Tuple of (reduced_data, last_state, last_metadata).

        Raises:
            ValueError: If ``self.mode`` is not one of mean/sum/max/min.
        """
        outputs, states, metadatas = self._execute_operators(operators, context)

        # Reduce based on strategy
        if self.mode == "mean":
            reduced_data = jax.tree.map(lambda *args: jnp.mean(jnp.stack(args), axis=0), *outputs)
        elif self.mode == "sum":
            reduced_data = jax.tree.map(lambda *args: sum(args), *outputs)
        elif self.mode == "max":
            reduced_data = jax.tree.map(lambda *args: jnp.max(jnp.stack(args), axis=0), *outputs)
        elif self.mode == "min":
            reduced_data = jax.tree.map(lambda *args: jnp.min(jnp.stack(args), axis=0), *outputs)
        else:
            raise ValueError(f"Unknown ensemble mode: {self.mode}")

        merged_state = states[-1] if states else context.state
        merged_metadata = metadatas[-1] if metadatas else context.metadata

        return reduced_data, merged_state, merged_metadata
