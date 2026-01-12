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
        outputs = []
        states = []
        metadatas = []

        for i, operator in enumerate(operators):
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]

            out_data, out_state, out_metadata = operator.apply(
                context.data, context.state, context.metadata, op_random_params
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)

            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)

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
