"""Branching composition strategy."""

import logging
from collections.abc import Callable
from typing import Any

import jax
from jaxtyping import PyTree

from datarax.core.operator import OperatorModule
from datarax.operators.strategies.base import CompositionStrategyImpl, StrategyContext


logger = logging.getLogger(__name__)


class BranchingStrategy(CompositionStrategyImpl):
    """Applies branching strategy with vmap-compatible integer routing."""

    def __init__(self, router: Callable[[PyTree], int | jax.Array]) -> None:
        """Initialize branching strategy.

        Args:
            router: Function that returns integer index (0, 1, 2, ...)
        """
        self.router = router

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Route input to exactly one operator via ``jax.lax.switch``.

        The router function returns an integer index selecting which operator
        to execute. Only the selected branch runs (JIT-efficient).

        Args:
            operators: Candidate operators (indexed by router output).
            context: Execution context with input data, state, and RNG params.

        Returns:
            Tuple of (data, state, metadata) from the selected branch.
        """
        # Router returns integer index (vmap/jit compatible)
        branch_index = self.router(context.data)

        # Create list of branch functions that call each operator's apply method
        def make_branch_fn(i: int, operator: OperatorModule) -> Callable:
            def branch_fn(operands: Any) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
                d, s, m, rp = operands
                # Extract random params for this specific operator
                op_random_params = None
                if rp and f"operator_{i}" in rp:
                    op_random_params = rp[f"operator_{i}"]

                out_data, out_state, out_metadata = operator.apply(d, s, m, op_random_params)

                return out_data, out_state, out_metadata

            return branch_fn

        branches = [make_branch_fn(i, op) for i, op in enumerate(operators)]

        # Use jax.lax.switch to select and call the branch
        # This is JIT-compiled efficiently - only the selected branch executes
        result_data, result_state, result_metadata = jax.lax.switch(
            branch_index,
            branches,
            (context.data, context.state, context.metadata, context.random_params),
        )

        return result_data, result_state, result_metadata
