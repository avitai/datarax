"""Branching composition strategy."""

from typing import Any
from collections.abc import Callable

import jax
from jaxtyping import PyTree

from datarax.operators.strategies.base import CompositionStrategyImpl, StrategyContext
from datarax.core.operator import OperatorModule


class BranchingStrategy(CompositionStrategyImpl):
    """Applies branching strategy with vmap-compatible integer routing."""

    def __init__(self, router: Callable[[PyTree], int | jax.Array]):
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
        def make_branch_fn(i, operator):
            def branch_fn(operands):
                d, s, m, rp = operands
                # Extract random params for this specific operator
                op_random_params = None
                if rp and f"operator_{i}" in rp:
                    op_random_params = rp[f"operator_{i}"]

                out_data, out_state, out_metadata = operator.apply(d, s, m, op_random_params)

                # We can't update stats inside branch_fn directly because it's inside lax.switch?
                # Actually lax.switch is compiled. If stats update is JAX side-effect
                # (nnx variable update),
                # it might be tricky.
                # CompositeOperatorModule.apply executes inside apply_batch which is vmapped.
                # NNX variables are updated by returning new state.
                # But stats update in CompositeOperatorModule:
                #    stats = self.operator_statistics.get_value()
                #    stats[...] = ...
                #    self.operator_statistics.set_value(stats)
                # This works if `self` is available.
                # Here we use callback.
                # But callback executes python code.
                # Inside JIT/vmap, callbacks are traced.
                # If stats_callback modifies NNX variable, it needs to happen inside.
                # But StrategyContext callback is Python callable.

                # Re-evaluating stats handling:
                # CompositeOperatorModule._apply_branching does NOT update statistics
                # in current implementation!
                # Let's check the code I read.
                # I read lines 800-938.
                # Lines 825-882 is _apply_branching.
                # It does NOT have stats update logic inside `make_branch_fn` or `branch_fn`.
                # So I can skip stats update for branching strategy for now,
                # or assume it's not supported there.

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
