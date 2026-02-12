"""Unified composite operator module.

Implements CompositeOperatorModule with 11 composition strategies:

- Sequential (3 variants): Chain operators
- Parallel (3 variants): Apply all to same input
- Ensemble (4 reductions): Parallel with mean/sum/max/min
- Branching (1 routing): Route through different paths

WEIGHTED_PARALLEL supports three mutually exclusive weight modes:

- **Static weights**: Fixed at construction via ``weights=[0.5, 0.5]``
- **Learnable weights**: Stored as ``nnx.Param`` via ``learnable_weights=True``
- **Dynamic external weights**: Extracted from ``data[weight_key]`` at each call
  via ``weight_key="op_weights"``, enabling upstream modules (e.g., Gumbel-Softmax
  policies) to supply per-call weights with full gradient flow

JAX vmap/JIT Compatibility Patterns
====================================

This module implements several critical patterns for vmap and JIT compatibility:

1. **Integer-Based Branching**:

   - Branching uses ``jax.lax.switch`` with integer indices, not dict lookups
   - Router functions must return integers (0, 1, 2, ...), not strings
   - Why: Traced JAX values cannot be used as dict keys or in Python if statements
   - Pattern: ``jax.lax.switch(index, [fn0, fn1, fn2], operands)``

2. **Fixed-Shape Conditional Outputs**:

   - Conditional strategies include ALL operator outputs (even False conditions)
   - False-condition operators return identity via ``jax.lax.cond`` noop function
   - Why: vmap requires all code paths to return the same PyTree structure
   - Pattern: No dynamic filtering, use masking in merge instead

3. **PyTree Structure Preservation in Dict Merge**:

   - Dict merge returns ``{key: {op_0: val, op_1: val}}`` not ``{op_0: {key: val}}``
   - Why: Preserves input PyTree structure for vmap out_axes specification
   - Pattern: Use ``jax.tree.map()`` to transform leaves into operator dicts

4. **Static Branching with jax.lax.cond**:

   - Conditional execution uses ``jax.lax.cond(condition, true_fn, false_fn, operands)``
   - Why: Python if statements break tracing, ``jax.lax.cond`` is trace-compatible
   - Pattern: Define apply_fn and noop_fn, use ``jax.lax.cond`` for selection

5. **weight_key Data Stripping**:

   - When ``weight_key`` is set, it is stripped from both ``data`` (in ``apply()``)
     and ``data_shapes`` (in ``generate_random_params()``)
   - Why: Children's random param trees must match the clean data they receive;
     a shape mismatch causes vmap failures
   - Pattern: Dict comprehension ``{k: v for k, v in d.items() if k != weight_key}``

These patterns ensure all strategies work correctly inside jax.vmap and jax.jit.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


class CompositionStrategy(Enum):
    """Strategy for composing multiple operators."""

    # Sequential strategies
    SEQUENTIAL = auto()  # Chain operators: out₁ → in₂
    CONDITIONAL_SEQUENTIAL = auto()  # Chain with conditions
    DYNAMIC_SEQUENTIAL = auto()  # Runtime-modifiable chain

    # Parallel strategies
    PARALLEL = auto()  # Apply all to same input, merge
    WEIGHTED_PARALLEL = auto()  # Parallel with weights
    CONDITIONAL_PARALLEL = auto()  # Parallel with conditions

    # Reduction strategies (ensemble)
    ENSEMBLE_MEAN = auto()  # Parallel + mean reduction
    ENSEMBLE_SUM = auto()  # Parallel + sum reduction
    ENSEMBLE_MAX = auto()  # Parallel + max reduction
    ENSEMBLE_MIN = auto()  # Parallel + min reduction

    # Routing strategies
    BRANCHING = auto()  # Route through different paths


@dataclass
class CompositeOperatorConfig(OperatorConfig):
    """Configuration for composite operators.

    Inherits from OperatorConfig:

        - name: str | None
        - stochastic: bool (whether any child is stochastic)
        - stream_name: str (for RNG if stochastic)

    WEIGHTED_PARALLEL supports three mutually exclusive weight modes:

        1. **Static weights** (default): ``weights=[0.5, 0.5]`` — fixed at construction.
        2. **Learnable weights**: ``learnable_weights=True`` — stored as ``nnx.Param``,
           optimized via gradient descent.
        3. **Dynamic external weights**: ``weight_key="op_weights"`` — extracted from
           ``data[weight_key]`` at each forward call. Enables upstream modules (e.g.,
           a Gumbel-Softmax policy) to supply weights that change per call, with
           gradients flowing back through the weights to the upstream parameters.

    When ``weight_key`` is set, the key is stripped from the data dict before
    passing to child operators, so children only see the actual data fields.

    Attributes:
        strategy: Composition strategy to use.
        operators: List of operators for all strategies.
        merge_strategy: How to merge parallel outputs ("concat", "stack", "sum", "mean", "dict").
        merge_fn: Custom merge function (overrides merge_strategy).
        merge_axis: Axis for stack/concat operations.
        weights: Weights for weighted parallel (None = equal weights).
        learnable_weights: Whether weights are learnable parameters.
        weight_key: Key in data dict for external dynamic weights. Mutually exclusive
            with ``weights`` and ``learnable_weights``. When set, weights are extracted
            from ``data[weight_key]`` at each call and the key is stripped from child data.
        conditions: Conditions for conditional strategies (returns JAX arrays).
        router: Router function for branching (returns integer index).
        default_branch: Default branch index for fallback behavior.
    """

    # Core composition settings
    strategy: CompositionStrategy | None = field(default=None)
    operators: list[OperatorModule] | None = field(default=None)

    # Merge settings (for parallel/ensemble strategies)
    merge_strategy: str | None = None  # "concat", "stack", "sum", "mean", "dict"
    merge_fn: Callable | None = None  # Custom merge function
    merge_axis: int = 0  # Axis for stack/concat

    # Weights (for weighted parallel)
    weights: list[float] | None = None
    learnable_weights: bool = False
    weight_key: str | None = None  # Key in data dict for external dynamic weights

    # Conditions (for conditional strategies)
    # Conditions can return Python bool or JAX scalar (converted automatically)
    conditions: list[Callable[[PyTree], bool | jax.Array]] | None = None
    # Note: require_at_least_one feature removed - incompatible with vmap tracing
    # Future: Could validate static conditions at config time, but dynamic
    # data-dependent conditions cannot be checked inside vmap

    # Router (for branching strategy)
    # Router returns integer index (Python int or JAX scalar) of operator to use
    router: Callable[[PyTree], int | jax.Array] | None = None
    default_branch: int | None = None  # Default branch index (fallback if needed)

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()

        # Validate required fields
        if self.strategy is None:
            raise ValueError("strategy is required")
        if self.operators is None:
            raise ValueError("operators is required")

        # Validate operators type
        if not isinstance(self.operators, list):
            raise ValueError("operators must be a list")
        if not self.operators:
            raise ValueError("operators list cannot be empty")

        # Validate strategy-specific requirements
        if self.strategy in [
            CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            CompositionStrategy.CONDITIONAL_PARALLEL,
        ]:
            if self.conditions is None:
                raise ValueError(f"{self.strategy.name} requires conditions")
            if len(self.conditions) != len(self.operators):
                raise ValueError("Number of conditions must match number of operators")

        if self.strategy == CompositionStrategy.WEIGHTED_PARALLEL:
            if self.weight_key is not None:
                # Dynamic mode: weights come from data[weight_key] at call time
                if self.learnable_weights:
                    raise ValueError("Cannot combine weight_key with learnable_weights")
                if self.weights is not None:
                    raise ValueError("Cannot combine weight_key with explicit weights")
            elif self.weights is None:
                # Default: equal weights
                object.__setattr__(
                    self, "weights", [1.0 / len(self.operators)] * len(self.operators)
                )
            elif len(self.weights) != len(self.operators):
                raise ValueError("Number of weights must match number of operators")

        if self.strategy == CompositionStrategy.BRANCHING:
            if self.router is None:
                raise ValueError("BRANCHING strategy requires router function")

        # Determine if composite is stochastic (any child stochastic)
        if isinstance(self.operators, list):
            child_stochastic = any(getattr(op.config, "stochastic", False) for op in self.operators)
        else:  # dict
            child_stochastic = any(
                getattr(op.config, "stochastic", False) for op in self.operators.values()
            )

        # Override stochastic based on children
        object.__setattr__(self, "stochastic", child_stochastic)


class CompositeOperatorModule(OperatorModule):
    """Unified composite operator supporting all composition strategies.

    Uses the Strategy Pattern internally — each ``CompositionStrategy`` enum value
    maps to a strategy implementation class (e.g., ``WeightedParallelStrategy``).

    For ``WEIGHTED_PARALLEL`` with ``weight_key``, the composite extracts weights
    from the data dict at each forward call, strips the key from child data, and
    delegates to ``WeightedParallelStrategy`` for the weighted sum. This enables
    differentiable pipelines where an upstream module (e.g., Gumbel-Softmax policy)
    supplies per-call weights with full gradient flow.
    """

    def __init__(
        self,
        config: CompositeOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize composite operator.

        Args:
            config: Composite operator configuration
            rngs: Optional RNGs for stochastic operators
        """
        super().__init__(config, rngs=rngs)

        # Type narrowing for pyright - config is CompositeOperatorConfig
        self.config: CompositeOperatorConfig = config

        # Store operators in appropriate container
        if isinstance(config.operators, dict):
            self.operators = nnx.Dict(config.operators)
        else:
            self.operators = nnx.List(config.operators)

        # Initialize weights if learnable
        if config.strategy == CompositionStrategy.WEIGHTED_PARALLEL and config.learnable_weights:
            self.weights = nnx.Param(jnp.array(config.weights))

        # Statistics tracking
        self.operator_statistics = nnx.Variable({})

        # Initialize strategy implementation
        self._init_strategy()

    def _init_strategy(self):
        """Initialize the composition strategy implementation."""
        from datarax.operators.strategies import (
            SequentialStrategy,
            ConditionalSequentialStrategy,
            ParallelStrategy,
            WeightedParallelStrategy,
            ConditionalParallelStrategy,
            EnsembleStrategy,
            BranchingStrategy,
        )

        if self.config.strategy == CompositionStrategy.SEQUENTIAL:
            self.strategy_impl = SequentialStrategy()
        elif self.config.strategy == CompositionStrategy.CONDITIONAL_SEQUENTIAL:
            self.strategy_impl = ConditionalSequentialStrategy(self.config.conditions)
        elif self.config.strategy == CompositionStrategy.DYNAMIC_SEQUENTIAL:
            self.strategy_impl = SequentialStrategy()
        elif self.config.strategy == CompositionStrategy.PARALLEL:
            self.strategy_impl = ParallelStrategy(
                merge_strategy=self.config.merge_strategy,
                merge_axis=self.config.merge_axis,
                merge_fn=self.config.merge_fn,
            )
        elif self.config.strategy == CompositionStrategy.WEIGHTED_PARALLEL:
            self.strategy_impl = WeightedParallelStrategy()
        elif self.config.strategy == CompositionStrategy.CONDITIONAL_PARALLEL:
            self.strategy_impl = ConditionalParallelStrategy(
                conditions=self.config.conditions,
                merge_strategy=self.config.merge_strategy,
                merge_axis=self.config.merge_axis,
                merge_fn=self.config.merge_fn,
            )
        elif self.config.strategy in [
            CompositionStrategy.ENSEMBLE_MEAN,
            CompositionStrategy.ENSEMBLE_SUM,
            CompositionStrategy.ENSEMBLE_MAX,
            CompositionStrategy.ENSEMBLE_MIN,
        ]:
            # Extract mode from enum name
            mode = self.config.strategy.name.split("_")[1].lower()
            self.strategy_impl = EnsembleStrategy(mode=mode)
        elif self.config.strategy == CompositionStrategy.BRANCHING:
            self.strategy_impl = BranchingStrategy(router=self.config.router)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> dict[str, Any]:
        """Generate random parameters for all child operators.

        When ``weight_key`` is configured, strips that key from ``data_shapes``
        before delegating to children. This ensures children's random param trees
        match the clean data they receive (without the weight key), preventing
        PyTree structure mismatches during vmap.
        """
        # Strip weight_key from data_shapes so children's random params
        # match the clean data they actually receive (without the weight key)
        child_data_shapes = data_shapes
        if (
            self.config.weight_key is not None
            and isinstance(data_shapes, dict)
            and self.config.weight_key in data_shapes
        ):
            child_data_shapes = {
                k: v for k, v in data_shapes.items() if k != self.config.weight_key
            }

        # Split RNG into n_operators keys (one per child)
        operators = self._get_operators_list()
        n_operators = len(operators)
        child_rngs = jax.random.split(rng, n_operators)

        # Generate random params for each child operator
        random_params = {}
        for i, operator in enumerate(operators):
            # Each child gets its own RNG slice
            child_rng = child_rngs[i]

            # Call child's generate_random_params
            child_params = operator.generate_random_params(child_rng, child_data_shapes)

            # Store under operator index key
            random_params[f"operator_{i}"] = child_params

        return random_params

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply composition based on the configured strategy.

        For ``WEIGHTED_PARALLEL`` with ``weight_key``, extracts weights from
        ``data[weight_key]``, strips the key from data, and passes clean data
        to the strategy. Raises ``ValueError`` if the key is missing from data.
        """
        from datarax.operators.strategies.base import StrategyContext

        # Prepare context
        extra_params = {}
        clean_data = data

        if self.config.strategy == CompositionStrategy.WEIGHTED_PARALLEL:
            if self.config.weight_key is not None:
                # Dynamic mode: extract weights from data dict
                if not isinstance(data, dict) or self.config.weight_key not in data:
                    raise ValueError(
                        f"weight_key '{self.config.weight_key}' not found in data. "
                        f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
                    )
                extra_params["weights"] = data[self.config.weight_key]
                # Strip weight_key so child operators don't receive it
                clean_data = {k: v for k, v in data.items() if k != self.config.weight_key}
            elif self.config.learnable_weights:
                extra_params["weights"] = self.weights.get_value()
            else:
                extra_params["weights"] = jnp.array(self.config.weights)

        # Stats callback
        def stats_callback(index: int, stats: dict[str, Any]):
            # Updates NNX variable
            current_stats = self.operator_statistics.get_value()
            current_stats[f"operator_{index}"] = stats
            self.operator_statistics.set_value(current_stats)

        context = StrategyContext(
            data=clean_data,
            state=state,
            metadata=metadata if metadata is not None else {},
            random_params=random_params,
            extra_params=extra_params if extra_params else None,
            stats_callback=stats_callback,
        )

        return self.strategy_impl.apply(self._get_operators_list(), context)

    def _get_operators_list(self) -> list[OperatorModule]:
        """Get list of operators."""
        if isinstance(self.operators, nnx.Dict):
            return list(self.operators.values())
        return list(self.operators)

    # Dynamic sequential methods
    def add_operator(self, operator: OperatorModule, index: int | None = None) -> None:
        """Add operator to dynamic sequential."""
        if self.config.strategy != CompositionStrategy.DYNAMIC_SEQUENTIAL:
            raise ValueError("add_operator only available for DYNAMIC_SEQUENTIAL")

        if index is None:
            self.operators.append(operator)
        else:
            self.operators.insert(index, operator)

    def remove_operator(self, index: int) -> OperatorModule:
        """Remove operator from dynamic sequential."""
        if self.config.strategy != CompositionStrategy.DYNAMIC_SEQUENTIAL:
            raise ValueError("remove_operator only available for DYNAMIC_SEQUENTIAL")

        return self.operators.pop(index)

    def clear_operators(self) -> None:
        """Clear all operators from dynamic sequential."""
        if self.config.strategy != CompositionStrategy.DYNAMIC_SEQUENTIAL:
            raise ValueError("clear_operators only available for DYNAMIC_SEQUENTIAL")

        self.operators.clear()

    def reorder_operators(self, new_order: list[int]) -> None:
        """Reorder operators in dynamic sequential."""
        if self.config.strategy != CompositionStrategy.DYNAMIC_SEQUENTIAL:
            raise ValueError("reorder_operators only available for DYNAMIC_SEQUENTIAL")

        if len(new_order) != len(self.operators):
            raise ValueError("new_order must have same length as operators")

        # Create new list with reordered operators
        operators_list = self._get_operators_list()
        reordered = [operators_list[i] for i in new_order]

        self.operators.clear()
        for op in reordered:
            self.operators.append(op)
