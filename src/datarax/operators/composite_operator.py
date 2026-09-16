"""Unified composite operator module.

Implements CompositeOperatorModule with 11 composition strategies:

- Sequential (3 variants): Chain operators
- Parallel (3 variants): Apply all to same input
- Ensemble (4 reductions): Parallel with mean/sum/max/min
- Branching (1 routing): Route through different paths

WEIGHTED_PARALLEL replaces the fields named in ``mix_fields`` with the weighted sum of the
operators' outputs and passes every other field through. It supports three mutually
exclusive weight modes:

- **Static weights**: A linear combination fixed at construction via ``weights=[1.0, 0.1]``
- **Learnable weights**: Logits stored as ``nnx.Param`` via ``learnable_weights=True``,
  mixed with ``softmax(logits / temperature)``
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

   - When ``weight_key`` is set, it is stripped from ``data`` in ``apply()`` before any
     child runs
   - Why: children see the fields the composite promises them; the weight is the
     composite's own input, not theirs
   - Pattern: Dict comprehension ``{k: v for k, v in d.items() if k != weight_key}``

These patterns ensure all strategies work correctly inside jax.vmap and jax.jit.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import child_statistics, OperatorModule, require_key


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CompositeOperatorConfig(OperatorConfig):
    """Configuration for composite operators.

    Inherits from OperatorConfig:

        - name: str | None
        - stochastic: bool (whether any child is stochastic)
        - stream_name: str (for RNG if stochastic)

    WEIGHTED_PARALLEL supports three mutually exclusive weight modes:

        1. **Static weights** (default): ``weights=[1.0, 0.1]`` — a linear combination
           fixed at construction.
        2. **Learnable weights**: ``learnable_weights=True`` — logits stored as
           ``nnx.Param``, mixed with ``softmax(logits / temperature)`` and optimized via
           gradient descent.
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
        weights: Weights for weighted parallel (None = equal weights). Static weights form
            a linear combination, e.g. DDSP's sum of harmonic and noise synthesizers.
        learnable_weights: Whether the weights are learned. They are stored as logits
            initialized to ``log(weights / sum(weights))`` and mixed with
            ``softmax(logits / temperature)``, the relaxation DARTS and Faster AutoAugment use.
        weight_key: Key in data dict for external dynamic weights. Mutually exclusive
            with ``weights`` and ``learnable_weights``. When set, weights are extracted
            from ``data[weight_key]`` at each call and the key is stripped from child data.
        mix_fields: Dotted paths of the data fields a weighted parallel combines. Every
            other field passes through from the input unchanged. Defaults to the fields the
            operators declare they write (``target_key`` or ``field_key``); required when an
            operator declares none.
        temperature: Softmax temperature for learnable weights; must be positive.
        conditions: Conditions for conditional strategies (returns JAX arrays).
        router: Router function for branching (returns integer index).
        default_branch: Default branch index for fallback behavior.
    """

    # Core composition settings
    strategy: CompositionStrategy | None = field(default=None)
    operators: Sequence[OperatorModule] | None = field(default=None)

    # Merge settings (for parallel/ensemble strategies)
    merge_strategy: str | None = None  # "concat", "stack", "sum", "mean", "dict"
    merge_fn: Callable | None = None  # Custom merge function
    merge_axis: int = 0  # Axis for stack/concat

    # Weights (for weighted parallel)
    weights: list[float] | None = None
    learnable_weights: bool = False
    weight_key: str | None = None  # Key in data dict for external dynamic weights
    mix_fields: Sequence[str] | None = None  # Fields combined; others pass through
    temperature: float = 1.0  # Softmax temperature for learnable weights

    # Conditions (for conditional strategies)
    # Conditions can return Python bool or JAX scalar (converted automatically)
    conditions: Sequence[Callable[[PyTree], bool | jax.Array]] | None = None
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
        self._validate_required_fields()
        self._validate_conditional_strategies()
        self._validate_weighted_parallel_strategy()
        self._validate_branching_strategy()
        has_stochastic_child = self._has_stochastic_child()
        if has_stochastic_child and self.stream_name is None:
            object.__setattr__(self, "stream_name", "composite")
        object.__setattr__(self, "stochastic", has_stochastic_child)

    def _validate_required_fields(self) -> None:
        """Validate required fields and operator container basics."""
        if self.strategy is None:
            raise ValueError("strategy is required")
        if self.operators is None:
            raise ValueError("operators is required")
        if not isinstance(self.operators, list):
            raise ValueError("operators must be a list")
        if not self.operators:
            raise ValueError("operators list cannot be empty")

    def _validate_conditional_strategies(self) -> None:
        """Validate conditions for conditional composition strategies."""
        if self.strategy in [
            CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            CompositionStrategy.CONDITIONAL_PARALLEL,
        ]:
            if self.conditions is None:
                raise ValueError(f"{self.strategy.name} requires conditions")
            if self.operators is None:
                raise ValueError("operators is required")
            if len(self.conditions) != len(self.operators):
                raise ValueError("Number of conditions must match number of operators")

    def _validate_weighted_parallel_strategy(self) -> None:
        """Validate weighted-parallel configuration and resolve its defaults."""
        if self.strategy != CompositionStrategy.WEIGHTED_PARALLEL:
            return
        if self.operators is None:
            raise ValueError("operators is required")
        self._validate_weight_mode(self.operators)
        self._resolve_mix_fields(self.operators)

    def _validate_weight_mode(self, operators: Sequence[OperatorModule]) -> None:
        """Validate the weight source: an external key, learnable logits or static weights."""
        if self.weight_key is not None:
            if self.learnable_weights:
                raise ValueError("Cannot combine weight_key with learnable_weights")
            if self.weights is not None:
                raise ValueError("Cannot combine weight_key with explicit weights")
            return
        if self.weights is None:
            object.__setattr__(self, "weights", [1.0 / len(operators)] * len(operators))
        elif len(self.weights) != len(operators):
            raise ValueError("Number of weights must match number of operators")
        if not self.learnable_weights:
            return
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if any(weight <= 0 for weight in cast(list[float], self.weights)):
            raise ValueError(
                "learnable_weights needs positive initial weights: the logits start at "
                f"log(weights / sum(weights)), got {self.weights}"
            )

    def _resolve_mix_fields(self, operators: Sequence[OperatorModule]) -> None:
        """Default ``mix_fields`` to the fields the operators declare they write."""
        if self.mix_fields is not None:
            if not self.mix_fields:
                raise ValueError("mix_fields must name at least one field")
            object.__setattr__(self, "mix_fields", tuple(self.mix_fields))
            return
        declared: list[str] = []
        for index, operator in enumerate(operators):
            written = getattr(operator.config, "target_key", None) or getattr(
                operator.config, "field_key", None
            )
            if written is None:
                raise ValueError(
                    f"WEIGHTED_PARALLEL needs mix_fields: operator {index} "
                    f"({type(operator).__name__}) declares no field_key, so the fields to "
                    "combine cannot be inferred"
                )
            declared.append(written)
        object.__setattr__(self, "mix_fields", tuple(dict.fromkeys(declared)))

    def _validate_branching_strategy(self) -> None:
        """Validate branching strategy requirements."""
        if self.strategy == CompositionStrategy.BRANCHING and self.router is None:
            raise ValueError("BRANCHING strategy requires router function")

    def _has_stochastic_child(self) -> bool:
        """Infer whether any child operator is stochastic."""
        if self.operators is None:
            return False
        return any(getattr(op.config, "stochastic", False) for op in self.operators)


class CompositeOperatorModule(OperatorModule):
    """Unified composite operator supporting all composition strategies.

    Uses the Strategy Pattern internally — each ``CompositionStrategy`` enum value
    maps to a strategy implementation class (e.g., ``WeightedParallelStrategy``).

    For ``WEIGHTED_PARALLEL`` with ``weight_key``, the composite extracts weights
    from the data dict at each forward call, strips the key from child data, and
    delegates to ``WeightedParallelStrategy`` for the weighted sum of the ``mix_fields``.
    This enables
    differentiable pipelines where an upstream module (e.g., Gumbel-Softmax policy)
    supplies per-call weights with full gradient flow.
    """

    def __init__(
        self,
        config: CompositeOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
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

        # Learnable weights are logits; the mixture is softmax(logits / temperature), which
        # starts at the configured weights normalized to sum to one.
        if config.strategy == CompositionStrategy.WEIGHTED_PARALLEL and config.learnable_weights:
            initial = jnp.asarray(config.weights)
            self.weight_logits = nnx.Param(jnp.log(initial / jnp.sum(initial)))

        # Initialize strategy implementation
        self._init_strategy()

    def _init_strategy(self) -> None:
        """Initialize the composition strategy implementation."""
        strategy = self.config.strategy
        builder = self._strategy_builders().get(strategy) if strategy is not None else None
        if builder is None:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy_impl = builder()

    def _strategy_builders(self) -> dict[CompositionStrategy, Callable[[], Any]]:
        """Map each composition strategy to a zero-arg factory for its implementation.

        ``conditions``/``router`` are guaranteed non-``None`` by the config's
        ``__post_init__`` for the strategies that require them, so they are narrowed
        with ``cast`` rather than re-validated here. The four ``ENSEMBLE_*`` strategies
        share one factory that reads the reduction mode from the enum name.

        Returns:
            Mapping from strategy enum to a callable building its ``CompositionStrategyImpl``.
        """
        from datarax.operators.strategies import (
            BranchingStrategy,
            ConditionalParallelStrategy,
            ConditionalSequentialStrategy,
            EnsembleStrategy,
            ParallelStrategy,
            SequentialStrategy,
            WeightedParallelStrategy,
        )

        cfg = self.config

        def build_ensemble() -> Any:
            # Only reached for ENSEMBLE_* keys, so strategy is a concrete enum here.
            # Extract mode from enum name, e.g. ENSEMBLE_MEAN -> "mean".
            strategy = cast(CompositionStrategy, cfg.strategy)
            return EnsembleStrategy(mode=strategy.name.split("_")[1].lower())

        return {
            CompositionStrategy.SEQUENTIAL: SequentialStrategy,
            CompositionStrategy.DYNAMIC_SEQUENTIAL: SequentialStrategy,
            CompositionStrategy.CONDITIONAL_SEQUENTIAL: lambda: ConditionalSequentialStrategy(
                cast(Sequence[Callable], cfg.conditions)
            ),
            CompositionStrategy.PARALLEL: lambda: ParallelStrategy(
                merge_strategy=cfg.merge_strategy,
                merge_axis=cfg.merge_axis,
                merge_fn=cfg.merge_fn,
            ),
            CompositionStrategy.WEIGHTED_PARALLEL: lambda: WeightedParallelStrategy(
                cast(Sequence[str], cfg.mix_fields)
            ),
            CompositionStrategy.CONDITIONAL_PARALLEL: lambda: ConditionalParallelStrategy(
                conditions=cast(Sequence[Callable], cfg.conditions),
                merge_strategy=cfg.merge_strategy,
                merge_axis=cfg.merge_axis,
                merge_fn=cfg.merge_fn,
            ),
            CompositionStrategy.ENSEMBLE_MEAN: build_ensemble,
            CompositionStrategy.ENSEMBLE_SUM: build_ensemble,
            CompositionStrategy.ENSEMBLE_MAX: build_ensemble,
            CompositionStrategy.ENSEMBLE_MIN: build_ensemble,
            CompositionStrategy.BRANCHING: lambda: BranchingStrategy(
                router=cast(Callable, cfg.router)
            ),
        }

    def compute_statistics(self, batch_data: PyTree) -> dict[str, Any] | None:
        """Return one entry per child, each computed on this composition's input.

        A composite applies no statistics of its own; it carries what its children computed, in
        child order, so each child receives its own. ``weight_key`` is stripped first, because a
        weighted composite removes it before any child runs and a child's statistics must
        describe the fields it is actually given.

        Args:
            batch_data: The batch about to be applied, with the batch on axis 0.

        Returns:
            The children's statistics, or ``None`` when no child has any.
        """
        _, clean_data = self._resolve_weighted_params(batch_data)
        return child_statistics(self._get_operators_list(), clean_data)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply composition based on the configured strategy.

        The record's key travels to the strategy, which folds each child's position into it
        so every child draws independently while still depending only on the record.

        For ``WEIGHTED_PARALLEL`` with ``weight_key``, extracts weights from
        ``data[weight_key]``, strips the key from data, and passes clean data
        to the strategy. Raises ``ValueError`` if the key is missing from data.
        """
        from datarax.operators.strategies.base import StrategyContext

        extra_params, clean_data = self._resolve_weighted_params(data)

        context = StrategyContext(
            data=clean_data,
            state=state,
            metadata=metadata if metadata is not None else {},
            key=require_key(key, self) if self.stochastic else key,
            stats=stats,
            extra_params=extra_params if extra_params else None,
        )

        return self.strategy_impl.apply(self._get_operators_list(), context)

    def _resolve_weighted_params(self, data: PyTree) -> tuple[dict[str, Any], PyTree]:
        """Resolve ``extra_params`` weights and strip ``weight_key`` from data.

        Only ``WEIGHTED_PARALLEL`` consumes weights; every other strategy returns the
        data unchanged with no extra params. For ``WEIGHTED_PARALLEL`` the weights come
        from (in priority order) a dynamic ``data[weight_key]`` entry, the softmax of the
        learnable ``weight_logits`` at ``temperature``, or the static ``config.weights``.

        Args:
            data: Input pytree passed to :meth:`apply`.

        Returns:
            Tuple of (extra_params, clean_data) where ``clean_data`` has ``weight_key``
            removed when dynamic weights were extracted from it.

        Raises:
            ValueError: If ``weight_key`` is configured but absent from ``data``.
        """
        if self.config.strategy != CompositionStrategy.WEIGHTED_PARALLEL:
            return {}, data

        if self.config.weight_key is not None:
            # Dynamic mode: extract weights from data dict.
            if not isinstance(data, dict) or self.config.weight_key not in data:
                available = list(data.keys()) if isinstance(data, dict) else "N/A"
                raise ValueError(
                    f"weight_key '{self.config.weight_key}' not found in data. "
                    f"Available keys: {available}"
                )
            # Strip weight_key so child operators don't receive it.
            clean_data = {k: v for k, v in data.items() if k != self.config.weight_key}
            return {"weights": data[self.config.weight_key]}, clean_data

        return {"weights": self.mixture_weights()}, data

    def mixture_weights(self) -> jax.Array:
        """Return the weights a ``WEIGHTED_PARALLEL`` composite applies to its operators' outputs.

        Static weights are returned as configured. Learnable weights are
        ``softmax(weight_logits / temperature)``, so the value follows training.

        Returns:
            One weight per operator.

        Raises:
            ValueError: If the strategy is not ``WEIGHTED_PARALLEL``, or if each record supplies
                the weights through ``weight_key``.
        """
        if self.config.strategy != CompositionStrategy.WEIGHTED_PARALLEL:
            raise ValueError(
                "mixture_weights applies to WEIGHTED_PARALLEL composites, "
                f"not {self.config.strategy}"
            )
        if self.config.weight_key is not None:
            raise ValueError(
                f"each record supplies the weights as data[{self.config.weight_key!r}] "
                "(weight_key), so the composite has no fixed mixture"
            )
        if self.config.learnable_weights:
            return nnx.softmax(self.weight_logits[...] / self.config.temperature)
        return jnp.asarray(self.config.weights)

    def _get_operators_list(self) -> list[OperatorModule]:
        """Get list of operators."""
        if isinstance(self.operators, nnx.Dict):
            return list(self.operators.values())
        return list(self.operators)

    # Dynamic sequential methods
    def add_operator(self, operator: OperatorModule, index: int | None = None) -> None:
        """Add operator to dynamic sequential.

        Args:
            operator: The operator to append or insert.
            index: Position to insert at, or ``None`` to append.

        Raises:
            ValueError: If the strategy is not ``DYNAMIC_SEQUENTIAL``, or the operator is
                stochastic while this composite is not.
        """
        if self.config.strategy != CompositionStrategy.DYNAMIC_SEQUENTIAL:
            raise ValueError("add_operator only available for DYNAMIC_SEQUENTIAL")

        # A composite's own mode is fixed when its config is built, and only a stochastic
        # composite is handed a key to fold for its children. Adding a stochastic child to a
        # deterministic composite would leave that child with no key, raising at apply time
        # far from the call that caused it.
        if getattr(operator.config, "stochastic", False) and not self.config.stochastic:
            raise ValueError(
                f"{type(operator).__name__} is stochastic but this composite is not, so it has "
                "no key to give it; construct the composite with a stochastic operator among "
                "its initial operators"
            )

        if index is None:
            self.operators.append(operator)
        else:
            self.operators.insert(index, operator)

    def remove_operator(self, index: int) -> OperatorModule:
        """Remove operator from dynamic sequential."""
        if self.config.strategy != CompositionStrategy.DYNAMIC_SEQUENTIAL:
            raise ValueError("remove_operator only available for DYNAMIC_SEQUENTIAL")

        # DYNAMIC_SEQUENTIAL always uses nnx.List
        operators = cast(nnx.List[OperatorModule], self.operators)
        return operators.pop(index)

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
