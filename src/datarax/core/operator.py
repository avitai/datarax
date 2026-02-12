"""OperatorModule - base class for parametric transformation modules.

This module provides OperatorModule, the base class for all parametric,
differentiable data transformations in Datarax.

Key Features:

- Config-based initialization with OperatorConfig
- Stochastic mode (with random parameter generation)
- Deterministic mode (no randomness)
- Batch processing with vmap
- JIT compatibility with static branching
- Statistics system (inherited from DataraxModule)
"""

from typing import Any, final

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.module import DataraxModule

# Module-level cache for output structure discovery.
# Using module-level cache instead of instance attribute avoids NNX pytree tracking,
# which is critical for nnx.cond/switch where branches must have identical pytree structure.
# Key: (unique_operator_id, PyTreeDef of input)
# Value: (out_data_struct, out_state_struct)
_OUTPUT_STRUCT_CACHE: dict[tuple[int, Any], tuple[PyTree, PyTree]] = {}

# Monotonically increasing ID counter for unique operator identification.
# Using id(self) is unsafe because Python reuses memory addresses after GC.
# This counter ensures each operator instance gets a unique, permanent ID.
_OPERATOR_ID_COUNTER = 0


class OperatorModule(DataraxModule):
    """Base class for parametric, differentiable operators.

    Operators work on Batch[Element] data and can have learnable parameters.
    They support both stochastic (random) and deterministic modes.

    The operator pattern separates RNG generation from transformation:
    1. generate_random_params() - Generates batch-level random parameters (impure)
    2. apply() - Applies transformation to single element (pure function)
    3. apply_batch() - Orchestrates batch processing with vmap (concrete implementation)

    Args:
        config: OperatorConfig (already validated via __post_init__)
        rngs: Random number generators (required if stochastic=True)
        name: Optional name for the operator

    Attributes:
        config: Operator configuration
        stochastic: Whether this operator uses randomness (from config)
        stream_name: RNG stream name (from config, required if stochastic=True)
    """

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize OperatorModule with config.

        Args:
            config: Operator configuration (already validated)
            rngs: Random number generators (required if stochastic=True)
            name: Optional operator name

        Raises:
            ValueError: If stochastic=True but rngs is None
        """
        super().__init__(config, rngs=rngs, name=name)

        # Assign unique ID for cache keying (avoids id() reuse after GC)
        global _OPERATOR_ID_COUNTER
        _OPERATOR_ID_COUNTER += 1
        self._unique_id = _OPERATOR_ID_COUNTER

        # Runtime validation: Stochastic operators require rngs
        if config.stochastic and rngs is None:
            raise ValueError(
                f"Stochastic operators require rngs parameter. "
                f"Pass rngs=nnx.Rngs(..., {config.stream_name}=...)"
            )

        # Convenience properties (avoid repeated config access)
        # Mark as static using nnx.static() - these are config values, not state
        # Without nnx.static(), Orbax checkpointing fails on non-JAX types
        self.stochastic = nnx.static(config.stochastic)
        self.stream_name = nnx.static(config.stream_name)

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> PyTree:
        """Generate random parameters for batch transformation.

        This method generates batch-level random parameters (one per batch element).
        It is impure (uses RNG) and should be called once per batch.

        Required for stochastic operators. Deterministic operators can leave default.

        Args:
            rng: JAX random key
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image": (batch_size, H, W, C)}

        Returns:
            PyTree of random parameters matching batch structure.
            Can be any structure (scalars, arrays, dicts, etc.)

        Raises:
            NotImplementedError: If stochastic=True but not implemented
        """
        if self.stochastic:
            raise NotImplementedError(
                f"{self.__class__.__name__} is stochastic but does not implement "
                "generate_random_params(). Stochastic operators must generate "
                "random parameters for batch processing."
            )
        return None

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply operator to single element (no batch dimension).

        This is a PURE FUNCTION that transforms a single data element.
        It should not access self.rngs or generate random numbers.
        All randomness comes through random_params argument.

        Subclasses MUST implement this method.

        Args:
            data: Element data PyTree (typically dict[str, Array], no batch dim)
            state: Element state PyTree (typically dict[str, Any])
            metadata: Element metadata as structured dict
            random_params: Random parameters for this element (from generate_random_params)
            stats: Optional statistics (from get_statistics() or passed explicitly)

        Returns:
            Tuple of (transformed_data, new_state, new_metadata)
            All return values are PyTrees matching input structure

        Examples:
            Example implementation:

            ```python
            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Apply random brightness
                factor = random_params if random_params is not None else 1.0
                transformed = {"image": data["image"] * factor}
                return transformed, state, metadata
            ```
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply() method")

    def get_output_structure(
        self,
        sample_data: PyTree,
        sample_state: PyTree,
    ) -> tuple[PyTree, PyTree]:
        """Declare output PyTree structure for vmap axis specification.

        Default uses jax.eval_shape to discover structure automatically.
        Override for efficiency or when eval_shape doesn't work (e.g., data-dependent shapes).

        Args:
            sample_data: Single element data (not batched)
            sample_state: Single element state (not batched)

        Returns:
            Tuple of (output_data_structure, output_state_structure) with None leaves.
            The structure (keys/nesting) matters, leaf values are ignored.

        Example override for operator that adds keys:
            def get_output_structure(self, sample_data, sample_state):
                out_data = {
                    **jax.tree.map(lambda _: None, sample_data),
                    "score": None,
                    "alignment": None,
                }
                return out_data, sample_state
        """

        # Default: use eval_shape to discover output structure without computation
        def apply_wrapper(data: PyTree, state: PyTree) -> tuple[PyTree, PyTree]:
            out_data, out_state, _ = self.apply(data, state, None)
            return out_data, out_state

        out_shapes = jax.eval_shape(apply_wrapper, sample_data, sample_state)
        # Use 0 as placeholder (not None!) because None is an empty pytree in JAX
        # and jax.tree.map won't transform it. Using 0 makes these directly usable
        # as vmap axis specifications.
        out_data_struct = jax.tree.map(lambda _: 0, out_shapes[0])
        out_state_struct = jax.tree.map(lambda _: 0, out_shapes[1])
        return out_data_struct, out_state_struct

    # ========================================================================
    # Concrete Methods (implemented by base class)
    # ========================================================================

    def _vmap_apply(
        self,
        batch_data: PyTree,
        batch_states: PyTree,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator over batch via vmap (parallel) or scan (sequential).

        Strategy is controlled by config.batch_strategy:
        - "vmap": jax.vmap — fast, O(batch_size) memory
        - "scan": jax.lax.scan — sequential, O(1) memory per element

        This is the computational heart shared by apply_batch(), _apply_on_raw(),
        and the DAG executor's fused chain.

        Args:
            batch_data: PyTree with arrays having batch dimension as axis 0.
            batch_states: PyTree with arrays having batch dimension as axis 0.
            stats: Optional statistics (if None, uses get_statistics()).

        Returns:
            Tuple of (transformed_data, transformed_states) as raw PyTrees.
        """
        if stats is None:
            stats = self.get_statistics()
        _stats = stats

        # === RNG + RANDOM PARAMS ===
        if self.stochastic:
            assert self.stream_name is not None, "stochastic=True requires stream_name"
            assert self.rngs is not None, "stochastic=True requires rngs"
            stream_name: str = self.stream_name
            rngs: nnx.Rngs = self.rngs
            rng = rngs[stream_name]()
        else:
            rng = jax.random.key(0)

        data_shapes = jax.tree.map(lambda x: x.shape, batch_data)
        random_params_batch = self.generate_random_params(rng, data_shapes)

        # === PER-ELEMENT FUNCTION + INPUTS (unified — DRY) ===
        has_rp = random_params_batch is not None
        if has_rp:

            def _apply_with_rp(data, state, rp):
                out_data, out_state, _ = self.apply(data, state, None, rp, _stats)
                return out_data, out_state

            apply_one = _apply_with_rp
            inputs = (batch_data, batch_states, random_params_batch)
        else:

            def _apply_no_rp(data, state):
                out_data, out_state, _ = self.apply(data, state, None, None, _stats)
                return out_data, out_state

            apply_one = _apply_no_rp
            inputs = (batch_data, batch_states)

        # === SCAN BRANCH (sequential, O(1) memory per element) ===
        if self.config.batch_strategy == "scan":
            _, result = jax.lax.scan(lambda carry, x: (carry, apply_one(*x)), None, inputs)
            return result

        # === VMAP BRANCH (parallel, needs output structure for axis specs) ===
        input_struct_key = jax.tree.structure(batch_data)
        cache_key = (self._unique_id, input_struct_key)
        if cache_key not in _OUTPUT_STRUCT_CACHE:
            sample_data = jax.tree.map(lambda x: x[0], batch_data)
            sample_state = jax.tree.map(lambda x: x[0], batch_states)
            _OUTPUT_STRUCT_CACHE[cache_key] = self.get_output_structure(sample_data, sample_state)
        out_data_axes, out_state_axes = _OUTPUT_STRUCT_CACHE[cache_key]

        in_data_axes = jax.tree.map(lambda _: 0, batch_data)
        in_state_axes = jax.tree.map(lambda _: 0, batch_states)
        if has_rp:
            in_axes = (in_data_axes, in_state_axes, 0)
        else:
            in_axes = (in_data_axes, in_state_axes)

        return jax.vmap(
            apply_one,
            in_axes=in_axes,
            out_axes=(out_data_axes, out_state_axes),
        )(*inputs)

    def _apply_on_raw(
        self,
        batch_data: PyTree,
        batch_states: PyTree,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator on raw dicts without Batch object creation.

        Thin wrapper around _vmap_apply for use in the fused operator chain.
        Returns raw (data_dict, states_dict) instead of a Batch object,
        enabling chaining without intermediate Batch construction.

        Args:
            batch_data: Dict of batched arrays (axis 0 is batch).
            batch_states: Dict of batched state arrays.
            stats: Optional statistics.

        Returns:
            Tuple of (transformed_data, transformed_states) as raw dicts.
        """
        return self._vmap_apply(batch_data, batch_states, stats)

    def apply_batch(
        self,
        batch: Batch,
        stats: dict[str, Any] | None = None,
    ) -> Batch:
        """Process entire batch with vmap and optional RNG generation.

        This method implements the batch processing logic for both stochastic
        and deterministic modes. It uses static branching on self.stochastic
        for JIT compilation efficiency.

        The implementation delegates to _vmap_apply() for the shared
        computational core, then wraps the result in a Batch object.

        Args:
            batch: Input batch (Batch[Element] structure)
            stats: Optional statistics (if None, uses get_statistics())

        Returns:
            Transformed batch with same structure

        Note:
            This method is concrete (not abstract). Subclasses typically don't
            override it, but can if they need custom batch processing logic.
        """
        # Extract batch components for vmap processing
        batch_data = batch.data.get_value()
        batch_states = batch.states.get_value()
        batch_metadata = batch._metadata_list

        # Check for empty PyTree edge case (vmap requires at least one array)
        has_data_arrays = len(jax.tree.leaves(batch_data)) > 0
        has_state_arrays = len(jax.tree.leaves(batch_states)) > 0

        if not has_data_arrays and not has_state_arrays:
            return batch

        if batch.batch_size == 0:
            return batch

        # Delegate to shared vmap core
        transformed_data, transformed_states = self._vmap_apply(batch_data, batch_states, stats)

        # Reconstruct batch (preserves batch-level data)
        return Batch.from_parts(
            data=transformed_data,
            states=transformed_states,
            metadata_list=batch_metadata,
            batch_metadata=batch._batch_metadata,
            batch_state=batch.batch_state.get_value(),
            validate=False,
        )

    @final
    def __call__(self, batch: Batch) -> Batch:
        """Main entry point for operator application.

        This method handles caching, statistics, and iteration tracking,
        then delegates to apply_batch().

        Args:
            batch: Input batch

        Returns:
            Transformed batch
        """
        # TODO: Add caching logic if config.cacheable

        # Delegate to apply_batch
        return self.apply_batch(batch)
