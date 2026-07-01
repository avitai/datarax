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

import logging
from typing import Any, final

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.metadata import Metadata
from datarax.core.module import DataraxModule


logger = logging.getLogger(__name__)


def extract_batch_size(data_shapes: PyTree) -> int:
    """Extract batch size from a PyTree of shape tuples.

    Traverses the PyTree treating tuples as atomic leaves (since JAX
    normally unfolds tuples as nodes) and returns the first axis of
    the first leaf shape.

    Args:
        data_shapes: PyTree with same structure as batch data, where each
                     leaf is a shape tuple (e.g. ``(batch_size, H, W, C)``).

    Returns:
        The batch size (first element of the first shape found).

    Raises:
        ValueError: If the shape tree has no leaves.
    """
    batch_sizes = jax.tree.map(
        lambda shape: shape[0], data_shapes, is_leaf=lambda x: isinstance(x, tuple)
    )
    batch_size_leaves = jax.tree.leaves(batch_sizes)

    if not batch_size_leaves:
        raise ValueError("Cannot extract batch size from an empty shape tree")

    return batch_size_leaves[0]


# Module-level cache for output structure discovery.
# Using module-level cache instead of instance attribute avoids NNX pytree tracking,
# which is critical for nnx.cond/switch where branches must have identical pytree structure.
# Key: (unique_operator_id, PyTreeDef of input)
# Value: (out_data_struct, out_state_struct)
_OUTPUT_STRUCT_CACHE: dict[tuple[int, Any], tuple[PyTree, PyTree]] = {}

# Upper bound on cache entries. Each unique (operator, input-structure) pair adds
# one entry; without a bound a long-running process that builds many operators or
# feeds many distinct input structures would grow this dict without limit. When
# full, the oldest entry is evicted (FIFO) — dicts preserve insertion order.
_OUTPUT_STRUCT_CACHE_MAXSIZE = 1024


def _global_indices_from_metadata(metadata_list: Any) -> jax.Array | None:
    """Extract per-record global indices from a batch's metadata list.

    Returns a ``(batch_size,)`` array of ``Metadata.index`` values, or ``None``
    when the metadata is absent or not a list of ``Metadata`` (in which case the
    caller falls back to a positional ``arange``). Requiring genuine ``Metadata``
    instances avoids mistaking unrelated objects for records — e.g. a plain
    ``str`` has a built-in ``.index`` *method*, which must not be treated as a
    record index. Safe under tracing: ``Metadata.index`` may be a JAX tracer.
    """
    if not metadata_list or not all(isinstance(meta, Metadata) for meta in metadata_list):
        return None
    return jnp.asarray([meta.index for meta in metadata_list], dtype=jnp.uint32)


def _store_output_struct(
    cache_key: tuple[int, Any],
    value: tuple[PyTree, PyTree],
) -> None:
    """Insert into the output-structure cache with bounded FIFO eviction."""
    if cache_key not in _OUTPUT_STRUCT_CACHE and (
        len(_OUTPUT_STRUCT_CACHE) >= _OUTPUT_STRUCT_CACHE_MAXSIZE
    ):
        # Evict the oldest inserted entry to keep the cache bounded.
        oldest_key = next(iter(_OUTPUT_STRUCT_CACHE))
        del _OUTPUT_STRUCT_CACHE[oldest_key]
    _OUTPUT_STRUCT_CACHE[cache_key] = value


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

    config: OperatorConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
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

        # Stable per-operator base key, drawn ONCE (not per batch). Per-record
        # keys are derived as fold_in(base_key, global_record_index), so a
        # record's randomness depends only on (base_key, its global index) —
        # invariant to batch composition, shuffle order, host count, and resume.
        # Stored as NNX state so it round-trips through checkpoints.
        if config.stochastic:
            assert rngs is not None  # guaranteed by the check above
            if config.stream_name is None:
                raise ValueError("Stochastic operators require config.stream_name to be set.")
            self._base_key = nnx.Variable(rngs[config.stream_name]())

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    def generate_random_params(
        self,
        element_keys: jax.Array | None,
        data_shapes: PyTree,
    ) -> PyTree:
        """Generate per-record random parameters from per-record PRNG keys.

        ``element_keys`` is a ``(batch_size, ...)`` array of stateless keys, one
        per record, each derived as ``fold_in(base_key, global_index)`` (see
        ``_vmap_apply``). Implementations should produce one parameter per record
        by mapping over the keys, e.g.::

            factors = jax.vmap(lambda k: jax.random.uniform(k, ()))(element_keys)
            return {"factor": factors}

        Keying on the per-record key (not a single per-batch draw) is what makes
        augmentation reproducible across batch composition, shuffle order, host
        count, and resume. Required for stochastic operators; deterministic
        operators leave the default (``element_keys`` is ``None``).

        Args:
            element_keys: ``(batch_size, ...)`` array of per-record PRNG keys, or
                ``None`` for deterministic operators.
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image": (batch_size, H, W, C)}

        Returns:
            PyTree of per-record random parameters (leading dim ``batch_size``),
            or ``None`` for deterministic operators.

        Raises:
            NotImplementedError: If stochastic=True but not implemented
        """
        del element_keys, data_shapes
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
        global_indices: jax.Array | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator over batch via vmap (parallel) or scan (sequential).

        Strategy is controlled by config.batch_strategy:
        - "vmap": jax.vmap — fast, O(batch_size) memory
        - "scan": jax.lax.scan — sequential, O(1) memory per element

        This is the computational heart shared by apply_batch(), _apply_on_raw(),
        and the DAG executor's fused chain.

        Randomness is keyed per record: each element's PRNG key is
        ``fold_in(self._base_key, global_index)`` (see ``generate_random_params``),
        so augmentation is invariant to batch composition, shuffle order, host
        count, and resume point.

        Args:
            batch_data: PyTree with arrays having batch dimension as axis 0.
            batch_states: PyTree with arrays having batch dimension as axis 0.
            stats: Optional statistics (if None, uses get_statistics()).
            global_indices: Optional int array ``(batch_size,)`` of stable global
                record indices for per-record RNG. When ``None`` (no positional
                information available), falls back to ``arange(batch_size)``,
                which is deterministic per batch layout but not globally unique.

        Returns:
            Tuple of (transformed_data, transformed_states) as raw PyTrees.
        """
        if stats is None:
            stats = self.get_statistics()
        _stats = stats

        data_shapes = jax.tree.map(lambda x: x.shape, batch_data)

        # === PER-RECORD RNG KEYS ===
        # Derive one stateless key per record from the operator's stable base key
        # and the record's global index — never from a per-batch stream draw.
        if self.stochastic:
            # Local import: importing datarax.utils.prng at module load would
            # cycle (utils/__init__ -> external -> core.operator). Python caches
            # the module, so this is a cheap dict lookup at trace time.
            from datarax.utils.prng import per_record_keys  # noqa: PLC0415

            batch_size = extract_batch_size(data_shapes)
            if global_indices is None:
                global_indices = jnp.arange(batch_size, dtype=jnp.uint32)
            element_keys = per_record_keys(self._base_key[...], global_indices)
            random_params_batch = self.generate_random_params(element_keys, data_shapes)
        else:
            # Deterministic operators receive no random parameters.
            random_params_batch = None

        # === PER-ELEMENT FUNCTION + INPUTS (unified — DRY) ===
        has_rp = random_params_batch is not None
        if has_rp:

            def _apply_with_rp(data: Any, state: Any, rp: Any) -> tuple[Any, Any]:
                out_data, out_state, _ = self.apply(data, state, None, rp, _stats)
                return out_data, out_state

            apply_one = _apply_with_rp
            inputs = (batch_data, batch_states, random_params_batch)
        else:

            def _apply_no_rp(data: Any, state: Any) -> tuple[Any, Any]:
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
            _store_output_struct(cache_key, self.get_output_structure(sample_data, sample_state))
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
        global_indices: jax.Array | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator on raw dicts without Batch object creation.

        Thin wrapper around _vmap_apply for use in the fused operator chain.
        Returns raw (data_dict, states_dict) instead of a Batch object,
        enabling chaining without intermediate Batch construction.

        Args:
            batch_data: Dict of batched arrays (axis 0 is batch).
            batch_states: Dict of batched state arrays.
            stats: Optional statistics.
            global_indices: Optional ``(batch_size,)`` global record indices for
                per-record RNG (see ``_vmap_apply``). The Pipeline threads the
                batch start position here so augmentation is position-keyed.

        Returns:
            Tuple of (transformed_data, transformed_states) as raw dicts.
        """
        return self._vmap_apply(batch_data, batch_states, stats, global_indices)

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

        # Per-record RNG: use the batch's stable global record indices when the
        # metadata carries them; otherwise _vmap_apply falls back to arange.
        global_indices = _global_indices_from_metadata(batch_metadata.get_value())

        # Delegate to shared vmap core
        transformed_data, transformed_states = self._vmap_apply(
            batch_data, batch_states, stats, global_indices
        )

        # Reconstruct batch (preserves batch-level data, including valid_mask).
        # Without explicit valid_mask propagation, a partial last batch's
        # padding mask would reset to all-True here and silently corrupt
        # mask-weighted loss downstream.
        return Batch.from_parts(
            data=transformed_data,
            states=transformed_states,
            metadata_list=batch_metadata.get_value(),
            batch_metadata=batch._batch_metadata.get_value(),
            batch_state=batch.batch_state.get_value(),
            validate=False,
            valid_mask=batch.valid_mask[...],
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

    def output_spec(self, input_spec: PyTree) -> PyTree:
        """Return the operator's output spec given an input spec.

        Most operators (normalization, additive noise, simple element-wise
        transforms) do not change shape; the default returns ``input_spec``
        unchanged. Shape-changing operators (Resize, Crop, Reshape) MUST
        override this method.

        Args:
            input_spec: PyTree of ``jax.ShapeDtypeStruct`` describing the input
                element (matching the upstream ``DataSourceModule.element_spec()``
                or another operator's ``output_spec``).

        Returns:
            PyTree of ``jax.ShapeDtypeStruct`` describing the operator's output.
            By default, equal to ``input_spec``.
        """
        return input_spec
