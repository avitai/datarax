"""OperatorModule - base class for parametric transformation modules.

This module provides OperatorModule, the base class for all parametric,
differentiable data transformations in Datarax.

Key Features:

- Config-based initialization with OperatorConfig
- Stochastic mode (each record draws from its own PRNG key)
- Deterministic mode (no randomness)
- Batch processing with vmap
- JIT compatibility with static branching
- A statistics store the operator applies to every record
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
from datarax.core.prng import per_record_keys


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


def require_key(key: jax.Array | None, operator: "OperatorModule") -> jax.Array:
    """Return the record's PRNG key, refusing a stochastic operator that was handed none.

    A stochastic operator draws everything it applies from the key its caller passes, so a
    missing key leaves nothing to draw from. Falling back to a fixed key instead would give
    every record in the batch the same draw, quietly, which is the failure this refuses.

    Args:
        key: The record's PRNG key, or ``None`` when the caller passed none.
        operator: The operator asking for the key; its class is named in the error.

    Returns:
        The key, unchanged.

    Raises:
        ValueError: If ``key`` is ``None``.
    """
    if key is None:
        raise ValueError(
            f"{type(operator).__name__} is stochastic and needs a per-record key; "
            "apply_batch, _apply_on_raw and Pipeline pass one, or call apply(..., key=...)"
        )
    return key


def _record_indices_from_metadata(metadata_list: Any) -> jax.Array | None:
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


class OperatorModule(DataraxModule):
    """Base class for parametric, differentiable operators.

    Operators work on Batch[Element] data and can have learnable parameters.
    They support both stochastic (random) and deterministic modes.

    The operator pattern keeps every transformation a pure function of its own record:
    1. apply() - Transforms a single element, drawing any randomness from that record's key
    2. apply_batch() - Derives one key per record and vmaps apply over the batch

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
        # keys are derived as fold_in(fold_in(base_key, epoch), record_index), so
        # within an epoch a record's randomness does not depend on batch
        # composition, shuffle order, worker split or resume point, and each
        # epoch draws fresh randomness. Stored as NNX state so it round-trips
        # through checkpoints.
        if config.stochastic:
            assert rngs is not None  # guaranteed by the check above
            if config.stream_name is None:
                raise ValueError("Stochastic operators require config.stream_name to be set.")
            self._base_key = nnx.Variable(rngs[config.stream_name]())

        # Fitted or fixed statistics, which every record's apply receives. A plain nnx.Variable,
        # so the statistics are module state rather than graphdef metadata: two operators with
        # equal configurations share one compiled trace whatever their statistics hold, and the
        # store round-trips through a checkpoint.
        self._statistics: nnx.Variable[dict[str, Any] | None] = nnx.Variable(None)

    # ========================================================================
    # Statistics
    # ========================================================================

    def compute_statistics(self, batch_data: PyTree) -> dict[str, Any] | None:
        """Return the statistics every record of this batch passes to ``apply``.

        The default is whatever was stored with ``set_statistics``. An operator that fits
        statistics to each batch overrides this.

        Args:
            batch_data: The batch about to be applied, with the batch on axis 0.

        Returns:
            The statistics to give ``apply``, or None when the operator has none.
        """
        del batch_data
        return self._statistics.get_value()

    def get_statistics(self) -> dict[str, Any] | None:
        """Return the stored statistics, or None when the operator has none.

        Returns:
            The statistics this operator applies, or None.
        """
        return self._statistics.get_value()

    def set_statistics(self, stats: dict[str, Any]) -> None:
        """Store the statistics this operator applies.

        Args:
            stats: The statistics to store. An operator that constrains them validates here, by
                overriding this method and calling ``super().set_statistics``.
        """
        self._statistics.set_value(stats)

    def reset_statistics(self) -> None:
        """Forget the stored statistics, so the operator applies none."""
        self._statistics.set_value(None)

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply operator to single element (no batch dimension).

        This is a PURE FUNCTION that transforms a single data element. It does not read
        ``self.rngs``: every random value it applies is drawn from ``key``, the record's own
        PRNG key, so the same record draws the same values whatever batch it arrives in.

        Subclasses MUST implement this method.

        Args:
            data: Element data PyTree (typically dict[str, Array], no batch dim)
            state: Element state PyTree (typically dict[str, Any])
            metadata: Element metadata as structured dict
            key: This record's PRNG key for a stochastic operator, ``None`` for a
                deterministic one. Pass it through ``require_key`` before drawing.
            stats: Optional statistics (from get_statistics() or passed explicitly)

        Returns:
            Tuple of (transformed_data, new_state, new_metadata)
            All return values are PyTrees matching input structure

        Raises:
            NotImplementedError: If a subclass does not override this method.

        Examples:
            Example implementation:

            ```python
            def apply(self, data, state, metadata, key=None, stats=None):
                # Draw this record's brightness from its own key
                factor = jax.random.uniform(require_key(key, self), (), minval=0.8, maxval=1.2)
                transformed = {"image": data["image"] * factor}
                return transformed, state, metadata
            ```
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply() method")

    # ========================================================================
    # Concrete Methods (implemented by base class)
    # ========================================================================

    def _vmap_apply(
        self,
        batch_data: PyTree,
        batch_states: PyTree,
        stats: dict[str, Any] | None = None,
        record_indices: jax.Array | None = None,
        epoch: jax.Array | int | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator over batch via vmap (parallel) or scan (sequential).

        Strategy is controlled by config.batch_strategy:
        - "vmap": jax.vmap — fast, O(batch_size) memory
        - "scan": jax.lax.scan — sequential, O(1) memory per element

        This is the computational heart shared by apply_batch(), _apply_on_raw(),
        and the DAG executor's fused chain.

        Randomness is keyed per record: each element's PRNG key is
        ``fold_in(fold_in(self._base_key, epoch), record_index)`` (see
        ``per_record_keys``), so within an epoch augmentation does not depend on
        batch composition, shuffle order, worker split or resume point.

        Args:
            batch_data: PyTree with arrays having batch dimension as axis 0.
            batch_states: PyTree with arrays having batch dimension as axis 0.
            stats: Optional statistics (if None, uses get_statistics()).
            record_indices: Optional int array ``(batch_size,)`` of stable record
                indices for per-record RNG. When ``None`` (no record information
                available), falls back to ``arange(batch_size)``, which is
                deterministic per batch layout but not globally unique.
            epoch: The epoch the records belong to, or ``None`` when the keys
                depend on the record index alone.

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
            batch_size = extract_batch_size(data_shapes)
            if record_indices is None:
                record_indices = jnp.arange(batch_size, dtype=jnp.uint32)
            element_keys = per_record_keys(self._base_key[...], record_indices, epoch)
        else:
            # A deterministic operator has nothing to draw, so it is handed no key.
            element_keys = None

        # === PER-ELEMENT FUNCTION + INPUTS (unified — DRY) ===
        has_keys = element_keys is not None
        if has_keys:

            def _apply_with_key(data: Any, state: Any, key: Any) -> tuple[Any, Any]:
                out_data, out_state, _ = self.apply(data, state, None, key, _stats)
                return out_data, out_state

            apply_one = _apply_with_key
            inputs = (batch_data, batch_states, element_keys)
        else:

            def _apply_no_key(data: Any, state: Any) -> tuple[Any, Any]:
                out_data, out_state, _ = self.apply(data, state, None, None, _stats)
                return out_data, out_state

            apply_one = _apply_no_key
            inputs = (batch_data, batch_states)

        # === SCAN BRANCH (sequential, O(1) memory per element) ===
        if self.config.batch_strategy == "scan":
            _, result = jax.lax.scan(lambda carry, x: (carry, apply_one(*x)), None, inputs)
            return result

        # === VMAP BRANCH (parallel) ===
        # Every leaf apply returns carries the batch on axis 0, whatever fields it adds, so the
        # integer 0 is a tree prefix of any output and no structure has to be discovered first.
        in_data_axes = jax.tree.map(lambda _: 0, batch_data)
        in_state_axes = jax.tree.map(lambda _: 0, batch_states)
        in_axes = (in_data_axes, in_state_axes, 0) if has_keys else (in_data_axes, in_state_axes)

        return jax.vmap(apply_one, in_axes=in_axes, out_axes=0)(*inputs)

    def _apply_on_raw(
        self,
        batch_data: PyTree,
        batch_states: PyTree,
        stats: dict[str, Any] | None = None,
        record_indices: jax.Array | None = None,
        epoch: jax.Array | int | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator on raw dicts without Batch object creation.

        Thin wrapper around _vmap_apply for use in the fused operator chain.
        Returns raw (data_dict, states_dict) instead of a Batch object,
        enabling chaining without intermediate Batch construction.

        Args:
            batch_data: Dict of batched arrays (axis 0 is batch).
            batch_states: Dict of batched state arrays.
            stats: Optional statistics.
            record_indices: Optional ``(batch_size,)`` stable record indices for
                per-record RNG (see ``_vmap_apply``). The Pipeline threads the
                indices its source names for the batch.
            epoch: The epoch the records belong to (see ``_vmap_apply``).

        Returns:
            Tuple of (transformed_data, transformed_states) as raw dicts.
        """
        return self._vmap_apply(batch_data, batch_states, stats, record_indices, epoch)

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
        record_indices = _record_indices_from_metadata(batch_metadata.get_value())

        # Delegate to shared vmap core
        transformed_data, transformed_states = self._vmap_apply(
            batch_data, batch_states, stats, record_indices
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
