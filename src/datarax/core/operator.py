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
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, final

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
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


def call_with_mode_key[In, Out](
    fn: Callable[[In, jax.Array], Out] | Callable[[In, None], Out],
    value: In,
    key: jax.Array | None,
) -> Out:
    """Call a user function in the shape the operator's mode gives it.

    A stochastic operator's function takes the record's key and a deterministic one's takes
    ``None``; which one ``fn`` is follows the operator's configuration, which the type system
    cannot see, so the call narrows on the key it was handed.

    Args:
        fn: The user function, keyed or keyless.
        value: What the function transforms.
        key: The record's key, or ``None`` for a deterministic operator.

    Returns:
        The function's result.
    """
    if key is None:
        return cast(Callable[[In, None], Out], fn)(value, None)
    return cast(Callable[[In, jax.Array], Out], fn)(value, key)


def _statistics_arrays(statistics: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
    """Return ``statistics`` with every leaf an array, so no Python value becomes state."""
    return jax.tree.map(jnp.asarray, statistics)


# The name a wrapper's statistics carry their children's entries under. A wrapper applies no
# statistics of its own: what it holds is what each child computed on the wrapper's input, in
# child order.
CHILD_STATISTICS = "children"


def child_statistics(
    operators: Sequence["OperatorModule"], batch_data: PyTree
) -> dict[str, Any] | None:
    """Return what each child computes for this batch, or ``None`` when no child has any.

    A child cannot compute statistics of its own while it runs: a wrapper applies its children
    inside one vectorized call, by which point the batch is gone. So the wrapper computes them
    once per batch, before the batch is vectorized.

    Args:
        operators: The wrapper's children, in the order it applies them.
        batch_data: The batch the wrapper is about to apply, with the batch on axis 0.

    Returns:
        One entry per child, or ``None`` when every child computed ``None`` — so a wrapper
        over children with no statistics passes nothing down rather than an empty shell.
    """
    computed = tuple(operator.compute_statistics(batch_data) for operator in operators)
    if all(entry is None for entry in computed):
        return None
    return {CHILD_STATISTICS: computed}


def statistics_for_child(stats: dict[str, Any] | None, index: int) -> dict[str, Any] | None:
    """Return the entry a wrapper computed for the child at ``index``.

    Args:
        stats: The wrapper's statistics, as ``child_statistics`` built them.
        index: The child's position in the wrapper.

    Returns:
        That child's statistics, or ``None`` when the wrapper carries none.
    """
    if stats is None:
        return None
    children = stats.get(CHILD_STATISTICS)
    if children is None:
        return None
    return children[index]


def _raw_identity(
    size: int, record_indices: jax.Array | None, epoch: jax.Array | int | None
) -> tuple[jax.Array | None, jax.Array | None]:
    """The two-word indices and per-record epochs of the raw-dict path's ``(B,)`` identity.

    The pipeline's raw path names records by int32 positions below ``2^31`` and one epoch for
    the batch or one per record; each becomes the low index word and a per-record epoch.
    """
    indices = (
        None
        if record_indices is None
        else jnp.stack(
            [jnp.zeros(size, jnp.uint32), jnp.asarray(record_indices).astype(jnp.uint32)], -1
        )
    )
    epochs = None if epoch is None else jnp.broadcast_to(jnp.asarray(epoch, jnp.int32), (size,))
    return indices, epochs


class OperatorModule(DataraxModule):
    """Base class for parametric, differentiable operators.

    Operators transform batches of records and can have learnable parameters.
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
        statistics: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        """Initialize OperatorModule with config.

        Args:
            config: Operator configuration (already validated)
            rngs: Random number generators (required if stochastic=True)
            name: Optional operator name
            statistics: Fitted or fixed statistics every record's ``apply`` receives, stored as
                arrays at their real shape. ``set_statistics`` may later replace their values,
                never their layout. An operator built without them holds none.

        Raises:
            ValueError: If stochastic=True but rngs is None
        """
        super().__init__(config, rngs=rngs, name=name)

        # The caller's Rngs is read once, below, to draw this operator's base key. It is not kept:
        # an operator's randomness afterwards comes from its own base key alone, so it never
        # reaches back into state the caller owns. Assigned through nnx.data so a subclass
        # may still store an Rngs here after super().__init__.
        self.rngs = nnx.data(None)

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
        # Flax's mode flag: ``eval()`` sets it, ``train()`` clears it, and ``nnx.view(module,
        # deterministic=True)`` sets it on a view through ``set_view``. A stochastic operator in
        # deterministic mode draws nothing and applies ``apply_deterministic``. A plain attribute,
        # as in ``nnx.Dropout``, so training and evaluation each compile one trace.
        self.deterministic = False

        # Stable per-operator base key, drawn ONCE (not per batch). A record's key folds in its
        # epoch, draw and index (``per_record_keys``), so a record's randomness does not depend on
        # batch composition, shuffle order, worker split or resume point, and each epoch and each
        # draw is fresh. It is array state typed nnx.RngKey, never
        # a static value: a static key would sit in the graphdef, and two operators differing
        # only in their seed would compile twice. Nothing else of the operator's randomness is
        # state, so applying an operator mutates nothing.
        if config.stochastic:
            assert rngs is not None  # guaranteed by the check above
            if config.stream_name is None:
                raise ValueError("Stochastic operators require config.stream_name to be set.")
            self._base_key = nnx.RngKey(rngs[config.stream_name]())

        # Fitted or fixed statistics, which every record's apply receives. A plain nnx.Variable of
        # arrays created here at its real shape, so the operator's state layout never changes after
        # construction: the statistics are module state rather than graphdef metadata, two operators
        # with equal configurations share one compiled trace whatever their statistics hold, and
        # the store round-trips through a checkpoint.
        if statistics is not None:
            self._statistics = nnx.Variable(_statistics_arrays(statistics))

    # ========================================================================
    # Mode
    # ========================================================================

    def set_view(self, deterministic: bool | None = None) -> None:
        """Set the mode for ``nnx.view``, as flax's own stochastic layers do.

        Args:
            deterministic: ``True`` turns this operator's randomness off, ``False`` on;
                ``None`` leaves it as it is.
        """
        if deterministic is not None:
            self.deterministic = deterministic

    def apply_deterministic(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Transform one record in deterministic mode: the record, unchanged, by default.

        A stochastic operator is an augmentation unless it says otherwise, and evaluation
        applies no augmentation. An operator whose deterministic form does something (a
        composition running its deterministic children) overrides this.

        Args:
            data: Element data PyTree (no batch dimension).
            state: Element state PyTree.
            metadata: Element metadata.
            stats: This batch's statistics.

        Returns:
            Tuple of (data, state, metadata).
        """
        del stats
        return data, state, metadata

    @final
    def apply_record(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Transform one record in this operator's mode; every framework path calls this.

        A stochastic operator in deterministic mode applies ``apply_deterministic``; otherwise
        ``apply`` runs, with the record's key for a stochastic operator and no key for a
        deterministic one, whatever its parent was handed. Wrappers call their children through
        this, so a child's mode holds wherever it sits.

        Args:
            data: Element data PyTree (no batch dimension).
            state: Element state PyTree.
            metadata: Element metadata.
            key: This record's PRNG key, or ``None`` for a deterministic operator.
            stats: This batch's statistics.

        Returns:
            Tuple of (transformed_data, new_state, new_metadata).
        """
        if not self.stochastic:
            return self.apply(data, state, metadata, None, stats)
        if self.deterministic:
            return self.apply_deterministic(data, state, metadata, stats)
        return self.apply(data, state, metadata, key, stats)

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
        return self.get_statistics()

    def get_statistics(self) -> dict[str, jax.Array] | None:
        """Return the stored statistics, or None when the operator was built without them.

        Returns:
            The statistics this operator applies, or None.
        """
        store = getattr(self, "_statistics", None)
        return None if store is None else store.get_value()

    def set_statistics(self, stats: Mapping[str, ArrayLike]) -> None:
        """Replace the stored statistics' values; their layout is fixed at construction.

        Args:
            stats: The new values, with the stored tree structure and each leaf's shape and
                dtype. An operator that constrains them validates here, by overriding this method
                and calling ``super().set_statistics``.

        Raises:
            ValueError: If the operator was built without statistics, or ``stats`` differs from
                the stored ones in structure, shape or dtype. Nothing is changed.
        """
        current = self.get_statistics()
        if current is None:
            raise ValueError(
                f"{type(self).__name__} was built without statistics; pass statistics= to its "
                "constructor to give it a store of fixed layout"
            )
        replacement = _statistics_arrays(stats)
        if jax.tree.structure(replacement) != jax.tree.structure(current):
            raise ValueError(
                f"{type(self).__name__} statistics have the structure "
                f"{jax.tree.structure(current)}; got {jax.tree.structure(replacement)}"
            )
        mismatched = [
            jax.tree_util.keystr(path)
            for (path, new), old in zip(
                jax.tree_util.tree_leaves_with_path(replacement),
                jax.tree.leaves(current),
                strict=True,
            )
            if new.shape != old.shape or new.dtype != old.dtype
        ]
        if mismatched:
            raise ValueError(
                f"{type(self).__name__} statistics keep their shapes and dtypes; "
                f"{', '.join(mismatched)} differ"
            )
        self._statistics.set_value(replacement)

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
            stats: This batch's statistics (from compute_statistics() or passed explicitly)

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
        indices: ArrayLike | None = None,
        epochs: ArrayLike | None = None,
        draws: ArrayLike | None = None,
    ) -> tuple[PyTree, PyTree]:
        """Apply operator over batch via vmap (parallel) or scan (sequential).

        Strategy is controlled by config.batch_strategy:
        - "vmap": jax.vmap — fast, O(batch_size) memory
        - "scan": jax.lax.scan — sequential, O(1) memory per element

        This is the computational heart shared by apply_batch(), _apply_on_raw(),
        and the DAG executor's fused chain.

        Randomness is keyed per record: each element's PRNG key folds the record's epoch,
        draw and two-word index into the operator's base key (see ``per_record_keys``), so
        augmentation does not depend on batch composition, shuffle order, worker split or
        resume point.

        Args:
            batch_data: PyTree with arrays having batch dimension as axis 0.
            batch_states: PyTree with arrays having batch dimension as axis 0.
            stats: The statistics to give every record. When ``None`` the operator computes
                them for this batch with ``compute_statistics``, which is what lets an
                operator fit statistics to each batch it is given.
            indices: uint32 ``(batch_size, 2)``, each record's index as ``(hi, lo)``. When
                ``None`` (no record information), row ``i`` is record ``(0, i)``, as in
                ``batch_ops.from_arrays``: the call repeats exactly but a record's draw follows
                its position.
            epochs: int32 ``(batch_size,)``, each record's epoch; zeros when ``None``.
            draws: int32 ``(batch_size,)``, each record's draw; zeros when ``None``.

        Returns:
            Tuple of (transformed_data, transformed_states) as raw PyTrees.
        """
        if stats is None:
            stats = self.compute_statistics(batch_data)
        _stats = stats

        data_shapes = jax.tree.map(lambda x: x.shape, batch_data)

        # === PER-RECORD RNG KEYS ===
        # Derive one stateless key per record from the operator's stable base key
        # and the record's global index — never from a per-batch stream draw.
        if self.stochastic and not self.deterministic:
            # The keys are a function of the base key and the records alone, so a batch applied
            # twice draws the same values. A call that names no records keys on positions.
            size = extract_batch_size(data_shapes)
            if indices is None:
                rows = jnp.arange(size, dtype=jnp.uint32)
                indices = jnp.stack([jnp.zeros_like(rows), rows], -1)
            element_keys = per_record_keys(
                self._base_key[...],
                indices,
                jnp.zeros(size, jnp.int32) if epochs is None else epochs,
                jnp.zeros(size, jnp.int32) if draws is None else draws,
            )
        else:
            # A deterministic operator, or a stochastic one in deterministic mode, draws nothing,
            # so it is handed no key.
            element_keys = None

        # === PER-ELEMENT FUNCTION + INPUTS (unified — DRY) ===
        has_keys = element_keys is not None
        if has_keys:

            def _apply_with_key(data: Any, state: Any, key: Any) -> tuple[Any, Any]:
                out_data, out_state, _ = self.apply_record(data, state, None, key, _stats)
                return out_data, out_state

            apply_one = _apply_with_key
            inputs = (batch_data, batch_states, element_keys)
        else:

            def _apply_no_key(data: Any, state: Any) -> tuple[Any, Any]:
                out_data, out_state, _ = self.apply_record(data, state, None, None, _stats)
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
            record_indices: Optional ``(batch_size,)`` record indices below ``2^31``, as the
                Pipeline's raw path names them; each is the low word of the record's index.
                ``None`` keys on positions (see ``_vmap_apply``).
            epoch: The epoch of the batch, or of each record ``(batch_size,)``; ``None`` is 0.

        Returns:
            Tuple of (transformed_data, transformed_states) as raw dicts.
        """
        size = extract_batch_size(jax.tree.map(lambda x: x.shape, batch_data))
        indices, epochs = _raw_identity(size, record_indices, epoch)
        return self._vmap_apply(batch_data, batch_states, stats, indices, epochs)

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
            batch: The batch; its identities key each record's randomness.
            stats: Optional statistics (if None, uses compute_statistics() on this batch)

        Returns:
            Transformed batch with same structure

        Note:
            This method is concrete (not abstract). Subclasses typically don't
            override it, but can if they need custom batch processing logic.
        """
        # vmap needs at least one array, and a batch of no rows has nothing to apply.
        if batch.batch_size == 0 or not jax.tree.leaves((batch.data, batch.states)):
            return batch

        data, states = self._vmap_apply(
            batch.data, batch.states, stats, batch.indices, batch.epochs, batch.draws
        )
        return batch.replace(data=data, states=states)

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
