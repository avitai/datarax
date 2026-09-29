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
from flax.errors import TraceContextError
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import ArrayValue, Batch, Element
from datarax.core.module import DataraxModule
from datarax.core.prng import per_record_keys, record_key


logger = logging.getLogger(__name__)


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
            "apply_batch and the Pipeline pass one, or call apply(..., key=...)"
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


def child_statistics(operators: Sequence["OperatorModule"], batch: Batch) -> dict[str, Any] | None:
    """Return what each child computes for this batch, or ``None`` when no child has any.

    A child cannot compute statistics of its own while it runs: a per-record wrapper applies its
    children inside one vectorized call, by which point the batch is gone. So the wrapper computes
    them once per batch, before the batch is vectorized.

    Args:
        operators: The wrapper's children, in the order it applies them.
        batch: The batch the wrapper is about to apply.

    Returns:
        One entry per child, or ``None`` when every child computed ``None`` — so a wrapper
        over children with no statistics passes nothing down rather than an empty shell.
    """
    computed = tuple(operator.compute_statistics(batch) for operator in operators)
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


def _on_values(
    child: "OperatorModule", element: Element, stats: dict[str, Any] | None
) -> Callable[[tuple[PyTree, PyTree]], tuple[PyTree, PyTree]]:
    """``child`` applied to a record's ``(data, state)``, the record's identity held fixed.

    ``jax.lax.cond`` and ``switch`` return new arrays for everything that passes through them, so a
    wrapper passes only the record's data and state through its branches and keeps its identity.

    Args:
        child: The operator a branch applies, in its own mode.
        element: The record, whose identity the branch keeps.
        stats: The child's statistics.

    Returns:
        A branch function from ``(data, state)`` to ``(data, state)``.
    """

    def run(values: tuple[PyTree, PyTree]) -> tuple[PyTree, PyTree]:
        out = child.apply_record(element.replace(data=values[0], state=values[1]), stats)
        return out.data, out.state

    return run


def apply_where(
    child: "OperatorModule",
    should_apply: bool | jax.Array,
    element: Element,
    stats: dict[str, Any] | None,
) -> Element:
    """Apply ``child`` to the record where ``should_apply`` holds, else return it unchanged.

    Args:
        child: The operator to apply, in its own mode (``apply_record``).
        should_apply: Whether to apply it; a traced boolean is decided by ``jax.lax.cond``.
        element: The record.
        stats: The child's statistics.

    Returns:
        The record, transformed or not.
    """
    data, state = jax.lax.cond(
        should_apply,
        _on_values(child, element, stats),
        lambda values: values,
        (element.data, element.state),
    )
    return element.replace(data=data, state=state)


def apply_selected(
    children: Sequence["OperatorModule"],
    selected: int | jax.Array,
    element: Element,
    stats: dict[str, Any] | None,
) -> Element:
    """Apply the child at ``selected`` to the record; ``jax.lax.switch`` runs only that branch.

    Args:
        children: The candidate operators.
        selected: The index of the one to apply; out of range is clamped, as ``switch`` does.
        element: The record.
        stats: One entry per child (``child_statistics``), or ``None``.

    Returns:
        The record, transformed by the selected child.
    """
    branches = [
        _on_values(child, element, statistics_for_child(stats, index))
        for index, child in enumerate(children)
    ]
    data, state = jax.lax.switch(selected, branches, (element.data, element.state))
    return element.replace(data=data, state=state)


def _require_identity_kept(element: Element, out: Element, operator: "OperatorModule") -> None:
    """Refuse a transformed record whose identity is not the one it arrived with.

    The check is by object identity, at trace time: ``Element.replace`` keeps the fields it is
    not given, so a record built with ``element.replace(data=..., state=...)`` passes, and the
    check costs nothing at run time.

    Args:
        element: The record as it was handed over.
        out: The record the transform returned.
        operator: The operator whose ``apply`` ran; its class is named in the error.

    Raises:
        ValueError: If ``index``, ``epoch`` or ``draw`` is not the one handed over.
    """
    if not (out.index is element.index and out.epoch is element.epoch and out.draw is element.draw):
        raise ValueError(
            f"{type(operator).__name__}.apply changed the record's identity; return "
            "element.replace(data=..., state=...) and keep index, epoch and draw"
        )


def require_record_form(child: "OperatorModule", wrapper: "OperatorModule") -> None:
    """Refuse a child a per-record wrapper cannot apply to one record.

    Args:
        child: The child operator.
        wrapper: The wrapper deciding or merging per record.

    Raises:
        TypeError: If the child works on the whole batch.
    """
    reason = child.whole_batch_reason()
    if reason is not None:
        raise TypeError(
            f"{type(wrapper).__name__} applies its children one record at a time, and "
            f"{type(child).__name__} works on the whole batch: {reason}. Apply it before or "
            "after the wrapper, or in a SEQUENTIAL composite."
        )


def require_distinct(operators: Sequence["OperatorModule"]) -> None:
    """Refuse the same operator instance twice among a wrapper's children.

    An operator keys each record from its own base key, so one instance at two places would
    draw the same values at both; two instances draw independently.

    Args:
        operators: The wrapper's children.

    Raises:
        ValueError: If an instance appears more than once.
    """
    if len({id(operator) for operator in operators}) != len(operators):
        raise ValueError(
            "the same operator instance appears twice; it would draw the same values at each "
            "place. Build one instance per place."
        )


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

    def apply_deterministic(self, element: Element, stats: dict[str, Any] | None = None) -> Element:
        """Transform one record in deterministic mode: the record, unchanged, by default.

        A stochastic operator is an augmentation unless it says otherwise, and evaluation
        applies no augmentation. An operator whose deterministic form does something overrides
        this.

        Args:
            element: The record, without a batch axis.
            stats: This batch's statistics.

        Returns:
            The record.
        """
        del stats
        return element

    # ========================================================================
    # The per-record form
    # ========================================================================

    def whole_batch_reason(self) -> str | None:
        """Say why this operator has no per-record form, or return ``None`` when it has one.

        An operator has a per-record form when it implements ``apply``; one that overrides
        ``apply_batch`` alone works on the whole batch. A wrapper derives its answer from its
        children.

        Returns:
            The reason, naming the operator, or ``None``.
        """
        if type(self).apply is OperatorModule.apply:
            return f"{type(self).__name__} overrides apply_batch and implements no apply"
        return None

    @property
    def has_record_form(self) -> bool:
        """Whether this operator can be applied to one record (``apply_record``)."""
        return self.whole_batch_reason() is None

    def fits_statistics_per_batch(self) -> bool:
        """Whether this operator computes its statistics from each batch it is given.

        True when it overrides ``compute_statistics``; a wrapper, whose statistics are its
        children's, answers for its children.

        Returns:
            Whether the statistics depend on the batch.
        """
        return type(self).compute_statistics is not OperatorModule.compute_statistics

    def record_key(self, element: Element) -> jax.Array:
        """Return the record's key: this operator's base key folded with the record's identity.

        Equal to the key ``__call__`` gives the same record in any batch, and independent of any
        wrapper the operator sits in.

        Args:
            element: The record.

        Returns:
            The record's PRNG key.

        Raises:
            ValueError: If the record has no identity to key on.
        """
        if element.index is None:
            raise ValueError(
                f"{type(self).__name__} keys each record by its identity, and this Element has "
                "none: build it with index=..., or call apply(element, key, stats) with a key"
            )
        return record_key(self._base_key[...], element.index, element.epoch, element.draw)

    @final
    def apply_record(self, element: Element, stats: dict[str, Any] | None = None) -> Element:
        """Transform one record in this operator's mode; wrappers apply their children with it.

        A deterministic operator's ``apply`` gets no key; a stochastic one in deterministic
        mode applies ``apply_deterministic``; otherwise ``apply`` gets the record's key
        (``record_key``), so a child draws what it draws at top level.

        Args:
            element: The record, without a batch axis.
            stats: This batch's statistics for this operator.

        Returns:
            The transformed record.

        Raises:
            TypeError: If the operator works on the whole batch.
        """
        reason = self.whole_batch_reason()
        if reason is not None:
            raise TypeError(f"no per-record form: {reason}; apply it to a batch (apply_batch)")
        if not self.stochastic:
            return self.apply(element, None, stats)
        if self.deterministic:
            return self.apply_deterministic(element, stats)
        return self.apply(element, self.record_key(element), stats)

    # ========================================================================
    # Statistics
    # ========================================================================

    def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
        """Return the statistics every record of this batch passes to ``apply``.

        The default is whatever was stored with ``set_statistics``. An operator that fits
        statistics to each batch overrides this; it runs once per batch, outside the per-record
        map.

        Args:
            batch: The batch about to be applied.

        Returns:
            The statistics to give ``apply``, or None when the operator has none.
        """
        del batch
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
    # The one method an operator implements
    # ========================================================================

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Transform one record: a pure function of the record, its key and the statistics.

        Every random value is drawn from ``key``, so the same record draws the same values
        whatever batch it arrives in. ``apply`` writes no module state (a write inside the
        per-record map is refused) and returns the record's identity unchanged: build the result
        with ``element.replace(data=..., state=...)``.

        Subclasses implement this, or override ``apply_batch`` alone for an operator with no
        per-record form.

        Args:
            element: The record, without a batch axis.
            key: The record's PRNG key for a stochastic operator, ``None`` for a deterministic
                one. Pass it through ``require_key`` before drawing.
            stats: This batch's statistics (``compute_statistics``).

        Returns:
            The transformed record.

        Raises:
            NotImplementedError: If a subclass does not override this method.

        Examples:
            ```python
            def apply(self, element, key=None, stats=None):
                factor = jax.random.uniform(require_key(key, self), (), minval=0.8, maxval=1.2)
                return element.update_data({"image": element.data["image"] * factor})
            ```
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply()")

    # ========================================================================
    # The batch form
    # ========================================================================

    def apply_batch(
        self, batch: Batch, keys: jax.Array | None, stats: dict[str, Any] | None
    ) -> Batch:
        """Apply the operator to a batch: ``apply`` mapped over its records by default.

        The records are mapped with ``jax.vmap``, or ``jax.lax.scan`` (O(1) memory per record)
        when ``config.batch_strategy`` is ``"scan"``. A whole-batch operator (one record mixed
        with another, a batch-dependent model in training) overrides this instead of
        implementing ``apply``, and draws from its first record's key, ``keys[0]``.

        Args:
            batch: The batch.
            keys: One key per record from ``__call__``, or ``None`` for a deterministic operator.
            stats: This batch's statistics.

        Returns:
            The transformed batch.
        """
        return self._map_records(batch, keys, lambda element, key: self.apply(element, key, stats))

    def _map_records(  # noqa: DOC503 - _require_identity_kept raises the ValueError
        self,
        batch: Batch,
        keys: jax.Array | None,
        transform: Callable[[Element, jax.Array | None], Element],
    ) -> Batch:
        """Apply ``transform`` to every record of ``batch`` by the configured batch strategy.

        Args:
            batch: The batch.
            keys: One key per record, or ``None`` when no record draws.
            transform: Maps a record and its key to the transformed record.

        Returns:
            The batch with every record's data and state transformed.

        Raises:
            ValueError: If ``transform`` changes a record's identity.
            TypeError: If ``transform`` writes module state, which a per-record map cannot do.
        """
        # vmap needs at least one array, and a batch of no rows has nothing to apply.
        if batch.batch_size == 0 or not jax.tree.leaves((batch.data, batch.states)):
            return batch

        def one(
            data: PyTree,
            state: PyTree,
            index: ArrayValue,
            epoch: ArrayValue,
            draw: ArrayValue,
            key: jax.Array | None,
        ) -> tuple[PyTree, PyTree]:
            element = Element(data, state=state, index=index, epoch=epoch, draw=draw)
            out = transform(element, key)
            _require_identity_kept(element, out, self)
            return out.data, out.state

        inputs = (batch.data, batch.states, batch.indices, batch.epochs, batch.draws, keys)
        try:
            if self.config.batch_strategy == "scan":
                _, (data, states) = jax.lax.scan(lambda carry, x: (carry, one(*x)), None, inputs)
            else:
                data, states = jax.vmap(one)(*inputs)
        except TraceContextError as error:
            raise TypeError(
                f"{type(self).__name__}.apply wrote module state, which one record cannot do "
                "inside the per-record map (a BatchNorm in training does). Make it a whole-batch "
                "operator: override apply_batch(batch, keys, stats) instead of apply."
            ) from error
        return batch.replace(data=data, states=states)

    @final
    def __call__(self, batch: Batch) -> Batch:
        """Apply the operator to a batch: the one entry point.

        Computes the batch's statistics, derives one key per record from its identity (none for
        a deterministic operator), and applies the mode: a stochastic operator in deterministic
        mode maps ``apply_deterministic`` (the batch unchanged by default); otherwise
        ``apply_batch``.

        Args:
            batch: Input batch.

        Returns:
            Transformed batch.
        """
        stats = self.compute_statistics(batch)
        if self.stochastic and self.deterministic:
            if type(self).apply_deterministic is OperatorModule.apply_deterministic:
                return batch
            return self._map_records(
                batch, None, lambda element, _: self.apply_deterministic(element, stats)
            )
        keys = (
            per_record_keys(self._base_key[...], batch.indices, batch.epochs, batch.draws)
            if self.stochastic
            else None
        )
        return self.apply_batch(batch, keys, stats)

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
