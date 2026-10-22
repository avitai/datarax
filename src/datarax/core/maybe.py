"""``Maybe``: a data field whose value may be missing for a record.

A record either has a value for a field or it does not. A field that can be missing is
``Maybe(value, present)`` in ``data``: ``value`` holds every record's value at the field's static
shape, with zeros where a record has none, and ``present`` says which records have one (bool,
leading axis ``B`` in a batch, ``()`` for one record). Every field keeps one structure whatever
records are missing, so batches with different presence share one compiled program.

``Maybe`` has no arithmetic. ``field * 2``, ``field + 1.0``, ``field == 0``, ``jnp.mean(field)``
and ``np.asarray(field)`` raise ``TypeError``; the value is read explicitly with
``field.value_or(fill)``, or with ``field.present`` in hand, so a fill value cannot be used by
accident. Missing slots hold zeros rather than NaN: a NaN hidden behind ``jnp.where`` still makes
the gradient NaN while the loss stays finite.

Two other cases have their own representation. A value an operator hides from the model but keeps
as the target (masked pretraining) stays in ``data`` and is marked in
``state[state_keys.MASKED][field]``; an operator that fills a missing value sets ``present`` and
marks ``state[state_keys.IMPUTED][field]``. A row that is not a record is padding
(``state[state_keys.WEIGHT]`` 0).
"""

from dataclasses import dataclass
from typing import Any, ClassVar, NoReturn

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from datarax.core.element_batch import ArrayValue
from datarax.core.field_nodes import FieldNode


_EXPLICIT_READS = "read it with .value_or(fill), or with .present in hand"


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Maybe[T](FieldNode):
    """A field's values and which records have one; a missing record's slot holds zeros.

    A frozen dataclass registered as a pytree node with two children and nothing in aux: it
    passes through every JAX and Flax NNX transform, batch operation and placement like any other
    pair of leaves, ``present`` placed with its rows. The constructor validates nothing, since JAX
    builds a ``Maybe`` from tracers, specs and shardings: ``T`` is what its two leaves are, arrays
    in data (``Maybe[jax.Array]``) and ``jax.ShapeDtypeStruct`` in a spec.

    Attributes:
        value: The values, ``(B, *shape)`` in a batch or ``shape`` for one record; zeros where a
            record has no value.
        present: bool, ``(B,)`` in a batch or ``()`` for one record: which records have a value.
    """

    kind: ClassVar[str] = "a Maybe (a value that can be missing)"
    fix: ClassVar[str] = (
        f"Fill it first (value_or) or apply an operator that reads its presence: {_EXPLICIT_READS}"
    )

    value: T
    present: T

    def value_or(self: "Maybe[ArrayValue]", fill: ArrayLike) -> jax.Array:
        """Return the values where present and ``fill`` where missing.

        ``present`` is broadcast over the value's trailing axes, so it works on a batch and, under
        ``vmap``, on one record. A Python scalar fill keeps the value's dtype; an array fill is
        broadcast against the value and promoted with it, as ``jnp.where`` does.

        Args:
            fill: What a missing record reads as: a scalar, or an array broadcastable to the value.

        Returns:
            The filled values.

        Raises:
            ValueError: If ``present``'s shape is not the leading part of the value's shape.
        """
        value, present = jnp.asarray(self.value), jnp.asarray(self.present)
        if present.shape != value.shape[: present.ndim]:
            raise ValueError(
                f"a Maybe's present {present.shape} must lead its value's shape {value.shape}"
            )
        mask = present.reshape(present.shape + (1,) * (value.ndim - present.ndim))
        return jnp.where(mask, value, fill)

    def __array__(self, dtype: Any = None, copy: Any = None) -> NoReturn:
        """Refuse conversion to an array, which would use the fill value unseen.

        Args:
            dtype: NumPy's requested dtype; unused.
            copy: NumPy's copy request; unused.

        Raises:
            TypeError: Always.
        """
        del dtype, copy
        raise TypeError(f"a Maybe field is not an array: {_EXPLICIT_READS}")

    def __eq__(self, other: object) -> bool:
        """Compare two ``Maybe`` descriptions field by field; refuse comparing to anything else.

        Specs compare this way (two ``Maybe`` of ``ShapeDtypeStruct``). Comparing a field's values
        to a number would silently compare the object instead.

        Args:
            other: Another ``Maybe``.

        Returns:
            Whether both fields are equal.

        Raises:
            TypeError: If ``other`` is not a ``Maybe``.
        """
        if not isinstance(other, Maybe):
            raise TypeError(
                f"a Maybe field has no comparison with {type(other).__name__}: {_EXPLICIT_READS}"
            )
        return (self.value, self.present) == (other.value, other.present)

    def __ne__(self, other: object) -> bool:
        """The negation of ``==``, with the same refusal.

        Args:
            other: Another ``Maybe``.

        Returns:
            Whether the fields differ.
        """
        return not self == other

    def __hash__(self) -> int:
        """Hash by field, so a ``Maybe`` of specs can sit in static configuration."""
        return hash((self.value, self.present))


__all__ = ["Maybe"]
