"""MapOperator - operator for applying functions to array leaves.

This module provides MapOperator, which applies user-provided array transformation
functions to leaves in element data PyTree.

Key Features:

- Unified function signature: fn(x: Array, key: Array) -> Array
- Deterministic mode: key parameter ignored
- Stochastic mode: key parameter provides per-leaf randomness
- Full-tree mode: Apply fn to all array leaves
- Subtree mode: Apply fn only to specified subtree leaves
- One function signature, ``fn(leaf, key)``, in both modes: ``key`` is ``None`` when the
  operator is deterministic
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import MapOperatorConfig
from datarax.core.element_batch import Element
from datarax.core.maybe import Maybe, missing_value_error
from datarax.core.operator import call_with_mode_key, OperatorModule


logger = logging.getLogger(__name__)


def _is_maybe(node: object) -> bool:
    return isinstance(node, Maybe)


class MapOperator(OperatorModule):
    """Unified operator for mapping functions over array leaves in data.

    Applies user-provided array transformation function to leaves in element.data
    PyTree. Supports both full-tree and subtree transformations, both deterministic
    and stochastic modes.

    BREAKING CHANGE: User Function Signature (ALWAYS required):
        fn(x: jax.Array, key: jax.Array) -> jax.Array

        - Deterministic mode (stochastic=False): Ignore key parameter
        - Stochastic mode (stochastic=True): Use key for randomness

    Two operational modes:
    1. **Full-tree mode** (subtree=None): Apply fn to all array leaves
       - One traversal of the leaves with their key paths

    2. **Subtree mode** (subtree specified): Apply fn only to subtree leaves
       - Path-based filtering via keypath matching
       - Other leaves pass through unchanged

    A ``Maybe`` field (a value that can be missing) is one entry, never two leaves: outside the
    subtree it passes through unchanged; selected, it is refused with a ``TypeError`` naming it,
    since ``fn`` would transform its fill value and its presence as data.

    Examples:
        # Deterministic full-tree (ignore key)
        config = MapOperatorConfig(subtree=None, stochastic=False)
        op = MapOperator(config, fn=lambda x, key: (x - 0.5) / 0.5, rngs=rngs)

        # Stochastic full-tree (use key for noise)
        config = MapOperatorConfig(subtree=None, stochastic=True, stream_name="augment")
        op = MapOperator(
            config,
            fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1,
            rngs=rngs
        )

        # Stochastic subtree (only augment image)
        config = MapOperatorConfig(
            subtree={"image": None},
            stochastic=True,
            stream_name="augment"
        )
        op = MapOperator(
            config,
            fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1,
            rngs=rngs
        )
    """

    def __init__(
        self,
        config: MapOperatorConfig,
        fn: Callable[[jax.Array, jax.Array], jax.Array] | Callable[[jax.Array, None], jax.Array],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize MapOperator.

        Args:
            config: Operator configuration
            fn: User function with signature ``fn(x: Array, key: Array | None) -> Array``;
                ``key`` is the leaf's own key when the operator is stochastic and ``None``
                when it is deterministic.
            rngs: Random number generators (required if stochastic=True, optional otherwise)
            name: Optional name for the operator
        """
        super().__init__(config, rngs=rngs, name=name)

        # Type narrowing for pyright - config is MapOperatorConfig
        self.config: MapOperatorConfig = config

        self.fn = fn

        # Cache whether we're in subtree mode for performance
        self._is_subtree_mode = hasattr(config, "subtree") and config.subtree is not None

    @staticmethod
    def _is_path_in_subtree_mask(keypath: tuple, subtree_mask: PyTree) -> bool:
        """Check if keypath exists in subtree mask and points to None.

        Navigates through nested dict structure following keypath.
        Returns True if path exists and final value is None.

        This is a static method to enable independent testing and reuse.
        JIT-compatible: uses pure functional path traversal on static config.

        Args:
            keypath: Tuple of JAX KeyEntry objects (e.g., DictKey, SequenceKey)
            subtree_mask: Nested dict structure with None marking transform targets

        Returns:
            True if keypath exists in mask and points to None, False otherwise

        Examples:
            mask = {"image": None, "features": {"depth": None}}
            keypath = (DictKey(key='image'),)
            MapOperator._is_path_in_subtree_mask(keypath, mask)
            True
            keypath = (DictKey(key='label'),)
            MapOperator._is_path_in_subtree_mask(keypath, mask)
            False
        """
        current = subtree_mask

        # Navigate through the keypath
        for key in keypath:
            # Extract actual key from JAX KeyEntry
            k = key.key if hasattr(key, "key") else key

            # Check if key exists in current dict level
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                # Path doesn't exist in mask
                return False

        # Check if we ended at a None leaf (transform marker)
        return current is None

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Apply array transformation to element (unified implementation).

        Single method handles all four modes:

        - Full-tree × deterministic
        - Full-tree × stochastic
        - Subtree × deterministic
        - Subtree × stochastic

        Traverses the leaves with their key paths once, filtering by subtree.

        Args:
            element: The record, without a batch axis.
            key: This record's PRNG key, or ``None`` for a deterministic operator
            stats: Optional batch statistics (unused)

        Returns:
            The transformed record.
        """
        data = element.data
        del stats
        # One key per data leaf, folded out of this record's key, so each leaf draws
        # independently while still depending only on the record. A deterministic operator
        # hands the function no key. A Maybe is one entry: fn never sees its presence as data.
        leaves, tree_def = jax.tree.flatten(data, is_leaf=_is_maybe)
        leaf_keys = [
            None if key is None else jax.random.fold_in(key, index) for index in range(len(leaves))
        ]

        def transform_leaf(
            keypath: tuple[jax.tree_util.KeyEntry, ...],
            leaf: jax.Array | Maybe,
            key: jax.Array | None,
        ) -> jax.Array | Maybe:
            """Transform leaf if it should be transformed."""
            # Check subtree filter (full-tree mode: always transform)
            if self._is_subtree_mode:
                if not self._is_path_in_subtree_mask(keypath, self.config.subtree):
                    return leaf  # Pass through unchanged
            if isinstance(leaf, Maybe):
                raise missing_value_error(
                    "MapOperator maps fn over the array leaves it selects",
                    [f"data{jax.tree_util.keystr(keypath)}"],
                )

            # Apply user function (always with key parameter)
            return call_with_mode_key(self.fn, leaf, key)

        paths = [path for path, _ in jax.tree.flatten_with_path(data, is_leaf=_is_maybe)[0]]
        transformed_data = jax.tree.unflatten(
            tree_def,
            [
                transform_leaf(path, leaf, leaf_key)
                for path, leaf, leaf_key in zip(paths, leaves, leaf_keys, strict=True)
            ],
        )
        return element.replace(data=transformed_data)
