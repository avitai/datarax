"""MapOperator - operator for applying functions to array leaves.

This module provides MapOperator, which applies user-provided array transformation
functions to leaves in element data PyTree.

Key Features:

- Unified function signature: fn(x: Array, key: Array) -> Array
- Deterministic mode: key parameter ignored
- Stochastic mode: key parameter provides per-leaf randomness
- Full-tree mode: Apply fn to all array leaves
- Subtree mode: Apply fn only to specified subtree leaves
- Uses jax.tree.map_with_path for unified implementation

BREAKING CHANGE: User functions MUST accept key parameter even in deterministic mode.
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import MapOperatorConfig
from datarax.core.operator import OperatorModule


logger = logging.getLogger(__name__)


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
       - Unified implementation with jax.tree.map_with_path

    2. **Subtree mode** (subtree specified): Apply fn only to subtree leaves
       - Path-based filtering via keypath matching
       - Other leaves pass through unchanged

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
        fn: Callable[[jax.Array, jax.Array], jax.Array],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize MapOperator.

        Args:
            config: Operator configuration
            fn: User function with signature: fn(x: Array, key: Array) -> Array
                BREAKING CHANGE: Must accept key parameter even for deterministic mode
                - Deterministic: ignore key parameter
                - Stochastic: use key for randomness
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

    def generate_random_params(
        self,
        element_keys: jax.Array,
        data_shapes: PyTree,
    ) -> PyTree | None:
        """Generate a PyTree of per-leaf, per-record PRNG keys.

        ``element_keys`` holds one stable per-record key
        (``fold_in(base_key, global_index)``). Each data leaf gets its own
        per-record key derived by folding the leaf index into ``element_keys``,
        so per-leaf randomness stays independent while remaining reproducible per
        record (invariant to batch composition, shuffle, host count, resume).

        Args:
            element_keys: ``(batch_size,)`` per-record PRNG keys.
            data_shapes: PyTree with same structure as batch.data, containing shapes.

        Returns:
            PyTree of keys matching data structure, each leaf ``(batch_size,)``
            per-record keys; or ``None`` for deterministic operators.
        """
        if not self.stochastic:
            return None

        # Flatten to get tree structure (tuples are atomic shape leaves).
        shape_leaves, tree_def = jax.tree.flatten(
            data_shapes, is_leaf=lambda x: isinstance(x, tuple)
        )
        n_leaves = len(shape_leaves)
        if n_leaves == 0:
            return jax.tree.unflatten(tree_def, [])

        # Per-leaf, per-record keys: fold the leaf index into each record's key.
        batched_keys = [
            jax.vmap(lambda key, leaf=leaf: jax.random.fold_in(key, leaf))(element_keys)
            for leaf in range(n_leaves)
        ]
        return jax.tree.unflatten(tree_def, batched_keys)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply array transformation to element (unified implementation).

        Single method handles all four modes:

        - Full-tree × deterministic
        - Full-tree × stochastic
        - Subtree × deterministic
        - Subtree × stochastic

        Uses jax.tree.map_with_path for unified traversal with keypath filtering.

        Args:
            data: Element data PyTree
            state: Element state PyTree (unchanged)
            metadata: Element metadata dict (unchanged)
            random_params: PyTree of keys (stochastic) or dummy keys (deterministic)
            stats: Optional batch statistics (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
            where state and metadata are unchanged
        """
        del stats
        # Get keys (real for stochastic, dummy/None for deterministic)
        # If random_params is None, create dummy keys matching data structure
        # This is needed for jax.eval_shape in get_output_structure()
        # Use dummy PRNG keys (not None) so stochastic functions can trace correctly
        if random_params is None:
            keys = jax.tree.map(lambda _: jax.random.key(0), data)
        else:
            keys = random_params

        def transform_leaf(keypath: Any, leaf: Any, key: Any) -> Any:
            """Transform leaf if it should be transformed."""
            # Check subtree filter (full-tree mode: always transform)
            if self._is_subtree_mode:
                if not self._is_path_in_subtree_mask(keypath, self.config.subtree):
                    return leaf  # Pass through unchanged

            # Apply user function (always with key parameter)
            return self.fn(leaf, key)

        # Single transformation call (works for all 4 cases!)
        transformed_data = jax.tree.map_with_path(transform_leaf, data, keys)
        return transformed_data, state, metadata
