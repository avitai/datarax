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

from typing import Any, Callable

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import MapOperatorConfig
from datarax.core.operator import OperatorModule


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
    ):
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
    def _path_in_subtree(keypath: tuple, subtree_mask: PyTree) -> bool:
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
            MapOperator._path_in_subtree(keypath, mask)
            True
            keypath = (DictKey(key='label'),)
            MapOperator._path_in_subtree(keypath, mask)
            False
        """
        current = subtree_mask

        # Navigate through the keypath
        for key in keypath:
            # Extract actual key from JAX KeyEntry
            if hasattr(key, "key"):
                k = key.key
            else:
                k = key

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
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> PyTree:
        """Generate random parameters for batch transformation.

        Generates PyTree of RNG keys matching data structure, with one key per
        batch element for each leaf. This enables per-leaf, per-element randomness.

        Args:
            rng: JAX random key (single key for entire batch)
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image": (batch_size, H, W, C)}

        Returns:
            PyTree of keys matching data structure, each leaf is Array[batch_size, 2]
            Examples: {"image": Array[batch_size, 2]} where 2 is PRNGKey shape

        Implementation:
            1. Flatten data_shapes to get list of shapes
            2. Extract batch_size from first shape
            3. Split rng into n_leaves keys (one per leaf type)
            4. For each leaf key, split into batch_size keys
            5. Unflatten into PyTree matching original structure
        """
        # Extract batch sizes from shape tuples (JAX gotcha: tuples are nodes, not leaves)
        # Use is_leaf to treat tuples as atomic values
        batch_sizes = jax.tree.map(
            lambda shape: shape[0], data_shapes, is_leaf=lambda x: isinstance(x, tuple)
        )
        batch_size_leaves = jax.tree.leaves(batch_sizes)

        if not batch_size_leaves:
            # Empty tree - return empty structure
            _, tree_def = jax.tree.flatten(data_shapes)
            return jax.tree.unflatten(tree_def, [])

        batch_size = batch_size_leaves[0]

        # Flatten to get list of shapes and tree structure
        shape_leaves, tree_def = jax.tree.flatten(
            data_shapes, is_leaf=lambda x: isinstance(x, tuple)
        )

        # Number of leaves in data PyTree
        n_leaves = len(shape_leaves)

        # Split rng into n_leaves keys (one per leaf type)
        leaf_rngs = jax.random.split(rng, n_leaves)

        # For each leaf, split into batch_size keys
        # Result: list of Array[batch_size, 2] keys
        batched_keys = [jax.random.split(leaf_rng, batch_size) for leaf_rng in leaf_rngs]

        # Unflatten back into PyTree matching data structure
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
        # Get keys (real for stochastic, dummy/None for deterministic)
        # If random_params is None, create dummy keys matching data structure
        # This is needed for jax.eval_shape in get_output_structure()
        # Use dummy PRNG keys (not None) so stochastic functions can trace correctly
        if random_params is None:
            keys = jax.tree.map(lambda _: jax.random.key(0), data)
        else:
            keys = random_params

        def transform_leaf(keypath, leaf, key):
            """Transform leaf if it should be transformed."""
            # Check subtree filter (full-tree mode: always transform)
            if self._is_subtree_mode:
                if not self._path_in_subtree(keypath, self.config.subtree):
                    return leaf  # Pass through unchanged

            # Apply user function (always with key parameter)
            return self.fn(leaf, key)

        # Single transformation call (works for all 4 cases!)
        transformed_data = jax.tree.map_with_path(transform_leaf, data, keys)
        return transformed_data, state, metadata
