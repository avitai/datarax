"""ElementOperator - operator for element-level transformations.

This module provides ElementOperator, which applies user-provided element
transformation functions to entire Element structures (data + state + metadata).

Key Difference from MapOperator:

- MapOperator: fn(array_leaf, key) -> array_leaf (per-array-leaf transformation)
- ElementOperator: fn(element, key) -> element (per-element transformation)

Key Features:

- Full element access: User function sees entire Element, can modify data/state/metadata
- Coordinated transformations: Transform multiple fields together
- Deterministic mode: key parameter ignored
- Stochastic mode: key parameter provides per-element randomness
- Uses Element.replace() pattern for immutable updates
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Element
from datarax.core.metadata import Metadata
from datarax.core.operator import OperatorModule
from datarax.typing import PRNGKey


logger = logging.getLogger(__name__)


class ElementOperator(OperatorModule):
    """Unified operator for element-level transformations.

    Applies user-provided element transformation function to entire Element
    structures. Unlike MapOperator (which transforms array leaves), ElementOperator
    provides access to the full element (data + state + metadata), enabling
    coordinated transformations.

    User Function Signature:

        fn(element: Element, key: jax.Array) -> Element

        - element: Element with .data, .state, .metadata attributes
        - key: JAX random key (use for stochastic ops, ignore for deterministic)
        - Returns: New Element (use element.replace() for immutable updates)

    Use Cases:
    1. **Coordinated transformations**: Flip image AND mask together
    2. **State tracking**: Update state based on transformation applied
    3. **Complex augmentation pipelines**: Access multiple fields at once
    4. **Metadata-aware processing**: Transform based on metadata values

    Examples:
        ```python
        def normalize(element, key):  # Deterministic element transformation
            new_data = {"value": element.data["value"] / 255.0}
            return element.replace(data=new_data)
        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=normalize, rngs=rngs)
        def add_noise(element, key):  # Stochastic element augmentation
            noise = jax.random.normal(key, element.data["image"].shape) * 0.1
            new_data = {"image": element.data["image"] + noise}
            return element.replace(data=new_data)
        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        op = ElementOperator(config, fn=add_noise, rngs=rngs)
        def flip_both(element, key):  # Coordinated augmentation
            flip = jax.random.uniform(key) < 0.5
            new_data = jax.lax.cond(
                flip,
                lambda e: {"image": e.data["image"][..., ::-1],
                           "mask": e.data["mask"][..., ::-1]},
                lambda e: e.data,
                element
            )
            return element.replace(data=new_data)
        ```
    """

    def __init__(
        self,
        config: ElementOperatorConfig,
        fn: Callable[[Element, PRNGKey], Element],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize ElementOperator.

        Args:
            config: Operator configuration
            fn: User function with signature: fn(element: Element, key: Array) -> Element
                - Deterministic mode: ignore key parameter
                - Stochastic mode: use key for randomness
            rngs: Random number generators (required if stochastic=True)
            name: Optional name for the operator
        """
        super().__init__(config, rngs=rngs, name=name)

        # Type narrowing for pyright - config is ElementOperatorConfig
        self.config: ElementOperatorConfig = config

        self.fn = fn

    def generate_random_params(
        self,
        element_keys: PRNGKey | None,
        data_shapes: PyTree,
    ) -> PRNGKey | None:
        """Return the per-record PRNG keys for the user function.

        ``element_keys`` already holds one stateless key per record (derived by
        the base class as ``fold_in(base_key, global_index)``), which is exactly
        what the user function's ``fn(element, key)`` signature expects — so this
        just passes them through. Each ``apply()`` receives its element's key.

        Args:
            element_keys: ``(batch_size,)`` per-record PRNG keys, or ``None`` for
                deterministic operators.
            data_shapes: Unused (kept for signature compatibility).

        Returns:
            The per-record keys, or ``None`` for deterministic operators.
        """
        del data_shapes
        if not self.stochastic:
            return None
        return element_keys

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply element transformation.

        Constructs an Element from data/state/metadata, passes to user function,
        and extracts results back.

        Args:
            data: Element data PyTree
            state: Element state PyTree
            metadata: Element metadata dict (unchanged - not vmapped)
            random_params: RNG key for this element (from generate_random_params)
            stats: Optional batch statistics (unused)

        Returns:
            Tuple of (transformed_data, transformed_state, transformed_metadata)
        """
        del stats
        # Construct Element for user function
        # Cast metadata to Metadata | None for Element constructor
        element = Element(data=data, state=state, metadata=cast(Metadata | None, metadata))

        # Get key (real for stochastic, dummy for deterministic)
        key = random_params if random_params is not None else jax.random.key(0)

        # Apply user function
        transformed_element = self.fn(element, key)

        # Extract results - cast metadata back to dict type for base class compatibility
        return (
            transformed_element.data,
            transformed_element.state,
            cast(dict[str, Any] | None, transformed_element.metadata),
        )
