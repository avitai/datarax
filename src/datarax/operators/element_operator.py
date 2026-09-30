"""ElementOperator - operator for element-level transformations.

This module provides ElementOperator, which applies user-provided element
transformation functions to whole records (``Element``: data and state).

Key Difference from MapOperator:

- MapOperator: fn(array_leaf, key) -> array_leaf (per-array-leaf transformation)
- ElementOperator: fn(element, key) -> element (per-element transformation)

Key Features:

- Full element access: User function sees the entire Element and can modify data and state
- Coordinated transformations: Transform multiple fields together
- Deterministic mode: key parameter ignored
- Stochastic mode: key parameter provides per-element randomness
- Uses Element.replace() pattern for immutable updates
"""

import logging
from collections.abc import Callable
from typing import Any

import jax
from flax import nnx

from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Element
from datarax.core.operator import call_with_mode_key, OperatorModule
from datarax.typing import PRNGKey


logger = logging.getLogger(__name__)


class ElementOperator(OperatorModule):
    """Unified operator for element-level transformations.

    Applies user-provided element transformation function to entire Element
    structures. Unlike MapOperator (which transforms array leaves), ElementOperator
    provides access to the full element (data and state), enabling
    coordinated transformations.

    User Function Signature:

        fn(element: Element, key: jax.Array | None) -> Element

        - element: Element with .data and .state
        - key: the record's PRNG key when the operator is stochastic, ``None`` when it is
          deterministic
        - Returns: New Element (use element.replace() for immutable updates)

    Use Cases:
    1. **Coordinated transformations**: Flip image AND mask together
    2. **State tracking**: Update state based on transformation applied
    3. **Complex augmentation pipelines**: Access multiple fields at once

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
        def flip_both(element, key):  # Coordinated augmentation: one decision, two fields
            flip = jax.random.uniform(key) < 0.5
            new_data = jax.lax.cond(
                flip,
                # [:, ::-1] is the width axis of an (H, W, C) image and an (H, W) mask
                lambda e: {"image": e.data["image"][:, ::-1],
                           "mask": e.data["mask"][:, ::-1]},
                lambda e: e.data,
                element
            )
            return element.replace(data=new_data)
        ```
    """

    def __init__(
        self,
        config: ElementOperatorConfig,
        fn: Callable[[Element, PRNGKey], Element] | Callable[[Element, None], Element],
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

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Apply the user function to the record.

        Args:
            element: The record, without a batch axis.
            key: This record's PRNG key, or ``None`` for a deterministic operator
            stats: Optional batch statistics (unused)

        Returns:
            The transformed record.
        """
        del stats
        return call_with_mode_key(self.fn, element, key)
