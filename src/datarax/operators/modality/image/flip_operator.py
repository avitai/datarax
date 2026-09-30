"""FlipOperator: mirror each record left to right or top to bottom.

A flip is deterministic; a random flip, torchvision's ``RandomHorizontalFlip(p)``, is
``ProbabilisticOperator(ProbabilisticOperatorConfig(probability=p), operator=flip)``: the wrapper
decides per record from the record's key and passes the record through in eval mode.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx

from datarax.core.element_batch import Element
from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.operators.modality.image import functional


FLIP_AXES = ("horizontal", "vertical")
"""``horizontal`` mirrors left to right (the width axis), ``vertical`` top to bottom."""


@dataclass(frozen=True)
class FlipOperatorConfig(ModalityOperatorConfig):
    """Configuration of a flip.

    Attributes:
        axis: ``"horizontal"`` (left to right) or ``"vertical"`` (top to bottom).
    """

    axis: str = field(default="horizontal", kw_only=True)

    def __post_init__(self) -> None:
        """Validate the axis and refuse a stochastic flip.

        Raises:
            ValueError: For an unknown axis, or ``stochastic=True``: a flip with a probability is
                a ``ProbabilisticOperator`` around this operator.
        """
        if self.stochastic:
            raise ValueError(
                "a flip is deterministic; flip with probability p through "
                "ProbabilisticOperator(ProbabilisticOperatorConfig(probability=p), operator=flip)"
            )
        super().__post_init__()
        if self.axis not in FLIP_AXES:
            raise ValueError(f"axis must be one of {FLIP_AXES}, got {self.axis!r}")


class FlipOperator(ModalityOperator):
    """Mirror each record's field on a fixed axis.

    Examples:
        CIFAR-10's augmentation (``RandomHorizontalFlip()``, p = 0.5):

        ```python
        flip = ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=0.5),
            operator=FlipOperator(FlipOperatorConfig(field_key="image")),
            rngs=nnx.Rngs(augment=0),
        )
        ```
    """

    def __init__(self, config: FlipOperatorConfig, *, rngs: nnx.Rngs | None = None) -> None:
        """Initialize the flip.

        Args:
            config: The flip's configuration.
            rngs: Unused by a deterministic operator; accepted as every operator accepts it.
        """
        super().__init__(config, rngs=rngs)
        self.config: FlipOperatorConfig = config

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Mirror one record's field.

        Args:
            element: The record, without a batch axis.
            key: Unused.
            stats: Unused.

        Returns:
            The record with its field mirrored.
        """
        del key, stats
        image = self._extract_field(element.data, self.config.field_key)
        mirror = (
            functional.flip_left_right
            if self.config.axis == "horizontal"
            else functional.flip_up_down
        )
        flipped = self._apply_clip_range(mirror(image))
        return element.replace(data=self._remap_field(element.data, flipped))
