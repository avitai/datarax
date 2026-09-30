"""RandomCropOperator: pad each record and crop it at its own random offset.

torchvision's ``RandomCrop`` (vision 6da25ff, ``transforms/v2/_geometry.py:783-935``) as a datarax
operator: padding in one of torchvision's forms and modes, then a crop of a fixed size whose top
and left offsets are drawn independently and uniformly over every valid offset. The offsets come
from the record's own key, so a record is cropped the same whatever batch it arrives in.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.element_batch import Element
from datarax.core.field_paths import get_field, set_field
from datarax.core.maybe import refuse_maybe
from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.core.operator import require_key
from datarax.operators.modality.image import functional


@dataclass(frozen=True)
class RandomCropOperatorConfig(ModalityOperatorConfig):
    """Configuration of a random crop.

    Attributes:
        size: ``(height, width)`` of the crop.
        padding: Padding before the crop: one int, ``(left/right, top/bottom)`` or
            ``(left, top, right, bottom)``, as torchvision takes it.
        padding_mode: ``"constant"`` (with ``fill``), ``"edge"``, ``"reflect"`` or
            ``"symmetric"``, ``jnp.pad``'s modes of those names.
        fill: The value of constant padding, applied in the record's dtype.
        stochastic: ``True`` draws each record's offset; ``False`` takes the centre crop.
    """

    size: tuple[int, int] = field(kw_only=True)
    padding: functional.Padding = field(default=0, kw_only=True)
    padding_mode: str = field(default="constant", kw_only=True)
    fill: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the size, the padding and its mode.

        Raises:
            ValueError: If the size is not a positive height and width, or the padding or its
                mode is invalid.
        """
        super().__post_init__()
        if len(self.size) != 2 or min(self.size) < 1:
            raise ValueError(f"size must be a positive (height, width), got {self.size!r}")
        functional.padding_sides(self.padding)
        if self.padding_mode not in functional.PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {functional.PADDING_MODES}, got {self.padding_mode!r}"
            )


class RandomCropOperator(ModalityOperator):
    """Pad each record and crop it, at a random offset or at the centre.

    A stochastic operator crops each record at an offset drawn from the record's key; in
    deterministic mode and in eval mode it takes the centre crop of the padded record, which is
    the record itself when the crop is the record's size.

    Examples:
        CIFAR-10's augmentation (``RandomCrop(32, padding=4)``):

        ```python
        crop = RandomCropOperator(
            RandomCropOperatorConfig(
                field_key="image", size=(32, 32), padding=4, stochastic=True, stream_name="crop"
            ),
            rngs=nnx.Rngs(crop=0),
        )
        ```
    """

    def __init__(
        self,
        config: RandomCropOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize the crop.

        Args:
            config: The crop's configuration.
            rngs: The RNG streams; a stochastic crop draws its base key from ``stream_name``.
        """
        super().__init__(config, rngs=rngs)
        self.config: RandomCropOperatorConfig = config

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Crop one record: at a random offset when stochastic, else at the centre.

        Args:
            element: The record, without a batch axis.
            key: This record's PRNG key, required when stochastic.
            stats: Unused.

        Returns:
            The record with its field cropped.
        """
        del stats
        if not self.config.stochastic:
            return self.apply_deterministic(element)
        image = self._extract_field(element.data, self.config.field_key)
        cropped = functional.random_crop(
            image,
            self.config.size,
            require_key(key, self),
            padding=self.config.padding,
            padding_mode=self.config.padding_mode,
            fill=self.config.fill,
        )
        return self._store(element, cropped)

    def apply_deterministic(self, element: Element, stats: dict[str, Any] | None = None) -> Element:
        """Take the centre crop of the padded record (eval mode, or a deterministic crop).

        Args:
            element: The record, without a batch axis.
            stats: Unused.

        Returns:
            The record with its field cropped at the centre.
        """
        del stats
        image = self._extract_field(element.data, self.config.field_key)
        padded = functional.pad(
            image, self.config.padding, mode=self.config.padding_mode, fill=self.config.fill
        )
        return self._store(element, functional.center_crop(padded, self.config.size))

    def output_spec(self, input_spec: PyTree) -> PyTree:
        """The input spec with the cropped field's height and width set to the crop's size.

        Args:
            input_spec: ``jax.ShapeDtypeStruct`` per field of one record.

        Returns:
            The spec of the record this operator returns.
        """
        source = get_field(input_spec, self.config.field_key)
        refuse_maybe(
            source,
            "RandomCropOperator crops its field as an array",
            f"data[{self.config.field_key!r}]",
        )
        cropped = jax.ShapeDtypeStruct((*self.config.size, *source.shape[2:]), source.dtype)
        return set_field(input_spec, self.config.target_key or self.config.field_key, cropped)

    def _store(self, element: Element, cropped: jax.Array) -> Element:
        """Clip if configured and write the crop into the record's target field."""
        cropped = self._apply_clip_range(cropped)
        return element.replace(data=self._remap_field(element.data, cropped))
