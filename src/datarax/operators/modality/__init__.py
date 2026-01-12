"""Modality-specific operators for Datarax.

This module contains operators that work on specific data modalities (images, text, audio, etc.).
These operators extend ModalityOperator base class and demonstrate how to build
modality-specific transformations.
"""

from datarax.operators.modality.image.brightness_operator import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)
from datarax.operators.modality.image.contrast_operator import (
    ContrastOperator,
    ContrastOperatorConfig,
)
from datarax.operators.modality.image.dropout_operator import (
    DropoutOperator,
    DropoutOperatorConfig,
)
from datarax.operators.modality.image import functional
from datarax.operators.modality.image.noise_operator import (
    NoiseOperator,
    NoiseOperatorConfig,
)
from datarax.operators.modality.image.patch_dropout_operator import (
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
)
from datarax.operators.modality.image.rotation_operator import (
    RotationOperator,
    RotationOperatorConfig,
)

__all__ = [
    "BrightnessOperator",
    "BrightnessOperatorConfig",
    "ContrastOperator",
    "ContrastOperatorConfig",
    "DropoutOperator",
    "DropoutOperatorConfig",
    "functional",
    "NoiseOperator",
    "NoiseOperatorConfig",
    "PatchDropoutOperator",
    "PatchDropoutOperatorConfig",
    "RotationOperator",
    "RotationOperatorConfig",
]
