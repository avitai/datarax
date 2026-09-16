"""Brightness and contrast operators use one parameter per mode.

A stochastic operator draws each record's value from its range; a deterministic operator applies
its fixed value. Each config refuses the parameter its mode does not use, so a value set for the
mode that ignores it is reported rather than silently dropped: a deterministic operator given
``brightness_range=(0.2, 0.2)`` used to return the image unchanged.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
)


IMAGES = jnp.linspace(0.2, 0.6, 24, dtype=jnp.float32).reshape(2, 2, 2, 3)

# Each operator with the names its two modes use and the values each mode resolves unset.
CASES = [
    pytest.param(
        BrightnessOperatorConfig,
        BrightnessOperator,
        "brightness_range",
        "brightness_delta",
        (-0.2, 0.2),
        0.0,
        (0.1, 0.3),
        0.1,
        id="brightness",
    ),
    pytest.param(
        ContrastOperatorConfig,
        ContrastOperator,
        "contrast_range",
        "contrast_factor",
        (0.8, 1.2),
        1.0,
        (1.2, 1.5),
        1.3,
        id="contrast",
    ),
]
MODE_CASES = pytest.mark.parametrize(
    (
        "config_cls",
        "operator_cls",
        "range_field",
        "fixed_field",
        "default_range",
        "default_fixed",
        "a_range",
        "a_fixed",
    ),
    CASES,
)


@MODE_CASES
def test_a_range_in_deterministic_mode_is_refused(
    config_cls: Any,
    operator_cls: Any,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
    a_range: tuple[float, float],
    a_fixed: float,
) -> None:
    """Deterministic mode applies the fixed value, so a range there would do nothing."""
    del operator_cls, default_range, default_fixed, a_fixed
    with pytest.raises(ValueError, match=f"{range_field}.*stochastic=True.*{fixed_field}"):
        config_cls(field_key="image", **{range_field: a_range})


@MODE_CASES
def test_a_fixed_value_in_stochastic_mode_is_refused(
    config_cls: Any,
    operator_cls: Any,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
    a_range: tuple[float, float],
    a_fixed: float,
) -> None:
    """Stochastic mode draws from the range, so a fixed value there would do nothing."""
    del operator_cls, default_range, default_fixed, a_range
    with pytest.raises(ValueError, match=f"{fixed_field}.*stochastic=False.*{range_field}"):
        config_cls(
            field_key="image", stochastic=True, stream_name="augment", **{fixed_field: a_fixed}
        )


@MODE_CASES
def test_each_mode_resolves_the_parameter_it_uses(
    config_cls: Any,
    operator_cls: Any,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
    a_range: tuple[float, float],
    a_fixed: float,
) -> None:
    """Unset, each mode fills in its own default and leaves the other mode's parameter None."""
    del operator_cls, a_range, a_fixed
    deterministic = config_cls(field_key="image")
    stochastic = config_cls(field_key="image", stochastic=True, stream_name="augment")

    assert getattr(deterministic, fixed_field) == default_fixed
    assert getattr(deterministic, range_field) is None
    assert getattr(stochastic, range_field) == default_range
    assert getattr(stochastic, fixed_field) is None


@MODE_CASES
def test_a_deterministic_operator_applies_its_fixed_value(
    config_cls: Any,
    operator_cls: Any,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
    a_range: tuple[float, float],
    a_fixed: float,
) -> None:
    """The value a deterministic operator is given reaches the image."""
    del range_field, default_range, default_fixed, a_range
    operator = operator_cls(
        config_cls(field_key="image", clip_range=None, **{fixed_field: a_fixed}),
        rngs=nnx.Rngs(0),
    )

    out, _, _ = operator.apply({"image": IMAGES[0]}, {}, {})

    assert not np.allclose(np.asarray(out["image"]), np.asarray(IMAGES[0]))


@MODE_CASES
def test_a_stochastic_operator_without_a_key_is_refused(
    config_cls: Any,
    operator_cls: Any,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
    a_range: tuple[float, float],
    a_fixed: float,
) -> None:
    """A stochastic operator draws from the record's key, so it cannot run without one."""
    del fixed_field, default_range, default_fixed, a_fixed
    operator = operator_cls(
        config_cls(
            field_key="image",
            stochastic=True,
            stream_name="augment",
            clip_range=None,
            **{range_field: a_range},
        ),
        rngs=nnx.Rngs(augment=0),
    )

    with pytest.raises(ValueError, match="stochastic and needs a per-record key"):
        operator.apply({"image": IMAGES[0]}, {}, {})


@MODE_CASES
def test_a_stochastic_batch_changes_every_record(
    config_cls: Any,
    operator_cls: Any,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
    a_range: tuple[float, float],
    a_fixed: float,
) -> None:
    """Every record of a batch is drawn for, so none comes back unchanged."""
    del fixed_field, default_range, default_fixed, a_fixed
    operator = operator_cls(
        config_cls(
            field_key="image",
            stochastic=True,
            stream_name="augment",
            clip_range=None,
            **{range_field: a_range},
        ),
        rngs=nnx.Rngs(augment=0),
    )
    batch = Batch([Element(data={"image": image}) for image in IMAGES])

    out = np.asarray(operator.apply_batch(batch).get_data()["image"])

    assert out.shape == IMAGES.shape
    assert all(not np.allclose(got, given) for got, given in zip(out, np.asarray(IMAGES)))
