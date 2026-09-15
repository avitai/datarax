"""WEIGHTED_PARALLEL mixes the fields its operators write and passes every other field through.

The mixture applies to named data fields, as Kornia's ``data_keys`` and DDSP's named signals do: a
field the operators do not write, such as a label or a conditioning signal, keeps its value and
dtype. Static weights form a linear combination (DDSP sums its harmonic and noise synthesizers).
Learnable weights are logits mixed with ``softmax(logits / temperature)``, the relaxation DARTS and
Faster AutoAugment use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.image.brightness_operator import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)


def _scale(factor: float) -> ElementOperator:
    """An operator that writes only ``signal``."""

    def scale(element, key):
        del key
        return element.update_data({"signal": element.data["signal"] * factor})

    return ElementOperator(ElementOperatorConfig(), fn=scale, rngs=nnx.Rngs(0))


def _record() -> dict[str, jax.Array]:
    return {
        "signal": jnp.array([10.0]),
        "f0_hz": jnp.array([440.0]),
        "label": jnp.array(3, dtype=jnp.int32),
    }


def _weighted(**config) -> CompositeOperatorModule:
    return CompositeOperatorModule(
        CompositeOperatorConfig(strategy=CompositionStrategy.WEIGHTED_PARALLEL, **config),
        rngs=nnx.Rngs(0),
    )


def test_static_weights_sum_the_named_field_and_pass_every_other_field_through() -> None:
    mix = _weighted(
        operators=[_scale(2.0), _scale(3.0)], weights=[1.0, 0.1], mix_fields=("signal",)
    )

    out = mix(Batch([Element(data=_record())])).get_data()

    np.testing.assert_allclose(np.asarray(out["signal"]), [[23.0]], rtol=1e-6)  # 20 + 0.1 * 30
    np.testing.assert_array_equal(np.asarray(out["f0_hz"]), [[440.0]])
    assert out["label"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out["label"]), [3])


def test_mixed_fields_default_to_the_fields_the_operators_declare() -> None:
    operators = [
        BrightnessOperator(
            BrightnessOperatorConfig(field_key="image", brightness_delta=delta, clip_range=None),
            rngs=nnx.Rngs(0),
        )
        for delta in (0.2, -0.2)
    ]
    mix = _weighted(operators=operators, weights=[0.75, 0.25])
    record = {"image": jnp.full((2, 2, 1), 0.5), "label": jnp.array(1, dtype=jnp.int32)}

    out = mix(Batch([Element(data=record)])).get_data()

    assert mix.config.mix_fields == ("image",)
    np.testing.assert_allclose(np.asarray(out["image"]), 0.6, rtol=1e-6)  # 0.75*0.7 + 0.25*0.3
    assert out["label"].dtype == jnp.int32


def test_operators_that_declare_no_field_need_mix_fields() -> None:
    with pytest.raises(ValueError, match="mix_fields"):
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[_scale(2.0), _scale(3.0)],
            weights=[0.5, 0.5],
        )


def test_learnable_weights_start_at_the_configured_mixture_and_sharpen_with_temperature() -> None:
    """Logits start at ``log(weights / sum)``, so ``softmax(logits / T)`` is proportional to
    ``weights ** (1 / T)``."""
    batch = Batch([Element(data=_record())])
    common = {
        "operators": [_scale(2.0), _scale(3.0)],
        "weights": [0.7, 0.3],
        "learnable_weights": True,
        "mix_fields": ("signal",),
    }
    warm = _weighted(**common)
    cold = _weighted(**common, temperature=0.5)

    sharpened = np.array([0.49, 0.09]) / 0.58
    np.testing.assert_allclose(np.asarray(warm(batch).get_data()["signal"]), [[23.0]], rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(cold(batch).get_data()["signal"]),
        [[float(sharpened @ np.array([20.0, 30.0]))]],
        rtol=1e-6,
    )


@pytest.mark.parametrize("weights", [[1.0, 0.0], [1.0, -0.5]])
def test_learnable_weights_need_positive_initial_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[_scale(2.0), _scale(3.0)],
            weights=weights,
            learnable_weights=True,
            mix_fields=("signal",),
        )


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="temperature"):
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[_scale(2.0), _scale(3.0)],
            learnable_weights=True,
            temperature=0.0,
            mix_fields=("signal",),
        )


def test_gradients_reach_the_weight_logits() -> None:
    mix = _weighted(
        operators=[_scale(2.0), _scale(3.0)],
        weights=[0.5, 0.5],
        learnable_weights=True,
        mix_fields=("signal",),
    )
    record = _record()

    def loss(model: CompositeOperatorModule) -> jax.Array:
        data, _, _ = model.apply(record, {}, None)
        return jnp.sum(data["signal"])

    grads = nnx.grad(loss, argnums=nnx.DiffState(0, nnx.Param))(mix)
    (logits_grad,) = jax.tree.leaves(grads)

    # Moving weight toward the operator with the larger output (30 > 20) raises the loss.
    assert float(logits_grad[0]) < 0.0 < float(logits_grad[1])


def test_weight_key_mixes_only_the_named_fields() -> None:
    mix = _weighted(
        operators=[_scale(2.0), _scale(3.0)], weight_key="op_weights", mix_fields=("signal",)
    )

    data, _, _ = mix.apply({**_record(), "op_weights": jnp.array([0.7, 0.3])}, {}, None)

    np.testing.assert_allclose(np.asarray(data["signal"]), [23.0], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(data["f0_hz"]), [440.0])
    assert "op_weights" not in data


def test_mixture_weights_are_the_static_weights_as_given() -> None:
    mix = _weighted(
        operators=[_scale(2.0), _scale(3.0)], weights=[1.0, 0.1], mix_fields=("signal",)
    )

    np.testing.assert_allclose(np.asarray(mix.mixture_weights()), [1.0, 0.1], rtol=1e-6)


@pytest.mark.parametrize(
    ("temperature", "expected"), [(1.0, [0.7, 0.3]), (0.5, [0.49 / 0.58, 0.09 / 0.58])]
)
def test_mixture_weights_of_learnable_weights_are_the_temperature_softmax(
    temperature: float, expected: list[float]
) -> None:
    mix = _weighted(
        operators=[_scale(2.0), _scale(3.0)],
        weights=[0.7, 0.3],
        learnable_weights=True,
        temperature=temperature,
        mix_fields=("signal",),
    )

    np.testing.assert_allclose(np.asarray(mix.mixture_weights()), expected, rtol=1e-6)


def test_mixture_weights_come_from_each_record_under_weight_key() -> None:
    mix = _weighted(
        operators=[_scale(2.0), _scale(3.0)], weight_key="op_weights", mix_fields=("signal",)
    )

    with pytest.raises(ValueError, match="weight_key"):
        mix.mixture_weights()


def test_only_weighted_parallel_composites_have_mixture_weights() -> None:
    sequential = CompositeOperatorModule(
        CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL, operators=[_scale(2.0)]),
        rngs=nnx.Rngs(0),
    )

    with pytest.raises(ValueError, match="WEIGHTED_PARALLEL"):
        sequential.mixture_weights()
