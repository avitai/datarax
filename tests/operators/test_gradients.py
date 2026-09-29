"""Gradients through every operator family, exact in float64.

Each input family's derivative is checked against finite differences in float64
(``substrax.testing.gradients.check_input_gradients``: ``check_grads`` at its float64 tolerance,
1e-5, refusing a gradient that is zero everywhere). Correctness is a property of the code path,
so the float64 run checks it on whatever it draws, as JAX's own x64 test leg does. The shipped
float32 gradient is checked to be finite and non-zero: a float32 finite difference on these
losses (a sum against random-sign weights) carries 1e-3 to 3e-3 of rounding by construction, so
it cannot measure a float32 gradient's accuracy at any step.

Randomness is fixed by naming the records, so each check differentiates one fixed draw. Inputs sit
away from clipping boundaries, where the transforms are smooth.
"""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.test_util import check_grads
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

import datarax.operators as operators_package
from datarax.core import batch_ops, cross_modal, modality
from datarax.core.config import (
    ElementOperatorConfig,
    MapOperatorConfig,
)
from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig
from datarax.core.element_batch import Element
from datarax.core.operator import OperatorModule
from datarax.operators import ElementOperator, MapOperator
from datarax.operators.batch_mix_operator import BatchMixOperator, BatchMixOperatorConfig
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.audio.crepe_model import decode_pitch_differentiable
from datarax.operators.modality.audio.f0_operator import CrepeF0Config, CrepeF0Operator
from datarax.operators.modality.audio.loudness_operator import LoudnessConfig, LoudnessOperator
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
from datarax.operators.modality.image.noise_operator import NoiseOperator, NoiseOperatorConfig
from datarax.operators.modality.image.patch_dropout_operator import (
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
)
from datarax.operators.modality.image.rotation_operator import (
    RotationOperator,
    RotationOperatorConfig,
)
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig
from datarax.pipeline.dag import name_records
from datarax.utils import external
from datarax.utils.external import ExternalAdapterConfig, ExternalLibraryAdapter, PureJaxAdapter


_BATCH = 4
_RECORDS = jnp.arange(_BATCH, dtype=jnp.uint32)


def _rngs() -> nnx.Rngs:
    return nnx.Rngs(augment=0, default=1)


def _image() -> jax.Array:
    """Values in [0.3, 0.7]: brightness, contrast and noise stay inside the clip range."""
    grid = jnp.linspace(0.3, 0.7, _BATCH * 8 * 8 * 3)
    return grid.reshape(_BATCH, 8, 8, 3)


def _brightness() -> OperatorModule:
    return BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
            brightness_range=(-0.1, 0.1),
        ),
        rngs=_rngs(),
    )


def _contrast() -> OperatorModule:
    return ContrastOperator(
        ContrastOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment", contrast_range=(0.8, 1.2)
        ),
        rngs=_rngs(),
    )


def _map_scale(factor: float) -> OperatorModule:
    return MapOperator(MapOperatorConfig(), fn=lambda x, key: x * factor)


@dataclass(frozen=True)
class Family:
    """An operator family: how to build it and which field its input gradient flows through."""

    build: Callable[[], OperatorModule]
    field: str = "image"
    data: Callable[[], jax.Array] = _image


_INPUT_FAMILIES: dict[str, Family] = {
    "brightness": Family(_brightness),
    "contrast": Family(_contrast),
    "rotation": Family(
        lambda: RotationOperator(
            RotationOperatorConfig(
                field_key="image",
                stochastic=True,
                stream_name="augment",
                angle_range=(-20.0, 20.0),
                clip_range=(-10.0, 10.0),
            ),
            rngs=_rngs(),
        )
    ),
    "noise gaussian": Family(
        lambda: NoiseOperator(NoiseOperatorConfig(field_key="image", noise_std=0.01), rngs=_rngs())
    ),
    "noise salt and pepper": Family(
        lambda: NoiseOperator(
            NoiseOperatorConfig(
                field_key="image", mode="salt_pepper", salt_prob=0.05, pepper_prob=0.05
            ),
            rngs=_rngs(),
        )
    ),
    "dropout pixel": Family(
        lambda: DropoutOperator(
            DropoutOperatorConfig(field_key="image", dropout_rate=0.3), rngs=_rngs()
        )
    ),
    "dropout channel": Family(
        lambda: DropoutOperator(
            DropoutOperatorConfig(field_key="image", dropout_rate=0.3, mode="channel"),
            rngs=_rngs(),
        )
    ),
    "patch dropout": Family(
        lambda: PatchDropoutOperator(
            PatchDropoutOperatorConfig(field_key="image", num_patches=2, patch_size=(2, 2)),
            rngs=_rngs(),
        )
    ),
    "element": Family(
        lambda: ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="augment"),
            fn=lambda element, key: element.replace(
                data={"image": element.data["image"] * (1.0 + 0.1 * jax.random.normal(key, ()))}
            ),
            rngs=_rngs(),
        )
    ),
    "map": Family(
        lambda: MapOperator(
            MapOperatorConfig(stochastic=True, stream_name="augment"),
            fn=lambda x, key: x + 0.01 * jax.random.normal(key, x.shape),
            rngs=_rngs(),
        )
    ),
    "batch mix mixup": Family(
        lambda: BatchMixOperator(BatchMixOperatorConfig(mode="mixup"), rngs=nnx.Rngs(batch_mix=0))
    ),
    "batch mix cutmix": Family(
        lambda: BatchMixOperator(BatchMixOperatorConfig(mode="cutmix"), rngs=nnx.Rngs(batch_mix=0))
    ),
    "probabilistic": Family(
        lambda: ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=0.5), operator=_brightness(), rngs=_rngs()
        )
    ),
    "selector": Family(
        lambda: SelectorOperator(
            SelectorOperatorConfig(), operators=[_brightness(), _contrast()], rngs=_rngs()
        )
    ),
    "composite sequential": Family(
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                stochastic=True,
                stream_name="augment",
            ),
            # 1.2 keeps [0.3, 0.7] inside Brightness's clip range after its +-0.1 shift; a larger
            # scale puts values beside the clip, where finite differences cross the kink.
            operators=[_map_scale(1.2), _brightness()],
        )
    ),
    "composite ensemble mean": Family(
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.ENSEMBLE_MEAN,
            ),
            operators=[_map_scale(2.0), _map_scale(3.0)],
        )
    ),
    "composite weighted parallel": Family(
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.WEIGHTED_PARALLEL,
                weights=[0.3, 0.7],
                mix_fields=("image",),
            ),
            operators=[_map_scale(2.0), _map_scale(3.0)],
        )
    ),
    "external adapter": Family(
        lambda: ExternalLibraryAdapter(
            ExternalAdapterConfig(),
            lambda data, key: {
                **data,
                "image": data["image"] * jax.random.uniform(key, ()),
            },
            rngs=_rngs(),
        )
    ),
    "pure jax adapter": Family(
        lambda: PureJaxAdapter(
            ExternalAdapterConfig(stochastic=False, stream_name=None),
            lambda data: {**data, "image": jnp.tanh(data["image"])},
        )
    ),
}


def _weights(shape: tuple[int, ...]) -> jax.Array:
    return jax.random.normal(jax.random.key(7), shape)


def _family_loss(family: Family) -> Callable[[OperatorModule, jax.Array], jax.Array]:
    weights = _weights(family.data().shape)

    def loss(model: OperatorModule, value: jax.Array) -> jax.Array:
        out = model(name_records(batch_ops.from_arrays({family.field: value}), _RECORDS, 0))
        return jnp.sum(out.data[family.field] * weights)

    return loss


def _input_loss(family: Family) -> Callable[[jax.Array], jax.Array]:
    operator = family.build()
    loss = _family_loss(family)
    return lambda value: loss(operator, value)


@pytest.mark.parametrize("name", sorted(_INPUT_FAMILIES))
def test_the_input_gradient_matches_finite_differences_in_float64(name: str) -> None:
    """The derivative, against finite differences at ``check_grads``' float64 tolerance."""
    family = _INPUT_FAMILIES[name]

    check_input_gradients(family.build(), _family_loss(family), family.data())


@pytest.mark.parametrize("name", sorted(_INPUT_FAMILIES))
def test_the_float32_input_gradient_is_finite_and_not_zero(name: str) -> None:
    """The shipped dtype's gradient: finite everywhere, and not zero everywhere."""
    family = _INPUT_FAMILIES[name]

    gradient = jax.jit(jax.grad(_input_loss(family)))(family.data())

    assert gradient.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)


def test_a_poisson_sample_passes_no_gradient_to_its_input() -> None:
    """A Poisson draw is an integer count, so it has no pathwise derivative in its rate."""
    operator = NoiseOperator(NoiseOperatorConfig(field_key="image", mode="poisson"), rngs=_rngs())
    weights = _weights(_image().shape)

    def loss(value: jax.Array) -> jax.Array:
        out = operator(name_records(batch_ops.from_arrays({"image": value}), _RECORDS, 0))
        out = out.data
        return jnp.sum(out["image"] * weights)

    gradient = jax.jit(jax.grad(loss))(_image())

    assert jnp.all(gradient == 0.0)


class _Fuse(CrossModalOperator):
    """Mixes two fields with a learnable weight."""

    def __init__(self, config: CrossModalOperatorConfig) -> None:
        super().__init__(config)
        self.mix = nnx.Param(jnp.asarray(0.3))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del key, stats
        fused = self.mix[...] * data["image"] + (1.0 - self.mix[...]) * data["mask"]
        return element.replace(data={**data, "fused": fused})


def test_a_cross_modal_parameter_gradient_matches_finite_differences() -> None:
    fuse = _Fuse(CrossModalOperatorConfig(input_fields=["image", "mask"], output_fields=["fused"]))
    data = {"image": _image(), "mask": _image()[::-1] * 0.5}
    weights = _weights(_image().shape)

    def loss(model: OperatorModule) -> jax.Array:
        out = model(name_records(batch_ops.from_arrays(data), _RECORDS, 0))
        out = out.data
        return jnp.sum(out["fused"] * weights)

    check_parameter_gradients(fuse, loss)


def test_learnable_composite_weights_have_the_finite_difference_gradient() -> None:
    composite = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            weights=[0.3, 0.7],
            learnable_weights=True,
            mix_fields=("image",),
        ),
        operators=[_map_scale(2.0), _map_scale(3.0)],
    )
    weights = _weights(_image().shape)

    def loss(model: OperatorModule) -> jax.Array:
        out = model(name_records(batch_ops.from_arrays({"image": _image()}), _RECORDS, 0))
        out = out.data
        return jnp.sum(out["image"] * weights)

    check_parameter_gradients(composite, loss)


def _audio(seconds: float = 1.0) -> jax.Array:
    time = jnp.linspace(0.0, seconds, int(16000 * seconds), endpoint=False)
    return 0.5 * jnp.sin(2 * jnp.pi * 440.0 * time) + 0.1 * jnp.sin(2 * jnp.pi * 97.0 * time)


def test_loudness_input_and_parameter_gradients_match_finite_differences() -> None:
    operator = LoudnessOperator(LoudnessConfig(), rngs=nnx.Rngs(0))

    def from_input(model: OperatorModule, audio: jax.Array) -> jax.Array:
        out = model.apply(Element({"audio": audio})).data
        return jnp.mean(out["loudness"])

    # A derivative summed over 16000 samples: checked in float64, as the parameters are (a float64
    # probe converged to 1.7e-9 as the step shrank from 1e-3 to 1e-5).
    check_input_gradients(operator, from_input, _audio())

    def loss(model: OperatorModule) -> jax.Array:
        out = model.apply(Element({"audio": _audio()})).data
        return jnp.mean(out["loudness"])

    check_parameter_gradients(operator, loss)


def test_the_differentiable_pitch_decoder_matches_finite_differences() -> None:
    """The decoder CREPE's operator adds: a temperature softmax and a weighted average of cents.

    The network before it is ReLU convolutions and max pooling, piecewise linear in its
    parameters, so a finite difference over the whole operator crosses kinks: measured in float64,
    the central difference scattered by 1.3% around the derivative with the step, while the
    derivative agreed within 3e-4 at steps that crossed none. The smooth part is checked here.
    """
    # The decoder takes a probability distribution; a finite difference taken on the probabilities
    # themselves would leave the simplex, so the distribution is the softmax of free logits.
    logits = jax.random.normal(jax.random.key(5), (360,))

    def pitch(logits: jax.Array) -> jax.Array:
        f0_hz, confidence = decode_pitch_differentiable(jax.nn.softmax(logits), 0.5)
        return f0_hz + 100.0 * confidence

    with jax.enable_x64(True):
        check_grads(jax.jit(pitch), (logits.astype(jnp.float64),), order=1, modes=("fwd", "rev"))
    assert jnp.any(jax.grad(pitch)(logits) != 0.0)


def test_crepe_f0_parameter_gradients_are_finite_and_reach_the_network() -> None:
    operator = CrepeF0Operator(
        CrepeF0Config(capacity="tiny", differentiable=True), rngs=nnx.Rngs(0)
    )
    operator.eval()
    graphdef, params, rest = nnx.split(operator, nnx.Param, ..., graph=False)

    def loss(params: nnx.State) -> jax.Array:
        out = nnx.merge(graphdef, params, rest).apply(Element({"audio": _audio()})).data
        return jnp.mean(out["f0_hz"])

    leaves = jax.tree.leaves(jax.jit(jax.grad(loss))(params))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert sum(bool(jnp.any(leaf != 0.0)) for leaf in leaves) == len(leaves)


def test_every_operator_class_in_src_is_covered() -> None:
    """The registry's denominator: adding an operator class without a gradient check fails here."""
    covered = {type(family.build()).__name__ for family in _INPUT_FAMILIES.values()} | {
        "NoiseOperator",
        "CrossModalOperator",
        "CrepeF0Operator",
        "LoudnessOperator",
    }
    modules = [operators_package, cross_modal, modality, external]
    for info in pkgutil.walk_packages(operators_package.__path__, "datarax.operators."):
        modules.append(importlib.import_module(info.name))
    defined = {
        value.__name__
        for module in modules
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, OperatorModule)
        and value.__module__ == module.__name__
    }
    abstract_bases = {"ModalityOperator"}

    assert defined - abstract_bases - covered == set()
