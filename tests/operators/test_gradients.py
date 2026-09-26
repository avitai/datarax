"""Gradients through every operator family, checked against finite differences.

``jax.test_util.check_grads`` compares forward- and reverse-mode derivatives with finite
differences along a random direction, at JAX's own tolerances for the dtype, so no bound here is
chosen by hand. It passes trivially on a function whose derivative is zero everywhere, so every
family whose output depends on its input or its parameters also asserts a non-zero gradient.

Randomness is fixed by naming the records, so each check differentiates one fixed draw. Inputs sit
away from clipping boundaries, where the transforms are smooth.
"""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.test_util import check_grads

import datarax.operators as operators_package
from datarax.core import cross_modal, modality
from datarax.core.config import (
    ElementOperatorConfig,
    MapOperatorConfig,
)
from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig
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
            ProbabilisticOperatorConfig(operator=_brightness(), probability=0.5), rngs=_rngs()
        )
    ),
    "selector": Family(
        lambda: SelectorOperator(
            SelectorOperatorConfig(operators=[_brightness(), _contrast()]), rngs=_rngs()
        )
    ),
    "composite sequential": Family(
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=[_map_scale(2.0), _brightness()],
                stochastic=True,
                stream_name="augment",
            ),
            rngs=_rngs(),
        )
    ),
    "composite ensemble mean": Family(
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.ENSEMBLE_MEAN,
                operators=[_map_scale(2.0), _map_scale(3.0)],
            )
        )
    ),
    "composite weighted parallel": Family(
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.WEIGHTED_PARALLEL,
                operators=[_map_scale(2.0), _map_scale(3.0)],
                weights=[0.3, 0.7],
                mix_fields=("image",),
            )
        )
    ),
    "external adapter": Family(
        lambda: ExternalLibraryAdapter(
            ExternalAdapterConfig(),
            lambda data, key: {**data, "image": data["image"] * jax.random.uniform(key, ())},
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


def _input_loss(family: Family) -> Callable[[jax.Array], jax.Array]:
    operator = family.build()
    weights = _weights(family.data().shape)

    def loss(value: jax.Array) -> jax.Array:
        out, _ = operator._apply_on_raw({family.field: value}, {}, None, _RECORDS, 0)
        return jnp.sum(out[family.field] * weights)

    return loss


@pytest.mark.parametrize("name", sorted(_INPUT_FAMILIES))
def test_the_input_gradient_matches_finite_differences(name: str) -> None:
    family = _INPUT_FAMILIES[name]
    loss = _input_loss(family)

    check_grads(jax.jit(loss), (family.data(),), order=1, modes=("fwd", "rev"))


@pytest.mark.parametrize("name", sorted(_INPUT_FAMILIES))
def test_the_input_gradient_is_not_zero(name: str) -> None:
    """The control ``check_grads`` lacks: a zero derivative would pass it trivially."""
    family = _INPUT_FAMILIES[name]

    gradient = jax.jit(jax.grad(_input_loss(family)))(family.data())

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)


def test_a_poisson_sample_passes_no_gradient_to_its_input() -> None:
    """A Poisson draw is an integer count, so it has no pathwise derivative in its rate."""
    operator = NoiseOperator(NoiseOperatorConfig(field_key="image", mode="poisson"), rngs=_rngs())
    weights = _weights(_image().shape)

    def loss(value: jax.Array) -> jax.Array:
        out, _ = operator._apply_on_raw({"image": value}, {}, None, _RECORDS, 0)
        return jnp.sum(out["image"] * weights)

    gradient = jax.jit(jax.grad(loss))(_image())

    assert jnp.all(gradient == 0.0)


def _float64(tree: nnx.State) -> nnx.State:
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
        tree,
    )


def _input_check_in_float64(
    module: OperatorModule,
    loss: Callable[[OperatorModule, jax.Array], jax.Array],
    value: jax.Array,
) -> None:
    """Check ``loss``'s input gradient on a float64 copy of the module."""
    graphdef, state = nnx.split(module, graph=False)
    with jax.enable_x64(True):
        copy = nnx.merge(graphdef, _float64(state))
        check_grads(
            jax.jit(lambda value: loss(copy, value)),
            (value.astype(jnp.float64),),
            order=1,
            modes=("fwd", "rev"),
        )


def _parameter_check(
    module: OperatorModule, loss: Callable[[OperatorModule], jax.Array]
) -> nnx.State:
    """Check the gradient of ``loss`` in the module's parameters, in float64; return it.

    A parameter's derivative is a sum over every output element, which a float32 finite
    difference cannot resolve at JAX's float32 tolerance: measured for the learnable composite
    weights and the loudness parameters, float32 missed by 0.5% and 1.8% while float64 matched to
    1e-9 at every step from 1e-3 to 1e-6. The derivative is the same code in either precision.
    """
    graphdef, params, rest = nnx.split(module, nnx.Param, ..., graph=False)

    with jax.enable_x64(True):
        params64, rest64 = _float64(params), _float64(rest)

        def of_params(params: nnx.State) -> jax.Array:
            return loss(nnx.merge(graphdef, params, rest64))

        check_grads(jax.jit(of_params), (params64,), order=1, modes=("fwd", "rev"))
        return jax.jit(jax.grad(of_params))(params64)


class _Fuse(CrossModalOperator):
    """Mixes two fields with a learnable weight."""

    def __init__(self, config: CrossModalOperatorConfig) -> None:
        super().__init__(config)
        self.mix = nnx.Param(jnp.asarray(0.3))

    def apply(self, data, state, metadata, key=None, stats=None):
        del key, stats
        fused = self.mix[...] * data["image"] + (1.0 - self.mix[...]) * data["mask"]
        return {**data, "fused": fused}, state, metadata


def test_a_cross_modal_parameter_gradient_matches_finite_differences() -> None:
    fuse = _Fuse(CrossModalOperatorConfig(input_fields=["image", "mask"], output_fields=["fused"]))
    data = {"image": _image(), "mask": _image()[::-1] * 0.5}
    weights = _weights(_image().shape)

    def loss(model: OperatorModule) -> jax.Array:
        out, _ = model._apply_on_raw(data, {}, None, _RECORDS, 0)
        return jnp.sum(out["fused"] * weights)

    gradient = _parameter_check(fuse, loss)

    assert gradient["mix"][...] != 0.0


def test_learnable_composite_weights_have_the_finite_difference_gradient() -> None:
    composite = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[_map_scale(2.0), _map_scale(3.0)],
            weights=[0.3, 0.7],
            learnable_weights=True,
            mix_fields=("image",),
        )
    )
    weights = _weights(_image().shape)

    def loss(model: OperatorModule) -> jax.Array:
        out, _ = model._apply_on_raw({"image": _image()}, {}, None, _RECORDS, 0)
        return jnp.sum(out["image"] * weights)

    gradient = _parameter_check(composite, loss)

    assert jnp.any(jax.tree.leaves(gradient)[0] != 0.0)


def _audio(seconds: float = 1.0) -> jax.Array:
    time = jnp.linspace(0.0, seconds, int(16000 * seconds), endpoint=False)
    return 0.5 * jnp.sin(2 * jnp.pi * 440.0 * time) + 0.1 * jnp.sin(2 * jnp.pi * 97.0 * time)


def test_loudness_input_and_parameter_gradients_match_finite_differences() -> None:
    operator = LoudnessOperator(LoudnessConfig(), rngs=nnx.Rngs(0))

    def from_input(model: OperatorModule, audio: jax.Array) -> jax.Array:
        out, _, _ = model.apply({"audio": audio}, {}, None)
        return jnp.mean(out["loudness"])

    # A derivative summed over 16000 samples: checked in float64, as the parameters are (a float64
    # probe converged to 1.7e-9 as the step shrank from 1e-3 to 1e-5).
    _input_check_in_float64(operator, from_input, _audio())
    assert jnp.any(jax.grad(lambda audio: from_input(operator, audio))(_audio()) != 0.0)

    def loss(model: OperatorModule) -> jax.Array:
        out, _, _ = model.apply({"audio": _audio()}, {}, None)
        return jnp.mean(out["loudness"])

    gradient = _parameter_check(operator, loss)
    assert any(jnp.any(leaf != 0.0) for leaf in jax.tree.leaves(gradient))


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
        out, _, _ = nnx.merge(graphdef, params, rest).apply({"audio": _audio()}, {}, None)
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
