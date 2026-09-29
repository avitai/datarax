"""Evaluation mode: a stochastic operator in deterministic mode applies no augmentation.

Flax's convention holds for every operator: ``module.eval()`` sets ``deterministic=True``,
``module.train()`` sets it back, and ``nnx.view(module, deterministic=True)`` returns a view in
that mode without touching the module. A stochastic operator in deterministic mode draws no key
and returns its record unchanged; a composition runs its deterministic children and skips its
stochastic ones.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from datarax.core import batch_ops
from datarax.core.config import ElementOperatorConfig, MapOperatorConfig, OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.operators import ElementOperator, MapOperator
from datarax.operators.batch_mix_operator import BatchMixOperator, BatchMixOperatorConfig
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
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


_BATCH = 6


def _images() -> dict[str, jax.Array]:
    values = jnp.linspace(0.2, 0.8, _BATCH * 8 * 8 * 3)
    return {"image": values.reshape(_BATCH, 8, 8, 3)}


def _rngs() -> nnx.Rngs:
    return nnx.Rngs(augment=0, default=1)


def _noise() -> OperatorModule:
    return NoiseOperator(
        NoiseOperatorConfig(field_key="image", stochastic=True, stream_name="augment"),
        rngs=_rngs(),
    )


def _brightness() -> OperatorModule:
    return BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
            brightness_range=(-0.2, 0.2),
        ),
        rngs=_rngs(),
    )


def _contrast() -> OperatorModule:
    return ContrastOperator(
        ContrastOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment", contrast_range=(0.5, 1.5)
        ),
        rngs=_rngs(),
    )


def _add_draw(element: Element, key: jax.Array) -> Element:
    return element.replace(data={"image": element.data["image"] + jax.random.normal(key, ())})


_STOCHASTIC_FAMILIES: dict[str, Callable[[], OperatorModule]] = {
    "noise": _noise,
    "dropout": lambda: DropoutOperator(
        DropoutOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment", dropout_rate=0.5
        ),
        rngs=_rngs(),
    ),
    "patch_dropout": lambda: PatchDropoutOperator(
        PatchDropoutOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
            num_patches=2,
            patch_size=(2, 2),
        ),
        rngs=_rngs(),
    ),
    "brightness": _brightness,
    "contrast": _contrast,
    "rotation": lambda: RotationOperator(
        RotationOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment", angle_range=(-30.0, 30.0)
        ),
        rngs=_rngs(),
    ),
    "element": lambda: ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="augment"), fn=_add_draw, rngs=_rngs()
    ),
    "map": lambda: MapOperator(
        MapOperatorConfig(subtree=None, stochastic=True, stream_name="augment"),
        fn=lambda leaf, key: leaf + jax.random.normal(key, leaf.shape),
        rngs=_rngs(),
    ),
    "probabilistic": lambda: ProbabilisticOperator(
        ProbabilisticOperatorConfig(probability=0.5), operator=_brightness(), rngs=_rngs()
    ),
    "selector": lambda: SelectorOperator(
        SelectorOperatorConfig(), operators=[_brightness(), _contrast()], rngs=_rngs()
    ),
}


def _apply(operator: OperatorModule) -> jax.Array:
    return operator(batch_ops.from_arrays(_images()))["image"]


@pytest.mark.parametrize("family", sorted(_STOCHASTIC_FAMILIES))
class TestEachStochasticFamily:
    """Every stochastic operator family honours the flax mode flag."""

    def test_train_mode_augments(self, family: str) -> None:
        """The control: without it, an operator that never augmented would pass the rest."""
        assert not jnp.allclose(_apply(_STOCHASTIC_FAMILIES[family]()), _images()["image"])

    def test_eval_returns_the_input(self, family: str) -> None:
        operator = _STOCHASTIC_FAMILIES[family]()
        operator.eval()

        assert jnp.array_equal(_apply(operator), _images()["image"])

    def test_train_after_eval_augments_as_before(self, family: str) -> None:
        reference = _apply(_STOCHASTIC_FAMILIES[family]())
        operator = _STOCHASTIC_FAMILIES[family]()
        operator.eval()
        operator.train()

        assert jnp.array_equal(_apply(operator), reference)

    def test_a_deterministic_view_leaves_the_operator_augmenting(self, family: str) -> None:
        operator = _STOCHASTIC_FAMILIES[family]()
        view = nnx.view(operator, deterministic=True)

        assert jnp.array_equal(_apply(view), _images()["image"])
        assert not jnp.allclose(_apply(operator), _images()["image"])


class TestBatchMix:
    """A batch-level operator is off in deterministic mode on both of its paths."""

    @staticmethod
    def _operator() -> BatchMixOperator:
        return BatchMixOperator(BatchMixOperatorConfig(mode="mixup"), rngs=nnx.Rngs(batch_mix=0))

    @staticmethod
    def _batch() -> Batch:
        return batch_ops.from_stacked(
            batch_ops.stack([Element(data={"value": jnp.array([float(i)])}) for i in range(4)])
        )

    def test_train_mode_mixes(self) -> None:
        mixed = self._operator().apply_batch(self._batch())
        assert not jnp.array_equal(mixed.data["value"], self._batch().data["value"])

    def test_eval_leaves_the_batch_on_both_paths(self) -> None:
        operator = self._operator()
        operator.eval()
        batch = self._batch()
        raw = {"value": batch.data["value"]}

        assert jnp.array_equal(operator.apply_batch(batch).data["value"], raw["value"])
        assert jnp.array_equal(operator(batch_ops.from_arrays(raw)).data["value"], raw["value"])


class TestComposite:
    """A composition in deterministic mode runs its deterministic children only."""

    @staticmethod
    def _composite() -> CompositeOperatorModule:
        shift = ElementOperator(
            ElementOperatorConfig(stochastic=False),
            fn=lambda element, key: element.replace(data={"image": element.data["image"] + 1.0}),
        )
        return CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                stochastic=True,
                stream_name="augment",
            ),
            operators=[shift, _noise()],
            rngs=_rngs(),
        )

    def test_train_mode_shifts_and_augments(self) -> None:
        out = _apply(self._composite())
        assert not jnp.allclose(out, _images()["image"] + 1.0)

    def test_eval_shifts_without_augmenting(self) -> None:
        composite = self._composite()
        composite.eval()

        assert jnp.array_equal(_apply(composite), _images()["image"] + 1.0)


_SEEN_KEYS: list[jax.Array | None] = []


class _KeySpy(OperatorModule):
    """Records every key it is handed and returns its record unchanged."""

    def apply(self, data, state, metadata, key=None, stats=None):
        del metadata, stats
        _SEEN_KEYS.append(key)
        return data, state, None


class TestDispatch:
    """What deterministic mode reaches and what it does not."""

    def test_a_stochastic_operator_in_eval_is_never_applied(self) -> None:
        _SEEN_KEYS.clear()
        spy = _KeySpy(OperatorConfig(stochastic=True, stream_name="augment"), rngs=_rngs())
        spy.eval()

        spy._vmap_apply(_images(), {})

        assert _SEEN_KEYS == []

    def test_the_spy_is_applied_in_train_mode(self) -> None:
        """The control for the test above."""
        _SEEN_KEYS.clear()
        _KeySpy(OperatorConfig(stochastic=True, stream_name="augment"), rngs=_rngs())._vmap_apply(
            _images(), {}
        )

        assert _SEEN_KEYS

    def test_eval_leaves_a_deterministic_operator_applying(self) -> None:
        shift = ElementOperator(
            ElementOperatorConfig(stochastic=False),
            fn=lambda element, key: element.replace(data={"image": element.data["image"] * 2.0}),
        )
        shift.eval()

        assert jnp.array_equal(_apply(shift), _images()["image"] * 2.0)

    def test_a_deterministic_element_function_is_handed_no_key(self) -> None:
        seen: list[jax.Array | None] = []

        def record(element: Element, key: jax.Array | None) -> Element:
            seen.append(key)
            return element

        ElementOperator(ElementOperatorConfig(stochastic=False), fn=record)._vmap_apply(
            _images(), {}
        )

        assert seen and all(key is None for key in seen)

    def test_train_and_eval_each_compile_once(self) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda operator, batch: operator._vmap_apply(batch, {})[0]))
        operator = _noise()

        with counter.expect(new_traces=1):
            apply(operator, _images())
        with counter.expect(new_traces=0):
            apply(operator, _images())
        operator.eval()
        with counter.expect(new_traces=1):
            apply(operator, _images())
        with counter.expect(new_traces=0):
            apply(operator, _images())


@pytest.mark.parametrize(
    "build",
    [
        lambda: NoiseOperatorConfig(field_key="image", stochastic=False),
        lambda: DropoutOperatorConfig(field_key="image", stochastic=False),
        lambda: PatchDropoutOperatorConfig(field_key="image", stochastic=False),
    ],
    ids=["noise", "dropout", "patch_dropout"],
)
def test_an_inherently_random_operator_refuses_to_be_deterministic(build: Callable) -> None:
    """Its draws come from a key; turning it off is evaluation mode, not a fixed key."""
    with pytest.raises(ValueError, match=r"\.eval\(\)"):
        build()
