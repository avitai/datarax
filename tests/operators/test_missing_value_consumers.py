"""Operators and missing values: a ``Maybe`` field passes through, or is refused by name.

An operator that moves records whole (per-record maps, wrappers deciding per record) carries a
``Maybe`` like any other field. An operator that treats a field as an array (maps a function over
leaves, mixes, merges or reduces values, transforms an image or a signal) would treat ``present``
as data or use the fill value, so it refuses a ``Maybe`` with a ``TypeError`` naming the field;
the caller fills the value first (``value_or``) or writes an operator that reads ``present``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core import batch_ops, Maybe
from datarax.core.config import MapOperatorConfig, OperatorConfig
from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import to_words
from datarax.core.operator import OperatorModule
from datarax.operators.batch_mix_operator import BatchMixOperator, BatchMixOperatorConfig
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator
from datarax.operators.modality.audio.f0_operator import CrepeF0Config, CrepeF0Operator
from datarax.operators.modality.audio.loudness_operator import LoudnessConfig, LoudnessOperator
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    DropoutOperator,
    DropoutOperatorConfig,
    FlipOperator,
    FlipOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
    RandomCropOperator,
    RandomCropOperatorConfig,
    RotationOperator,
    RotationOperatorConfig,
)
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.pipeline.dag import name_records


B = 6
PRESENT = np.array([True, False, True, True, False, False])


def _depth() -> Maybe:
    values = np.arange(B * 4, dtype=np.float32).reshape(B, 2, 2, 1) + 1.0
    return Maybe(
        jnp.asarray(np.where(PRESENT[:, None, None, None], values, 0.0)), jnp.asarray(PRESENT)
    )


def _batch(**extra: Any) -> Batch:
    data = {"x": jnp.linspace(0.0, 1.0, B * 3, dtype=jnp.float32).reshape(B, 3), "depth": _depth()}
    return name_records(batch_ops.from_arrays({**data, **extra}), to_words(jnp.arange(B) + 10), 1)


def _assert_depth_unchanged(out: Batch) -> None:
    field = out.data["depth"]
    assert isinstance(field, Maybe)
    np.testing.assert_array_equal(field.present, PRESENT)
    np.testing.assert_array_equal(field.value, _depth().value)


class Shift(OperatorModule):
    """Adds a drawn offset to ``x``; leaves every other field as it is."""

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        offset = 0.0 if key is None else jax.random.uniform(key, ())
        return element.update_data({"x": element.data["x"] + offset})


def _shift(strategy: str = "vmap", stochastic: bool = True) -> Shift:
    config = OperatorConfig(
        stochastic=stochastic,
        stream_name="augment" if stochastic else None,
        batch_strategy=strategy,
    )
    return Shift(config, rngs=nnx.Rngs(augment=0))


class TestPassedThroughWhole:
    """Operators that move records whole carry a ``Maybe`` field unchanged."""

    @pytest.mark.parametrize("strategy", ["vmap", "scan"])
    def test_a_per_record_operator_under_either_strategy(self, strategy: str) -> None:
        out = jax.jit(lambda b: _shift(strategy)(b))(_batch())

        _assert_depth_unchanged(out)
        assert not np.array_equal(out.data["x"], _batch().data["x"])

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_under_nnx_jit_in_graph_and_tree_mode(self, graph: bool) -> None:
        out = nnx.jit(lambda o, b: o(b), graph=graph)(_shift(), _batch())

        _assert_depth_unchanged(out)

    def test_a_probabilistic_wrapper_deciding_per_record(self) -> None:
        wrapper = ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=0.5),
            operator=_shift(stochastic=False),
            rngs=nnx.Rngs(augment=1),
        )

        _assert_depth_unchanged(wrapper(_batch()))

    def test_a_sequential_composite(self) -> None:
        composite = CompositeOperatorModule(
            CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL),
            operators=[_shift(), Shift(OperatorConfig(stochastic=False))],
        )

        _assert_depth_unchanged(composite(_batch()))

    def test_map_operator_over_a_subtree_without_the_field(self) -> None:
        op = MapOperator(MapOperatorConfig(subtree={"x": None}), fn=lambda x, _key: x * 2)

        out = op(_batch())

        _assert_depth_unchanged(out)
        np.testing.assert_allclose(out.data["x"], _batch().data["x"] * 2)

    def test_batch_mix_of_another_field(self) -> None:
        op = BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", data_field="x"), rngs=nnx.Rngs(batch_mix=0)
        )

        _assert_depth_unchanged(op(_batch()))


def _merging(strategy: CompositionStrategy, **config: Any) -> CompositeOperatorModule:
    ops = [Shift(OperatorConfig(stochastic=False)), Shift(OperatorConfig(stochastic=False))]
    return CompositeOperatorModule(
        CompositeOperatorConfig(strategy=strategy, **config), operators=ops
    )


def _image_operators() -> list[Any]:
    """Every image operator, reading the ``Maybe`` field ``depth``."""
    rngs = nnx.Rngs(0, augment=0)
    return [
        pytest.param(
            lambda: BrightnessOperator(BrightnessOperatorConfig(field_key="depth")), id="brightness"
        ),
        pytest.param(
            lambda: ContrastOperator(ContrastOperatorConfig(field_key="depth")), id="contrast"
        ),
        pytest.param(
            lambda: DropoutOperator(
                DropoutOperatorConfig(field_key="depth", stochastic=True, stream_name="augment"),
                rngs=rngs,
            ),
            id="dropout",
        ),
        pytest.param(lambda: FlipOperator(FlipOperatorConfig(field_key="depth")), id="flip"),
        pytest.param(
            lambda: NoiseOperator(
                NoiseOperatorConfig(field_key="depth", stochastic=True, stream_name="augment"),
                rngs=rngs,
            ),
            id="noise",
        ),
        pytest.param(
            lambda: PatchDropoutOperator(
                PatchDropoutOperatorConfig(field_key="depth", patch_size=(1, 1)), rngs=rngs
            ),
            id="patch-dropout",
        ),
        pytest.param(
            lambda: RandomCropOperator(RandomCropOperatorConfig(field_key="depth", size=(1, 1))),
            id="random-crop",
        ),
        pytest.param(
            lambda: RotationOperator(RotationOperatorConfig(field_key="depth", angle=10.0)),
            id="rotation",
        ),
    ]


class TestRefusedByName:
    """Operators that treat a field as an array refuse a ``Maybe`` and name the field."""

    def test_map_operator_over_the_whole_tree(self) -> None:
        op = MapOperator(MapOperatorConfig(), fn=lambda x, _key: x * 2)

        with pytest.raises(TypeError, match=r"\['depth'\].*Maybe"):
            op(_batch())

    def test_map_operator_over_a_subtree_holding_the_field(self) -> None:
        op = MapOperator(MapOperatorConfig(subtree={"depth": None}), fn=lambda x, _key: x * 2)

        with pytest.raises(TypeError, match=r"\['depth'\].*Maybe"):
            op(_batch())

    @pytest.mark.parametrize(
        "strategy",
        [
            CompositionStrategy.ENSEMBLE_MEAN,
            CompositionStrategy.ENSEMBLE_SUM,
            CompositionStrategy.ENSEMBLE_MAX,
            CompositionStrategy.ENSEMBLE_MIN,
        ],
    )
    def test_an_ensemble_reduction(self, strategy: CompositionStrategy) -> None:
        with pytest.raises(TypeError, match=r"\['depth'\].*Maybe"):
            _merging(strategy)(_batch())

    @pytest.mark.parametrize("merge", ["concat", "stack", "sum", "mean", "dict"])
    def test_a_parallel_merge(self, merge: str) -> None:
        with pytest.raises(TypeError, match=r"\['depth'\].*Maybe"):
            _merging(CompositionStrategy.PARALLEL, merge_strategy=merge)(_batch())

    @pytest.mark.parametrize("merge", ["concat", "sum", "mean"])
    def test_a_conditional_parallel_merge(self, merge: str) -> None:
        composite = _merging(
            CompositionStrategy.CONDITIONAL_PARALLEL,
            merge_strategy=merge,
            conditions=[lambda _data: True, lambda _data: False],
        )

        with pytest.raises(TypeError, match=r"\['depth'\].*Maybe"):
            composite(_batch())

    def test_a_parallel_merge_by_a_function_of_the_caller_passes(self) -> None:
        """A custom ``merge_fn`` reads the outputs as they are; it decides what a Maybe means."""
        composite = _merging(CompositionStrategy.PARALLEL, merge_fn=lambda outputs: outputs[0])

        _assert_depth_unchanged(composite(_batch()))

    def test_a_weighted_parallel_mixing_the_field(self) -> None:
        composite = _merging(
            CompositionStrategy.WEIGHTED_PARALLEL, weights=[0.5, 0.5], mix_fields=["depth"]
        )

        with pytest.raises(TypeError, match=r"depth.*Maybe"):
            composite(_batch())

    @pytest.mark.parametrize("mode", ["mixup", "cutmix"])
    def test_batch_mix_of_the_field(self, mode: str) -> None:
        op = BatchMixOperator(
            BatchMixOperatorConfig(mode=mode, data_field="depth"), rngs=nnx.Rngs(batch_mix=0)
        )

        with pytest.raises(TypeError, match=r"depth.*Maybe"):
            op(_batch())

    @pytest.mark.parametrize("build", _image_operators())
    def test_an_image_operator(self, build: Callable[[], OperatorModule]) -> None:
        with pytest.raises(TypeError, match=r"depth.*Maybe"):
            build()(_batch())

    def test_a_crop_output_spec(self) -> None:
        spec = {
            "depth": Maybe(
                jax.ShapeDtypeStruct((2, 2, 1), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_)
            )
        }
        op = RandomCropOperator(RandomCropOperatorConfig(field_key="depth", size=(1, 1)))

        with pytest.raises(TypeError, match=r"depth.*Maybe"):
            op.output_spec(spec)

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: LoudnessOperator(LoudnessConfig()), id="loudness"),
            pytest.param(lambda: CrepeF0Operator(CrepeF0Config()), id="crepe-f0"),
        ],
    )
    def test_an_audio_operator(self, build: Callable[[], OperatorModule]) -> None:
        audio = Maybe(jnp.zeros((B, 1024), jnp.float32), jnp.asarray(PRESENT))

        with pytest.raises(TypeError, match=r"audio.*Maybe"):
            build()(_batch(audio=audio))

    def test_a_cross_modal_operator(self) -> None:
        class Concatenate(CrossModalOperator):
            def apply(
                self,
                element: Element,
                key: jax.Array | None = None,
                stats: dict[str, Any] | None = None,
            ) -> Element:
                fused = jnp.concatenate(self._extract_inputs(element.data), axis=-1)
                return element.replace(data=self._store_outputs(element.data, [fused]))

        op = Concatenate(
            CrossModalOperatorConfig(input_fields=["x", "depth"], output_fields=["fused"]),
            rngs=nnx.Rngs(0),
        )

        with pytest.raises(TypeError, match=r"depth.*Maybe"):
            op(_batch())
