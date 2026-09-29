"""Wrappers under the one-method contract (design D6).

A sequential composite calls each child on the whole batch, so it holds per-record and whole-batch
children. A wrapper deciding or merging per record takes per-record children only. An operator
keys a record from its own base key wherever it sits; a wrapper's key serves its own decision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core import batch_ops
from datarax.core.config import BatchMixOperatorConfig, OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.image.noise_operator import NoiseOperator, NoiseOperatorConfig
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig
from datarax.pipeline.dag import name_records


B = 8


def _batch() -> Batch:
    image = jnp.full((B, 4, 4, 1), 0.5, jnp.float32)
    return name_records(batch_ops.from_arrays({"image": image}), jnp.arange(B) + 3, 1)


def _noise(seed: int = 0) -> NoiseOperator:
    return NoiseOperator(
        NoiseOperatorConfig(field_key="image", mode="gaussian", clip_range=None),
        rngs=nnx.Rngs(augment=seed),
    )


def _mixup() -> BatchMixOperator:
    return BatchMixOperator(BatchMixOperatorConfig(mode="mixup"), rngs=nnx.Rngs(batch_mix=0))


class Shift(OperatorModule):
    """Adds 1 to the image."""

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        return element.replace(data={**element.data, "image": element.data["image"] + 1.0})


class Center(OperatorModule):
    """Subtracts the batch mean, fitted per batch."""

    def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
        return {"mean": jnp.mean(batch.data["image"])}

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        assert stats is not None
        return element.replace(
            data={**element.data, "image": element.data["image"] - stats["mean"]}
        )


def _sequential(*operators: OperatorModule) -> CompositeOperatorModule:
    return CompositeOperatorModule(
        CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL), operators=list(operators)
    )


def _composite_of(
    strategy: CompositionStrategy, **extra: Any
) -> Callable[[OperatorModule], OperatorModule]:
    """A builder of a one-child composite with this strategy."""

    def build(child: OperatorModule) -> OperatorModule:
        return CompositeOperatorModule(
            CompositeOperatorConfig(strategy=strategy, **extra), operators=[child]
        )

    return build


def _probabilistic(child: OperatorModule) -> OperatorModule:
    return ProbabilisticOperator(
        ProbabilisticOperatorConfig(probability=0.5), operator=child, rngs=nnx.Rngs(0)
    )


def _selector(child: OperatorModule) -> OperatorModule:
    return SelectorOperator(SelectorOperatorConfig(), operators=[child], rngs=nnx.Rngs(augment=0))


def _always(data: Any) -> bool:
    return True


def _first(data: Any) -> int:
    return 0


# Every wrapper that decides or merges per record.
PER_RECORD_WRAPPERS: dict[str, Callable[[OperatorModule], OperatorModule]] = {
    "probabilistic": _probabilistic,
    "selector": _selector,
    "parallel": _composite_of(CompositionStrategy.PARALLEL, merge_strategy="mean"),
    "weighted parallel": _composite_of(
        CompositionStrategy.WEIGHTED_PARALLEL, mix_fields=("image",)
    ),
    "ensemble": _composite_of(CompositionStrategy.ENSEMBLE_MEAN),
    "conditional sequential": _composite_of(
        CompositionStrategy.CONDITIONAL_SEQUENTIAL, conditions=[_always]
    ),
    "conditional parallel": _composite_of(
        CompositionStrategy.CONDITIONAL_PARALLEL, conditions=[_always], merge_strategy="sum"
    ),
    "branching": _composite_of(CompositionStrategy.BRANCHING, router=_first),
}


class TestChildKeys:
    """A child draws the same values wherever it sits."""

    def test_a_child_draws_the_same_values_in_every_wrapper(self) -> None:
        batch = _batch()
        top = _noise()(batch).data["image"]
        forms = {
            "sequential": _sequential(_noise()),
            "probabilistic p=1": ProbabilisticOperator(
                ProbabilisticOperatorConfig(probability=1.0), operator=_noise()
            ),
            "one-child selector": SelectorOperator(
                SelectorOperatorConfig(), operators=[_noise()], rngs=nnx.Rngs(augment=7)
            ),
        }
        for name, wrapper in forms.items():
            np.testing.assert_allclose(
                wrapper(batch).data["image"], top, rtol=0, atol=1e-6, err_msg=name
            )

    def test_a_wrappers_seed_does_not_change_its_childs_draws(self) -> None:
        batch = _batch()
        outputs = [
            SelectorOperator(
                SelectorOperatorConfig(), operators=[_noise()], rngs=nnx.Rngs(augment=seed)
            )(batch).data["image"]
            for seed in (7, 8)
        ]
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_stacking_apply_record_equals_the_batch_form_in_training(self) -> None:
        batch = _batch()
        wrapper = ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=0.5),
            operator=_sequential(_noise(), Shift(OperatorConfig())),
            rngs=nnx.Rngs(augment=2),
        )
        stacked = jnp.stack(
            [wrapper.apply_record(batch_ops.element(batch, i)).data["image"] for i in range(B)]
        )
        np.testing.assert_allclose(stacked, wrapper(batch).data["image"], rtol=0, atol=1e-6)

    def test_a_wrapper_that_decides_nothing_draws_no_key(self) -> None:
        assert not _sequential(_noise()).stochastic
        assert not ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=1.0), operator=_noise()
        ).stochastic

    def test_the_same_instance_twice_is_refused(self) -> None:
        noise = _noise()
        with pytest.raises(ValueError, match="same operator"):
            _sequential(noise, noise)
        with pytest.raises(ValueError, match="same operator"):
            SelectorOperator(
                SelectorOperatorConfig(), operators=[noise, noise], rngs=nnx.Rngs(augment=0)
            )
        composite = CompositeOperatorModule(
            CompositeOperatorConfig(strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL),
            operators=[noise],
        )
        with pytest.raises(ValueError, match="same operator"):
            composite.add_operator(noise)


class TestSequentialComposition:
    """A sequential composite equals calling its children in turn."""

    def test_it_equals_its_children_in_turn_over_per_record_and_whole_batch_children(
        self,
    ) -> None:
        batch = _batch()
        expected = _noise(1)(_mixup()(_noise(0)(batch)))
        out = _sequential(_noise(0), _mixup(), _noise(1))(batch)
        np.testing.assert_allclose(out.data["image"], expected.data["image"], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(out.states["mix_partner"], expected.states["mix_partner"])

    def test_each_child_fits_its_statistics_on_its_own_input(self) -> None:
        out = _sequential(Shift(OperatorConfig()), Center(OperatorConfig()))(_batch())
        np.testing.assert_allclose(jnp.mean(out.data["image"]), 0.0, atol=1e-6)

    def test_its_children_follow_their_own_mode(self) -> None:
        composite = _sequential(_noise(), Shift(OperatorConfig()))
        composite.eval()
        np.testing.assert_allclose(composite(_batch()).data["image"], 1.5)


class TestPerRecordWrappersRefuseWholeBatchChildren:
    """A wrapper deciding or merging per record refuses a child with no per-record form."""

    @pytest.mark.parametrize("build", PER_RECORD_WRAPPERS.values(), ids=PER_RECORD_WRAPPERS.keys())
    def test_a_whole_batch_child_is_refused_at_construction(
        self, build: Callable[[OperatorModule], OperatorModule]
    ) -> None:
        with pytest.raises(TypeError, match="whole batch"):
            build(_mixup())
        with pytest.raises(TypeError, match="whole batch"):
            build(_sequential(_noise(), _mixup()))

    def test_a_chain_whose_later_child_fits_statistics_per_batch_is_refused(self) -> None:
        chain = _sequential(Shift(OperatorConfig()), Center(OperatorConfig()))
        assert not chain.has_record_form
        with pytest.raises(TypeError, match="Center"):
            ProbabilisticOperator(ProbabilisticOperatorConfig(probability=1.0), operator=chain)
        with pytest.raises(TypeError, match="Center"):
            CompositeOperatorModule(
                CompositeOperatorConfig(
                    strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
                    conditions=[lambda data: True, lambda data: True],
                ),
                operators=[Shift(OperatorConfig()), Center(OperatorConfig())],
            )

    def test_a_chain_of_per_record_children_is_accepted(self) -> None:
        chain = _sequential(Center(OperatorConfig()), Shift(OperatorConfig()))
        assert chain.has_record_form
        wrapper = ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=1.0), operator=chain
        )
        np.testing.assert_allclose(wrapper(_batch()).data["image"], 1.0, atol=1e-6)


class TestNestedWrappersInAChain:
    """A wrapper answers for its children whether it fits statistics per batch."""

    def test_a_chain_whose_later_child_wraps_a_statistics_fitter_is_refused(self) -> None:
        for later in (
            ProbabilisticOperator(
                ProbabilisticOperatorConfig(probability=1.0), operator=Center(OperatorConfig())
            ),
            SelectorOperator(
                SelectorOperatorConfig(),
                operators=[Center(OperatorConfig())],
                rngs=nnx.Rngs(augment=0),
            ),
            _composite_of(CompositionStrategy.PARALLEL, merge_strategy="mean")(
                Center(OperatorConfig())
            ),
        ):
            chain = _sequential(Shift(OperatorConfig()), later)
            assert not chain.has_record_form, type(later).__name__

    def test_a_chain_whose_later_wrapper_fits_nothing_has_a_record_form(self) -> None:
        later = SelectorOperator(
            SelectorOperatorConfig(), operators=[Shift(OperatorConfig())], rngs=nnx.Rngs(augment=0)
        )
        assert _sequential(Shift(OperatorConfig()), later).has_record_form


def test_an_unknown_strategy_is_refused_at_construction() -> None:
    config = CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL)
    object.__setattr__(config, "strategy", "not-a-strategy")
    with pytest.raises(ValueError, match="Unknown strategy"):
        CompositeOperatorModule(config, operators=[Shift(OperatorConfig())])
