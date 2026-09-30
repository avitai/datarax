"""FlipOperator mirrors each record on a static axis; ProbabilisticOperator makes it random.

torchvision's ``RandomHorizontalFlip(p)`` is ``ProbabilisticOperator(p)`` around a horizontal
``FlipOperator``: the wrapper decides per record from the record's key, and in eval mode passes
the record through, as torchvision's test transform applies no flip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core import batch_ops
from datarax.core.element_batch import Batch, Element
from datarax.operators.modality.image.flip_operator import FlipOperator, FlipOperatorConfig
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.pipeline.dag import name_records


def coordinates(height: int, width: int) -> jax.Array:
    rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    return jnp.stack([rows, cols], axis=-1).astype(jnp.float32)


def named_batch(images: jax.Array) -> Batch:
    return name_records(
        batch_ops.from_arrays({"image": images}),
        np.arange(images.shape[0], dtype=np.uint32),
        0,
    )


def random_horizontal_flip(probability: float) -> ProbabilisticOperator:
    flip = FlipOperator(FlipOperatorConfig(field_key="image", axis="horizontal"))
    return ProbabilisticOperator(
        ProbabilisticOperatorConfig(probability=probability),
        operator=flip,
        rngs=nnx.Rngs(augment=0),
    )


class TestConfig:
    def test_the_axis_is_horizontal_or_vertical(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            FlipOperatorConfig(field_key="image", axis="diagonal")

    def test_randomness_belongs_to_the_probabilistic_wrapper(self) -> None:
        with pytest.raises(ValueError, match="ProbabilisticOperator"):
            FlipOperatorConfig(field_key="image", stochastic=True, stream_name="flip")


class TestFlip:
    @pytest.mark.parametrize(
        ("axis", "reverse"),
        [("horizontal", lambda x: x[:, ::-1]), ("vertical", lambda x: x[::-1])],
    )
    def test_each_record_is_mirrored_on_its_axis(self, axis: str, reverse) -> None:
        flip = FlipOperator(FlipOperatorConfig(field_key="image", axis=axis))
        image = coordinates(4, 5)

        assert jnp.array_equal(flip.apply(Element({"image": image})).data["image"], reverse(image))

    def test_a_probability_flips_each_record_on_its_own_draw(self) -> None:
        wrapped = random_horizontal_flip(0.5)
        image = coordinates(4, 5)
        images = jnp.broadcast_to(image, (400, 4, 5, 2))

        out = wrapped(named_batch(images))["image"]

        flipped = np.asarray(jnp.all(out == image[:, ::-1], axis=(1, 2, 3)))
        unchanged = np.asarray(jnp.all(out == image, axis=(1, 2, 3)))
        assert np.all(flipped | unchanged)
        assert abs(float(flipped.mean()) - 0.5) < 0.08  # binomial sd 0.025

    def test_eval_mode_flips_nothing(self) -> None:
        wrapped = random_horizontal_flip(0.5)
        wrapped.eval()
        images = jnp.broadcast_to(coordinates(4, 5), (16, 4, 5, 2))

        assert jnp.array_equal(wrapped(named_batch(images))["image"], images)

    def test_under_jit_a_repeated_call_compiles_nothing(self) -> None:
        wrapped = random_horizontal_flip(0.5)
        step = nnx.jit(lambda op, batch: op(batch))
        batch = named_batch(jnp.zeros((4, 8, 8, 3)))
        step(wrapped, batch)

        with expect_compiles(0):
            jax.block_until_ready(step(wrapped, batch)["image"])
