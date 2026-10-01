"""RandomCropOperator: torchvision's RandomCrop as a datarax operator, one offset per record.

A stochastic operator pads each record and crops it at an offset drawn from the record's own key
(``functional.random_crop``); in deterministic mode, and in eval mode, it takes the centre crop
of the padded record (the identity when the crop is the image's size, as CIFAR-10's
``RandomCrop(32, padding=4)`` on 32x32 images).
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
from datarax.core.index_words import to_words
from datarax.operators.modality.image import functional
from datarax.operators.modality.image.random_crop_operator import (
    RandomCropOperator,
    RandomCropOperatorConfig,
)
from datarax.pipeline.dag import name_records


def coordinates(height: int, width: int) -> jax.Array:
    """An (H, W, 2) image whose pixel (r, c) holds (r + 1, c + 1); padding stays 0."""
    rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    return jnp.stack([rows, cols], axis=-1).astype(jnp.float32) + 1.0


def cifar_crop(stochastic: bool = True) -> RandomCropOperator:
    config = RandomCropOperatorConfig(
        field_key="image",
        size=(32, 32),
        padding=4,
        stochastic=stochastic,
        stream_name="crop" if stochastic else None,
    )
    return RandomCropOperator(config, rngs=nnx.Rngs(crop=0) if stochastic else None)


def named_batch(images: jax.Array, indices: list[int]) -> Batch:
    return name_records(
        batch_ops.from_arrays({"image": images, "label": jnp.arange(len(indices))}),
        to_words(indices),
        0,
    )


class TestConfig:
    def test_a_stochastic_crop_needs_its_stream(self) -> None:
        with pytest.raises(ValueError, match="stream_name"):
            RandomCropOperatorConfig(field_key="image", size=(32, 32), stochastic=True)

    @pytest.mark.parametrize("size", [(0, 32), (32, -1), (32,), (32, 32, 3)])
    def test_the_size_is_a_positive_height_and_width(self, size: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="size"):
            # One- and three-value sizes are outside the type on purpose: the refusal is at runtime
            RandomCropOperatorConfig(field_key="image", size=size)  # pyright: ignore[reportArgumentType]

    def test_invalid_padding_and_modes_are_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RandomCropOperatorConfig(field_key="image", size=(8, 8), padding=-1)
        with pytest.raises(ValueError, match="mode"):
            RandomCropOperatorConfig(field_key="image", size=(8, 8), padding_mode="wrap")

    def test_no_clipping_by_default(self) -> None:
        assert RandomCropOperatorConfig(field_key="image", size=(8, 8)).clip_range is None


class TestCrop:
    def test_a_record_is_cropped_from_its_own_key(self) -> None:
        crop = cifar_crop()
        image = coordinates(32, 32)

        cropped = crop.apply(Element({"image": image}), key=jax.random.key(7)).data["image"]

        expected = functional.random_crop(image, (32, 32), jax.random.key(7), padding=4)
        assert jnp.array_equal(cropped, expected)

    def test_records_get_their_own_offsets_whatever_the_batch(self) -> None:
        crop = cifar_crop()
        images = jnp.broadcast_to(coordinates(32, 32), (8, 32, 32, 2))

        forward = crop(named_batch(images, list(range(8))))["image"]
        backward = crop(named_batch(images, list(range(7, -1, -1))))["image"]

        # Identical images, so any difference is the per-record offset
        assert len({np.asarray(forward[i]).tobytes() for i in range(8)}) > 1
        # Record k is cropped the same wherever it sits in the batch
        assert jnp.array_equal(forward, backward[::-1])

    def test_eval_mode_takes_the_centre_of_the_padded_record(self) -> None:
        crop = cifar_crop()
        crop.eval()
        images = jnp.broadcast_to(coordinates(32, 32), (4, 32, 32, 2))

        assert jnp.array_equal(crop(named_batch(images, [0, 1, 2, 3]))["image"], images)

    def test_a_deterministic_crop_is_the_centre_crop(self) -> None:
        config = RandomCropOperatorConfig(field_key="image", size=(6, 4), padding=(1, 2))
        crop = RandomCropOperator(config)
        image = coordinates(8, 8)

        cropped = crop.apply(Element({"image": image})).data["image"]

        padded = functional.pad(image, (1, 2))
        assert jnp.array_equal(cropped, functional.center_crop(padded, (6, 4)))

    def test_the_output_spec_names_the_cropped_shape(self) -> None:
        crop = RandomCropOperator(
            RandomCropOperatorConfig(
                field_key="image", size=(24, 20), stochastic=True, stream_name="crop"
            ),
            rngs=nnx.Rngs(crop=0),
        )
        spec = {
            "image": jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8),
            "label": jax.ShapeDtypeStruct((), jnp.int32),
        }

        out = crop.output_spec(spec)

        assert out["image"] == jax.ShapeDtypeStruct((24, 20, 3), jnp.uint8)
        assert out["label"] == spec["label"]

    def test_the_output_spec_matches_a_crop_it_produced(self) -> None:
        crop = cifar_crop()
        images = jnp.zeros((2, 32, 32, 3), jnp.uint8)
        out = crop(named_batch(images, [0, 1]))["image"]
        spec = crop.output_spec({"image": jax.ShapeDtypeStruct((32, 32, 3), jnp.uint8)})
        assert (out.shape[1:], out.dtype) == (spec["image"].shape, spec["image"].dtype)

    def test_under_jit_a_repeated_call_compiles_nothing(self) -> None:
        crop = cifar_crop()
        step = nnx.jit(lambda op, batch: op(batch))
        batch = named_batch(jnp.zeros((4, 32, 32, 3)), [0, 1, 2, 3])
        step(crop, batch)

        with expect_compiles(0):
            jax.block_until_ready(step(crop, batch)["image"])
