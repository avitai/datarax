"""The functional crop, padding and flip helpers, against the family they implement.

The semantics are torchvision's ``RandomCrop``/``CenterCrop``/``RandomHorizontalFlip``
(vision 6da25ff, ``transforms/v2/_geometry.py:783-935``, ``transforms.py:589-691``): padding given
as one int, (left/right, top/bottom) or (left, top, right, bottom); the modes constant (with
``fill``), edge, reflect (the edge pixel not repeated) and symmetric (repeated), which are
``jnp.pad``'s modes of the same names; a crop offset drawn per image, uniformly over every valid
integer offset including the last, top and left independently; a crop larger than the padded
image refused. Records are ``(H, W)`` or ``(H, W, C)``, without a batch axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing.compiles import expect_compiles

from datarax.operators.modality.image import functional


def coordinates(height: int, width: int) -> jax.Array:
    """An (H, W, 2) image whose pixel (r, c) holds (r, c): a crop names its own offset."""
    rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    return jnp.stack([rows, cols], axis=-1).astype(jnp.float32)


class TestPad:
    @pytest.mark.parametrize("mode", ["constant", "edge", "reflect", "symmetric"])
    def test_each_mode_is_jnp_pad(self, mode: str) -> None:
        image = jnp.arange(4 * 5 * 3, dtype=jnp.float32).reshape(4, 5, 3)

        padded = functional.pad(image, 2, mode=mode)

        expected = jnp.pad(image, ((2, 2), (2, 2), (0, 0)), mode=mode)
        assert jnp.array_equal(padded, expected)

    def test_constant_padding_uses_the_fill_in_the_image_dtype(self) -> None:
        image = jnp.zeros((2, 2), dtype=jnp.uint8)

        padded = functional.pad(image, 1, fill=7)

        assert padded.dtype == jnp.uint8
        assert int(padded[0, 0]) == 7 and int(padded[1, 1]) == 0

    @pytest.mark.parametrize(
        ("padding", "sides"),
        [
            (3, ((3, 3), (3, 3))),
            ((1, 2), ((2, 2), (1, 1))),  # (left/right, top/bottom)
            ((1, 2, 3, 4), ((2, 4), (1, 3))),  # (left, top, right, bottom)
        ],
    )
    def test_padding_takes_torchvisions_three_forms(
        self, padding: functional.Padding, sides: tuple[tuple[int, int], tuple[int, int]]
    ) -> None:
        image = jnp.ones((4, 4))

        padded = functional.pad(image, padding)

        assert jnp.array_equal(padded, jnp.pad(image, sides))

    @pytest.mark.parametrize(
        ("padding", "mode", "message"),
        [
            (-1, "constant", "non-negative"),
            ((1, 2, 3), "constant", "one, two or four"),
            (1, "wrap", "mode"),
            (4, "reflect", "reflect"),  # pad >= axis size: torch refuses, numpy repeats
            (5, "symmetric", "symmetric"),  # pad > axis size
        ],
    )
    def test_invalid_padding_is_refused(
        self, padding: int | tuple[int, ...], mode: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            # A three-value padding is outside the type on purpose: the refusal is at runtime
            functional.pad(jnp.ones((4, 4, 3)), padding, mode=mode)  # pyright: ignore[reportArgumentType]

    def test_symmetric_padding_up_to_the_axis_size_is_accepted(self) -> None:
        image = jnp.arange(4.0).reshape(1, 4)
        padded = functional.pad(image, (4, 0), mode="symmetric")
        assert jnp.array_equal(padded[0], jnp.array([3.0, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]))


class TestCenterCrop:
    @pytest.mark.parametrize(("size", "top_left"), [((4, 4), (2, 2)), ((5, 3), (2, 2))])
    def test_offsets_round_as_torchvision_does(
        self, size: tuple[int, int], top_left: tuple[int, int]
    ) -> None:
        # torchvision: top = int(round((H - h) / 2.0)), left likewise (Python's round)
        cropped = functional.center_crop(coordinates(8, 7), size)

        assert cropped.shape == (*size, 2)
        assert tuple(int(v) for v in cropped[0, 0]) == top_left

    def test_a_crop_larger_than_the_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="larger than"):
            functional.center_crop(jnp.ones((8, 8, 3)), (9, 8))


class TestRandomCrop:
    def test_a_crop_is_the_padded_image_at_its_offset(self) -> None:
        image = coordinates(8, 8) + 1.0  # 0 is left for the padding
        padded = functional.pad(image, 2)

        for seed in range(20):
            cropped = functional.random_crop(image, (8, 8), jax.random.key(seed), padding=2)
            inside = jnp.argwhere(cropped[..., 0] > 0)[0]
            i, j = int(inside[0]), int(inside[1])
            top = int(cropped[i, j, 0]) - 1 + 2 - i
            left = int(cropped[i, j, 1]) - 1 + 2 - j
            assert jnp.array_equal(cropped, padded[top : top + 8, left : left + 8])

    def test_offsets_are_uniform_inclusive_and_independent(self) -> None:
        image = coordinates(12, 12)
        keys = jax.random.split(jax.random.key(0), 9000)

        crops = jax.vmap(lambda key: functional.random_crop(image, (4, 4), key))(keys)
        tops = np.asarray(crops[:, 0, 0, 0]).astype(int)
        lefts = np.asarray(crops[:, 0, 0, 1]).astype(int)

        # Every offset in [0, 12 - 4] inclusive, each about 1000 times (binomial sd ~30)
        for offsets in (tops, lefts):
            counts = np.bincount(offsets, minlength=9)
            assert counts.shape == (9,)
            assert counts.min() > 850 and counts.max() < 1150
        # Top and left drawn apart: all 81 pairs occur and top == left about 1/9 of the time
        assert len(set(zip(tops.tolist(), lefts.tolist(), strict=True))) == 81
        assert abs(float(np.mean(tops == lefts)) - 1 / 9) < 0.02

    def test_a_crop_the_size_of_the_padded_image_has_offset_zero(self) -> None:
        image = coordinates(6, 6)
        cropped = functional.random_crop(image, (8, 8), jax.random.key(3), padding=1)
        assert jnp.array_equal(cropped, functional.pad(image, 1))

    def test_a_crop_larger_than_the_padded_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="larger than"):
            functional.random_crop(jnp.ones((8, 8, 3)), (11, 8), jax.random.key(0), padding=1)

    def test_a_grayscale_record_is_cropped(self) -> None:
        cropped = functional.random_crop(jnp.ones((8, 8)), (4, 5), jax.random.key(0))
        assert cropped.shape == (4, 5)

    @pytest.mark.parametrize("dtype", [jnp.uint8, jnp.float32, jnp.bfloat16])
    def test_the_dtype_is_kept(self, dtype: jnp.dtype) -> None:
        image = jnp.ones((8, 8, 3), dtype=dtype)
        cropped = functional.random_crop(image, (8, 8), jax.random.key(0), padding=4)
        assert cropped.dtype == dtype

    def test_float64_is_kept_under_x64(self) -> None:
        with jax.enable_x64(True):
            image = jnp.ones((8, 8, 3), dtype=jnp.float64)
            cropped = functional.random_crop(image, (8, 8), jax.random.key(0), padding=4)
            assert cropped.dtype == jnp.float64

    def test_jit_compiles_once_for_every_key(self) -> None:
        crop = jax.jit(lambda image, key: functional.random_crop(image, (8, 8), key, padding=4))
        image = jnp.ones((8, 8, 3))
        crop(image, jax.random.key(0))

        with expect_compiles(0):
            for seed in range(1, 4):
                jax.block_until_ready(crop(image, jax.random.key(seed)))

    def test_scan_crops_each_record_with_its_own_key(self) -> None:
        image = coordinates(12, 12)
        keys = jax.random.split(jax.random.key(1), 16)

        def step(carry: None, key: jax.Array) -> tuple[None, jax.Array]:
            return carry, functional.random_crop(image, (4, 4), key)

        _, scanned = jax.lax.scan(step, None, keys)
        mapped = jax.vmap(lambda key: functional.random_crop(image, (4, 4), key))(keys)
        assert jnp.array_equal(scanned, mapped)

    def test_the_gradient_is_the_mask_of_kept_pixels(self) -> None:
        image = jnp.ones((8, 8, 3))
        key = jax.random.key(5)

        grad = jax.grad(lambda x: jnp.sum(functional.random_crop(x, (8, 8), key, padding=2)))(image)
        cropped = functional.random_crop(image, (8, 8), key, padding=2)

        assert set(np.unique(np.asarray(grad)).tolist()) <= {0.0, 1.0}
        assert float(grad.sum()) == float(cropped.sum())  # one per kept image pixel


class TestFlip:
    def test_left_right_reverses_the_width_axis(self) -> None:
        image = coordinates(3, 4)
        flipped = functional.flip_left_right(image)
        assert jnp.array_equal(flipped, image[:, ::-1])
        assert jnp.array_equal(functional.flip_left_right(image[..., 0]), image[:, ::-1, 0])

    def test_up_down_reverses_the_height_axis(self) -> None:
        image = coordinates(3, 4)
        assert jnp.array_equal(functional.flip_up_down(image), image[::-1])

    @pytest.mark.parametrize(
        ("flip", "reverse"),
        [
            (functional.random_flip_left_right, lambda x: x[:, ::-1]),
            (functional.random_flip_up_down, lambda x: x[::-1]),
        ],
    )
    def test_a_random_flip_happens_with_its_probability(self, flip, reverse) -> None:
        image = coordinates(3, 4)
        keys = jax.random.split(jax.random.key(0), 4000)

        flipped = jax.vmap(lambda key: flip(image, key, 0.3))(keys)

        was_flipped = np.asarray(jnp.all(flipped == reverse(image), axis=(1, 2, 3)))
        unchanged = np.asarray(jnp.all(flipped == image, axis=(1, 2, 3)))
        assert np.all(was_flipped | unchanged)
        assert abs(float(was_flipped.mean()) - 0.3) < 0.03  # binomial sd ~0.007

    def test_the_probability_may_be_traced(self) -> None:
        image = coordinates(3, 4)
        flip = jax.jit(functional.random_flip_left_right)
        assert jnp.array_equal(flip(image, jax.random.key(0), 1.0), image[:, ::-1])
        assert jnp.array_equal(flip(image, jax.random.key(0), 0.0), image)
