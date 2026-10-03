"""A stream pass's key is folded on the host CPU device from the key's words (SQ11, C5b review).

A stream's pass ``p`` is ordered by ``fold_in(key, p)``. The pipeline's key is read once per run
as uint32 words on the host, and every fold runs in a program on the CPU device from those words,
so a pass opens with no read back from an accelerator and under a global
``jax.transfer_guard("disallow")``, which refuses every implicit transfer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.prng import key_words
from datarax.sources._source_base import key_integer, pass_generator, pass_seed
from tests.test_common.transfers import device_to_host_raises, implicit_upload_raises


def _words(seed: int) -> np.ndarray:
    return np.asarray(jax.random.key_data(jax.random.key(seed)), np.uint32)


@pytest.mark.parametrize("seed", [0, 3, 2**31 + 1])
@pytest.mark.parametrize("pass_index", [0, 1, 7, 2**31 + 5])
def test_a_pass_seed_is_the_fold_of_the_key_s_words(seed: int, pass_index: int) -> None:
    """The value is ``fold_in(key, pass)``'s data, as the stream has always seeded a pass."""
    words = _words(seed)
    folded = jax.random.fold_in(jax.random.wrap_key_data(jnp.asarray(words)), pass_index)
    assert pass_seed(words, pass_index) == key_integer(key_words(folded))


def test_a_pass_seed_needs_no_implicit_transfer() -> None:
    assert implicit_upload_raises(), "the guard must fire on an implicit upload"
    assert device_to_host_raises() or jax.default_backend() == "cpu"
    words = _words(1)
    expected = [pass_seed(words, p) for p in range(3)]
    with jax.transfer_guard("disallow"):
        assert [pass_seed(words, p) for p in range(3)] == expected
        generator = pass_generator(words, 2)
    np.testing.assert_array_equal(
        generator.permutation(10), pass_generator(words, 2).permutation(10)
    )


def test_passes_never_share_a_seed() -> None:
    words = _words(5)
    assert len({pass_seed(words, p) for p in range(100)}) == 100


@pytest.mark.parametrize("given", ["typed", "raw", "words"])
def test_the_key_s_words_are_read_from_any_form_of_the_key(given: str) -> None:
    key = jax.random.key(9)
    form = {
        "typed": key,
        "raw": jax.random.key_data(key),
        "words": np.asarray(jax.random.key_data(key), np.uint32),
    }[given]
    words = key_words(form)
    assert isinstance(words, np.ndarray) and words.dtype == np.uint32
    np.testing.assert_array_equal(words, np.asarray(jax.random.key_data(key)))


def test_a_key_integer_takes_the_words() -> None:
    words = np.asarray([1, 2], np.uint32)
    assert key_integer(words) == int.from_bytes(words.tobytes(), "little")
