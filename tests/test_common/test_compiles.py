"""``expect_first_call_compiles`` measures a first call whatever ran earlier in the process."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from substrax.testing.compiles import compiled_programs

from tests.test_common.compiles import expect_first_call_compiles


def _double(x: jax.Array) -> jax.Array:
    return x * 2


def test_a_program_compiled_earlier_in_the_process_is_built_again() -> None:
    ones = jnp.ones(3)
    jax.block_until_ready(jax.jit(_double)(ones))
    with compiled_programs() as served_from_the_cache:
        jax.block_until_ready(jax.jit(_double)(ones))
    assert served_from_the_cache == [], "control: a second jit of one function shares its cache"
    with expect_first_call_compiles("jit(_double)"):
        jax.block_until_ready(jax.jit(_double)(ones))


def test_a_block_building_other_programs_is_refused() -> None:
    with (
        pytest.raises(AssertionError, match=r"jit\(_double\).*observed \[\]"),
        expect_first_call_compiles("jit(_double)"),
    ):
        pass
