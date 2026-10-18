"""A GPU queues every operator's batch program without the host waiting on it.

On a GPU, XLA runs a ``while`` whose trip count it has not proved, or a ``conditional``, by copying
the loop predicate or branch index to the host and waiting for it. A per-batch program holding one
cannot return before the device reaches it, so the data path stalls behind the training step:
rejection samplers such as JAX's default ``beta`` and ``poisson`` lower to such loops. Every
operator kind the per-record fixture records is compiled here and searched for them, under the
default batch strategy and, for the noise operator, which branches on its data, under ``scan``.
Counting instructions holds on any backend (``host_round_trips``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core import batch_ops
from datarax.core.config import MapOperatorConfig
from datarax.core.operator import OperatorModule
from datarax.operators import MapOperator
from datarax.operators.modality.image import NoiseOperator, NoiseOperatorConfig
from datarax.pipeline.dag import name_records
from tests.scripts.script_loader import load_script
from tests.test_common.step_jaxpr import compiled_text, host_round_trips


generator = load_script("generate_per_record_outputs")


def _round_trips(operator: OperatorModule, data: dict[str, jax.Array]) -> list[str]:
    """The instructions of ``operator``'s compiled batch program that make the host wait."""
    batch = name_records(batch_ops.from_arrays(data), generator.INDICES, generator.EPOCH)
    graphdef, state = nnx.split(operator)
    return host_round_trips(
        compiled_text(lambda state, batch: nnx.merge(graphdef, state)(batch), state, batch)
    )


type NoiseMode = Literal["gaussian", "salt_pepper", "poisson"]
NOISE_MODES: tuple[NoiseMode, ...] = ("gaussian", "salt_pepper", "poisson")


def _scanned_noise(mode: NoiseMode) -> NoiseOperator:
    return NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode=mode,
            batch_strategy="scan",
            **generator.stochastic_kwargs(True),
        ),
        rngs=generator.rngs(),
    )


CASES = [
    *(pytest.param(build, data, id=name) for name, build, data in generator.CASES),
    *(
        pytest.param(
            functools.partial(_scanned_noise, mode), generator.IMAGE_LABEL, id=f"{mode} scan"
        )
        for mode in NOISE_MODES
    ),
]


@pytest.mark.parametrize(("build", "data"), CASES)
def test_the_compiled_batch_program_makes_the_host_wait_nowhere(
    build: Callable[[], OperatorModule], data: dict[str, Any]
) -> None:
    trips = _round_trips(build(), data)
    assert not trips, f"{len(trips)} instructions make the host wait, the first: {trips[0][:200]}"


def _drawing(draw: Callable[[jax.Array, jax.Array], jax.Array]) -> MapOperator:
    return MapOperator(
        MapOperatorConfig(**generator.stochastic_kwargs(True)), fn=draw, rngs=generator.rngs()
    )


def test_positive_control_a_rejection_sampler_and_a_data_branch_are_found() -> None:
    exact_beta = _drawing(lambda x, key: x * jax.random.beta(key, 0.4, 0.4))
    assert _round_trips(exact_beta, generator.IMAGE_ONLY)
    branch = compiled_text(
        lambda x: jax.lax.cond(jnp.max(x) > 0.5, jnp.sin, jnp.cos, x), generator.IMAGE
    )
    assert host_round_trips(branch)


def test_negative_control_a_loop_free_draw_and_a_counted_loop_are_not() -> None:
    uniform = _drawing(lambda x, key: x * jax.random.uniform(key))
    assert _round_trips(uniform, generator.IMAGE_ONLY) == []
    counted = compiled_text(
        lambda x: jax.lax.fori_loop(0, 4, lambda _, y: jnp.sin(y), x), generator.IMAGE
    )
    assert "known_trip_count" in counted
    assert host_round_trips(counted) == []
