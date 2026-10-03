"""A mix's union batches, read on the host, through operators and every transform a step meets.

Batches of one mix share one structure whatever children their rows come from, so a compiled
step over them compiles once across presence patterns: under ``jax.jit``, ``nnx.jit`` in graph
and tree mode, the Trainer's ``nnx.jit_partial`` binding and a placement sharded over a data
mesh. ``value_or`` reads a record's value under ``vmap``, a ``(K, B, ...)`` chunk scans like K
calls, and gradients through an operator filling missing values match a float64 finite-difference
reference and are zero where nothing was filled.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from substrax.spmd import place_batch_on_shards
from substrax.testing.compiles import expect_compiles
from substrax.testing.gradients import check_parameter_gradients

from datarax.core import batch_ops
from datarax.core.element_batch import Batch
from datarax.core.state_keys import IMPUTED
from tests.core.test_maybe import _imputed_loss, CrossModalFill
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.mixing import four_presence_cases


_B = 8


def _patterns() -> list[Batch]:
    """Four batches of one mix whose rows come from different children."""
    mix = four_presence_cases()
    order = np.asarray(mix.record_indices_at(0, len(mix)))
    owner = np.arange(len(mix)) % 4
    rows = [
        np.flatnonzero(np.isin(owner, [0, 1]))[:_B],
        np.flatnonzero(np.isin(owner, [2, 3]))[:_B],
        np.arange(_B),
        np.flatnonzero(owner == 3)[:_B].repeat(2)[:_B],
    ]
    return [mix.get_batch(order[r]) for r in rows]


def _on_device(batches: list[Batch]) -> list[Batch]:
    return [jax.device_put(batch) for batch in batches]


class TestOneProgramAcrossPresencePatterns:
    def test_jax_jit(self) -> None:
        op = CrossModalFill(rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(op)

        def run(state: Any, batch: Batch) -> jax.Array:
            return _imputed_loss(nnx.merge(graphdef, state), batch)

        step = jax.jit(run)
        first, *rest = _on_device(_patterns())
        with expect_first_call_compiles("jit(run)"):
            jax.block_until_ready(step(state, first))
        with expect_compiles(0):
            for batch in rest:
                jax.block_until_ready(step(state, batch))

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_nnx_jit(self, graph: bool) -> None:
        op = CrossModalFill(rngs=nnx.Rngs(0))
        step = nnx.jit(lambda model, batch: _imputed_loss(model, batch), graph=graph)
        first, *rest = _on_device(_patterns())
        with expect_compiles(1):
            jax.block_until_ready(step(op, first))
        with expect_compiles(0):
            for batch in rest:
                jax.block_until_ready(step(op, batch))

    def test_the_trainer_s_jit_partial_binding(self) -> None:
        op = CrossModalFill(rngs=nnx.Rngs(0))
        bound = nnx.jit_partial(lambda model, batch: _imputed_loss(model, batch), op, graph=False)
        first, *rest = _on_device(_patterns())
        with expect_compiles(1):
            jax.block_until_ready(bound(first))
        with expect_compiles(0):
            for batch in rest:
                jax.block_until_ready(bound(batch))

    def test_a_placement_sharded_over_a_data_mesh(self) -> None:
        mesh = Mesh(np.array(jax.devices()[:4]), ("data",))
        rows, replicated = NamedSharding(mesh, P("data")), NamedSharding(mesh, P())
        placed = [
            place_batch_on_shards(batch, batch_ops.shardings(batch, rows, replicated))
            for batch in _patterns()
        ]
        count = jax.jit(lambda b: jnp.sum(b["image"].present) + jnp.sum(b["text"].value_or(0.0)))
        jax.block_until_ready(count(placed[0]))
        with expect_compiles(0):
            for batch in placed[1:]:
                jax.block_until_ready(count(batch))


class TestReadingMissingValues:
    def test_value_or_per_record_under_vmap(self) -> None:
        batch = _patterns()[2]
        image = batch["image"]
        corners = jax.vmap(lambda field: field.value_or(-1.0)[0, 0, 0])(image)
        expected = np.where(image.present, np.asarray(image.value)[:, 0, 0, 0], -1.0)
        np.testing.assert_array_equal(corners, expected)

    def test_a_scanned_chunk_equals_its_calls(self) -> None:
        batches = _patterns()
        chunk = batch_ops.stack(batches)

        def total(batch: Batch) -> jax.Array:
            return jnp.sum(batch["image"].value_or(0.0)) + jnp.sum(batch["text"].value_or(0.0))

        _, scanned = jax.jit(lambda c: jax.lax.scan(lambda s, b: (s, total(b)), None, c))(chunk)
        np.testing.assert_allclose(scanned, [total(b) for b in batches], rtol=1e-6)

    def test_a_fill_marks_what_it_imputed(self) -> None:
        batch = _patterns()[2]  # rows of the four children in turn
        out = CrossModalFill(rngs=nnx.Rngs(0))(batch)
        np.testing.assert_array_equal(out.states[IMPUTED]["text"], [0, 1, 0, 0] * 2)
        np.testing.assert_array_equal(out.states[IMPUTED]["image"], [0, 0, 1, 0] * 2)
        np.testing.assert_array_equal(out["image"].present, [1, 1, 1, 0] * 2)


class TestGradients:
    def test_gradients_match_a_float64_finite_difference_reference(self) -> None:
        batch = _patterns()[2]
        check_parameter_gradients(
            CrossModalFill(rngs=nnx.Rngs(0)), lambda model: _imputed_loss(model, batch)
        )

    def test_no_gradient_from_records_with_nothing_to_fill(self) -> None:
        """A record missing both fields, or holding both, fills nothing and passes no gradient."""
        neither = _patterns()[3]
        gradient = check_parameter_gradients(
            CrossModalFill(rngs=nnx.Rngs(0)),
            lambda model: _imputed_loss(model, neither),
            allow_zero=True,
        )
        assert all(not np.asarray(leaf).any() for leaf in jax.tree.leaves(gradient))
