"""A name datarax chose never replaces a field without the user's say-so (W1-W4, W8).

An operator declares the fields it creates (``created_fields()``, empty by default). The final
``OperatorModule.__call__`` refuses a declared create the incoming record already has, naming the
field, its spec, and both fixes: another output, or ``overwrite=True``. A name the user typed, as
a ``target_key``, replaces. Inside a DAG the refusal adds the stage and the field's first writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from datarax.core import batch_ops
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.field_paths import FieldCreateError, FieldName, set_field
from datarax.core.operator import CreatedField, OperatorModule
from datarax.operators.modality.image import BrightnessOperator, BrightnessOperatorConfig
from datarax.pipeline.dag import OperatorDag
from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


@dataclass(frozen=True)
class _CreatesConfig(OperatorConfig):
    output: FieldName = "f0_hz"
    overwrite: bool = False


class _Creates(OperatorModule):
    """Creates ``output`` from ``x``, declaring it unless ``overwrite`` is set."""

    calls = 0

    def __init__(self, config: _CreatesConfig) -> None:
        super().__init__(config)
        self.config: _CreatesConfig = config
        self.scale = nnx.Param(jnp.asarray(2.0))

    def created_fields(self) -> tuple[CreatedField, ...]:
        type(self).calls += 1
        if self.config.overwrite:
            return ()
        return (CreatedField(self.config.output, "output"),)

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        value = jnp.broadcast_to(element.data["x"] * self.scale[...], (63,))
        return element.replace(data=set_field(element.data, self.config.output, value))


class _AddF0(OperatorModule):
    """Writes ``f0_hz`` with no declaration, as user code does."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig())

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        return element.update_data({"f0_hz": jnp.broadcast_to(element.data["x"], (63,))})


def _batch(size: int = 2, **extra: Any) -> Batch:
    return batch_ops.from_arrays({"x": jnp.arange(size, dtype=jnp.float32), **extra})


def _creates(**options: Any) -> _Creates:
    return _Creates(_CreatesConfig(**options))


FIXES = (
    "Creating it would replace that value. To keep both, name another output with "
    "_CreatesConfig(output=...). To replace it, pass _CreatesConfig(overwrite=True)."
)


class TestTheDefault:
    def test_an_operator_declares_no_creates_by_default(self) -> None:
        class _Plain(OperatorModule):
            def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
                return element

        assert _Plain(OperatorConfig()).created_fields() == ()

    def test_a_declared_create_of_a_new_name_is_served(self) -> None:
        out = _creates()(_batch())
        assert out["f0_hz"].shape == (2, 63)


class TestARefusedCreate:
    def test_a_create_onto_an_existing_field_names_it_its_spec_and_both_fixes(self) -> None:
        batch = _batch(f0_hz=jnp.zeros((2, 63), jnp.float32))
        with pytest.raises(FieldCreateError) as caught:
            _creates()(batch)
        assert str(caught.value) == (
            "_Creates creates the field 'f0_hz', and the record already has 'f0_hz' "
            f"(float32[63]). {FIXES}"
        )

    def test_a_create_onto_a_group_names_its_fields(self) -> None:
        meta = {"mask": jnp.zeros(2, bool), "id": jnp.zeros(2, jnp.int32)}  # listed in name order
        with pytest.raises(FieldCreateError) as caught:
            _creates(output="meta")(_batch(meta=meta))
        assert str(caught.value) == (
            "_Creates creates the field 'meta', and the record already has 'meta' as a group of 2 "
            "fields ('meta/id', 'meta/mask'). Creating it would replace the whole group. To keep "
            "both, name another output with _CreatesConfig(output=...). To replace it, pass "
            "_CreatesConfig(overwrite=True)."
        )

    def test_overwrite_replaces(self) -> None:
        batch = _batch(f0_hz=jnp.zeros((2, 63), jnp.float32))
        out = _creates(overwrite=True)(batch)
        np.testing.assert_array_equal(out["f0_hz"][1], np.full(63, 2.0, np.float32))

    def test_a_name_the_user_typed_replaces(self) -> None:
        """W1: a ``target_key`` is the user's own name, so writing onto it is a replace."""
        op = BrightnessOperator(
            BrightnessOperatorConfig(field_key="image", target_key="other", brightness_delta=0.5)
        )
        batch = batch_ops.from_arrays(
            {"image": jnp.ones((2, 4, 4, 3)), "other": jnp.zeros((2, 4, 4, 3))}
        )
        assert float(op(batch)["other"].sum()) != 0.0


class TestInADag:
    """W8: the stage and the field's first writer, looked up only when the create is refused."""

    def test_the_refusal_names_the_stage_and_the_stage_that_wrote_the_field(self) -> None:
        dag = OperatorDag.from_stages([_AddF0(), _creates(output="g"), _creates()])
        with pytest.raises(FieldCreateError) as caught:
            dag(_batch())
        assert str(caught.value) == (
            "stage 'stage_2' (_Creates) creates the field 'f0_hz', and the record already has "
            f"'f0_hz' (float32[63]), written by stage 'stage_0' (_AddF0). {FIXES}"
        )

    @pytest.mark.parametrize("path", ["step", "iteration", "scan", "session"])
    def test_in_a_pipeline_a_field_the_source_holds_is_written_by_the_source(
        self, path: str
    ) -> None:
        data = {"x": np.arange(4, dtype=np.float32), "f0_hz": np.zeros((4, 63), np.float32)}
        pipe = Pipeline(
            source=MemorySource(MemorySourceConfig(), data),
            stages=[_creates()],
            batch_size=2,
            rngs=nnx.Rngs(0),
        )
        try:
            with pytest.raises(FieldCreateError) as caught:
                if path == "step":
                    pipe.step()
                elif path == "iteration":
                    next(iter(pipe))
                elif path == "scan":
                    pipe.scan(lambda batch: batch["x"].sum(), length=2)
                else:
                    next(iter(pipe.session()))
        finally:
            pipe.close()  # the kept traceback holds the run open; close its read threads
        assert str(caught.value) == (
            "stage 'stage_0' (_Creates) creates the field 'f0_hz', and the record already has "
            f"'f0_hz' (float32[63]), written by the source (MemorySource). {FIXES}"
        )


@pytest.mark.parametrize("graph", [True, False])
class TestUnderTransforms:
    def test_the_check_runs_while_tracing_and_never_on_a_cached_call(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch)["f0_hz"]), graph=graph)
        op = _creates()
        _Creates.calls = 0
        with counter.expect(new_traces=1):
            for _ in range(6):
                apply(op, _batch())
        assert _Creates.calls == 1
        with counter.expect(new_traces=1):
            apply(op, _batch(4))
        assert _Creates.calls == 2

    def test_rebuilt_operators_with_equal_configs_share_one_trace(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch)["f0_hz"]), graph=graph)
        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(_creates(), _batch())
        assert nnx.graphdef(_creates(), graph=graph) == nnx.graphdef(_creates(), graph=graph)

    def test_the_refusal_under_jit_is_the_eager_one(self, graph: bool) -> None:
        batch = _batch(f0_hz=jnp.zeros((2, 63), jnp.float32))
        with pytest.raises(FieldCreateError) as eager:
            _creates()(batch)
        with pytest.raises(FieldCreateError) as jitted:
            nnx.jit(lambda op, batch: op(batch), graph=graph)(_creates(), batch)
        assert str(eager.value) == str(jitted.value)

    def test_gradients_flow_through_a_declared_create(self, graph: bool) -> None:
        def loss(op: _Creates, batch: Batch) -> jax.Array:
            return op(batch)["f0_hz"].sum()

        grads = nnx.jit(nnx.grad(loss, graph=graph), graph=graph)(_creates(), _batch())
        assert float(grads.scale[...]) == pytest.approx(63.0)

    def test_nnx_vmap_over_k_batches_checks_once(self, graph: bool) -> None:
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch())
        _Creates.calls = 0
        out = nnx.vmap(lambda op, batch: op(batch)["f0_hz"], in_axes=(None, 0), graph=graph)(
            _creates(), stacked
        )
        assert out.shape == (2, 2, 63)
        assert _Creates.calls == 1

    def test_nnx_scan_over_a_chunk_checks_once(self, graph: bool) -> None:
        chunk = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch(), _batch())

        def body(op: _Creates, total: jax.Array, batch: Batch) -> jax.Array:
            return total + op(batch)["f0_hz"].sum()

        _Creates.calls = 0
        scan = nnx.scan(body, in_axes=(None, nnx.Carry, 0), out_axes=nnx.Carry, graph=graph)
        total = scan(_creates(), jnp.float32(0), chunk)
        assert float(total) == pytest.approx(3 * 63 * 2.0)  # x = [0, 1] scaled by 2
        assert _Creates.calls == 1

    def test_an_eager_call_checks_every_call(self, graph: bool) -> None:
        op = _creates()
        _Creates.calls = 0
        for _ in range(3):
            op(_batch())
        assert _Creates.calls == 3


class TestUnderJaxTransforms:
    """A split operator under plain JAX: the check traces with it and is gone from the program."""

    def test_jax_jit_over_a_static_graphdef_traces_once_and_checks_while_tracing(self) -> None:
        graphdef, state = nnx.split(_creates())
        counter = TraceCounter()

        @jax.jit
        @counter.wrap
        def apply(state: Any, batch: Batch) -> jax.Array:
            return nnx.merge(graphdef, state)(batch)["f0_hz"]

        _Creates.calls = 0
        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(state, _batch())
            for _ in range(3):
                apply(nnx.split(_creates())[1], _batch())
        assert _Creates.calls == 1
        with counter.expect(new_traces=1):  # a new batch size traces again, and checks again
            apply(state, _batch(4))
        assert _Creates.calls == 2

    def test_jax_jit_with_a_closure_checks_only_while_tracing(self) -> None:
        op = _creates()
        _Creates.calls = 0
        apply = jax.jit(lambda batch: op(batch)["f0_hz"])
        for _ in range(3):
            apply(_batch())
        assert _Creates.calls == 1

    def test_vmap_and_lax_scan_check_once(self) -> None:
        graphdef, state = nnx.split(_creates())
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch())
        _Creates.calls = 0
        out = jax.jit(jax.vmap(lambda batch: nnx.merge(graphdef, state)(batch)["f0_hz"]))(stacked)
        assert out.shape == (2, 2, 63)
        assert _Creates.calls == 1

        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, None]:
            return total + nnx.merge(graphdef, state)(batch)["f0_hz"].sum(), None

        total, _ = jax.jit(lambda chunk: jax.lax.scan(body, jnp.float32(0), chunk))(stacked)
        assert float(total) == pytest.approx(2 * 63 * 2.0)
        assert _Creates.calls == 2

    def test_jax_grad_values_are_the_closed_form(self) -> None:
        graphdef, params, rest = nnx.split(_creates(), nnx.Param, ...)

        def loss(params: Any, batch: Batch) -> jax.Array:
            return nnx.merge(graphdef, params, rest)(batch)["f0_hz"].sum()

        grads = jax.jit(jax.grad(loss))(params, _batch())
        assert float(grads["scale"][...]) == pytest.approx(63.0)
