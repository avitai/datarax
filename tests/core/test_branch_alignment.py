"""A per-record wrapper's branches line up datarax's typed state; a user entry is refused by name.

``apply_where`` (``jax.lax.cond``) and ``apply_selected`` (``jax.lax.switch``) pass a record's
data, state, weight and marks through every branch. ``cond`` and ``switch`` need every branch to
return one structure, and a child that writes an entry only some branches have gives two, which
jax refuses in its own words, naming no entry. A typed attribute has a neutral value:
the weight is always present, and a ``masked``/``imputed`` mark a branch does not write is False at
the writer's shape. A user ``state`` entry has none, so a branch-only one is refused, naming the
wrapper, the child and the entry. An entry no branch writes is the record's own array.

Writes are told from pass-throughs by array identity while tracing (``written_fields``), which
rests on jax forwarding an untouched array through ``cond``, ``switch``, ``vmap`` and ``scan``:
``TestJaxForwarding`` pins that, so a jax release that stops forwarding fails here, by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from datarax.core import batch_ops, operator as operator_module
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig


B, L = 4, 6


class _Writes(OperatorModule):
    """Writes one entry per record: the weight, a ``masked`` mark, or a user ``state`` entry."""

    calls = 0

    def __init__(self, what: str) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.what = what

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        type(self).calls += 1
        if self.what == "weight":
            return element.replace(weight=jnp.float32(0.5))
        if self.what == "masked":
            return element.replace(masked={**element.masked, "tokens": element.data["tokens"] > 2})
        return element.replace(state={**element.state, "score": jnp.float32(2.0)})


class _Keeps(OperatorModule):
    """Returns the record it is given."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig(stochastic=False))

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        return element


def _batch(size: int = B, **present: Any) -> Batch:
    """A batch of ``tokens`` records, with the entries ``present`` names already on it."""
    data = {"tokens": np.arange(size * L, dtype=np.int32).reshape(size, L) % 5}
    batch = batch_ops.from_arrays(data, states=present.get("states"))
    if "masked" in present:
        batch = batch.replace(masked=present["masked"])
    return batch


_WRAPPERS = {
    "Probabilistic (cond)": lambda what: ProbabilisticOperator(
        ProbabilisticOperatorConfig(probability=0.5), operator=_Writes(what), rngs=nnx.Rngs(0)
    ),
    "Selector (switch)": lambda what: SelectorOperator(
        SelectorOperatorConfig(), operators=[_Writes(what), _Keeps()], rngs=nnx.Rngs(0)
    ),
}

_MODES = {
    "eager": lambda op, batch: op(batch),
    "jax.jit": lambda op, batch: jax.jit(lambda b: op(b))(batch),
    "nnx.jit graph": lambda op, batch: nnx.jit(lambda o, b: o(b))(op, batch),
    "nnx.jit tree": lambda op, batch: nnx.jit(lambda o, b: o(b), graph=False)(op, batch),
}

_REFUSALS = {
    "Probabilistic (cond)": (
        "ProbabilisticOperator: child 0 (_Writes) writes state entry 'score' only where it "
        "applies, and the record has no 'score' where it does not. Give 'score' a value before "
        "the ProbabilisticOperator."
    ),
    "Selector (switch)": (
        "SelectorOperator: child 0 (_Writes) writes state entry 'score' only where it applies, "
        "and the record has no 'score' where it does not. Give 'score' a value before the "
        "SelectorOperator."
    ),
}


class TestABranchOnlyEntry:
    """An entry a child writes in one branch only, under both wrappers and four call modes."""

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize("wrapper", list(_WRAPPERS))
    def test_a_branch_only_weight_is_aligned(self, wrapper: str, mode: str) -> None:
        out = _MODES[mode](_WRAPPERS[wrapper]("weight"), _batch())
        assert out.weights.shape == (B,)
        assert set(np.asarray(out.weights).tolist()) <= {0.5, 1.0}

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize("wrapper", list(_WRAPPERS))
    def test_a_branch_only_mark_is_aligned(self, wrapper: str, mode: str) -> None:
        out = _MODES[mode](_WRAPPERS[wrapper]("masked"), _batch())
        assert out.masked["tokens"].shape == (B, L)
        assert out.masked["tokens"].dtype == jnp.bool_

    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize("wrapper", list(_WRAPPERS))
    def test_a_branch_only_user_entry_is_refused_by_name(self, wrapper: str, mode: str) -> None:
        with pytest.raises(TypeError) as caught:
            _MODES[mode](_WRAPPERS[wrapper]("score"), _batch())
        assert str(caught.value) == _REFUSALS[wrapper]

    @pytest.mark.parametrize("what", ["weight", "masked", "score"])
    @pytest.mark.parametrize("mode", list(_MODES))
    @pytest.mark.parametrize("wrapper", list(_WRAPPERS))
    def test_an_entry_the_record_already_has_passes(
        self, wrapper: str, mode: str, what: str
    ) -> None:
        present = {
            "weight": {},
            "masked": {"masked": {"tokens": np.zeros((B, L), bool)}},
            "score": {"states": {"score": np.zeros(B, np.float32)}},
        }[what]
        out = _MODES[mode](_WRAPPERS[wrapper](what), _batch(**present))
        assert out.batch_size == B


def _records(keep: list[bool], **present: Any) -> tuple[Batch, jax.Array]:
    return _batch(len(keep), **present), jnp.asarray(keep)


def _apply_where_rows(child: OperatorModule, batch: Batch, keep: jax.Array) -> Batch:
    """``apply_where`` over every row of ``batch``, applying ``child`` where ``keep`` holds."""

    def one(element: Element, applies: jax.Array) -> Element:
        return operator_module.apply_where(
            child, applies, element, None, owner="ProbabilisticOperator", position=0
        )

    rows = [batch_ops.element(batch, i) for i in range(batch.batch_size)]
    stacked = batch_ops.stack(rows)
    out = jax.vmap(one)(stacked, keep)
    return batch_ops.from_stacked(out)


class TestAlignment:
    """What each branch carries, by value."""

    def test_a_mark_is_false_where_its_writer_does_not_apply(self) -> None:
        batch, keep = _records([True, False, True, False])
        out = _apply_where_rows(_Writes("masked"), batch, keep)
        tokens = np.asarray(batch["tokens"])
        expected = np.where(np.asarray(keep)[:, None], tokens > 2, False)
        assert np.array_equal(np.asarray(out.masked["tokens"]), expected)

    def test_the_weight_is_the_record_s_where_its_writer_does_not_apply(self) -> None:
        batch, keep = _records([True, False, False, True])
        out = _apply_where_rows(_Writes("weight"), batch, keep)
        assert np.asarray(out.weights).tolist() == [0.5, 1.0, 1.0, 0.5]

    def test_an_entry_no_branch_writes_is_the_record_s_own_array(self) -> None:
        image = jnp.zeros((), bool)
        seen: dict[str, bool] = {}

        def one(element: Element, applies: jax.Array) -> Element:
            element = element.replace(masked={"image": image})
            out = operator_module.apply_where(
                _Writes("masked"), applies, element, None, owner="Probe", position=0
            )
            seen["own"] = out.masked["image"] is element.masked["image"]
            return out

        jax.make_jaxpr(one)(batch_ops.element(_batch(1), 0), jnp.bool_(True))
        assert seen["own"]

    def test_a_dropped_user_entry_is_refused_by_name(self) -> None:
        class _Drops(OperatorModule):
            def __init__(self) -> None:
                super().__init__(OperatorConfig(stochastic=False))

            def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
                return element.replace(state={})

        batch, keep = _records([True, False], states={"score": np.zeros(2, np.float32)})
        with pytest.raises(TypeError, match=r"child 0 \(_Drops\) drops state entry 'score'"):
            _apply_where_rows(_Drops(), batch, keep)

    def test_two_branches_marking_one_field_at_two_shapes_are_refused(self) -> None:
        class _MarksWhole(OperatorModule):
            def __init__(self) -> None:
                super().__init__(OperatorConfig(stochastic=False))

            def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
                return element.replace(masked={"tokens": jnp.bool_(True)})

        selector = SelectorOperator(
            SelectorOperatorConfig(), operators=[_Writes("masked"), _MarksWhole()], rngs=nnx.Rngs(0)
        )
        with pytest.raises(TypeError, match=r"marks 'tokens' in masked with shapes"):
            selector(_batch())


class TestEveryCaller:
    """All five callers align, and each names itself in the refusal."""

    @staticmethod
    def _composite(strategy: str, children: Sequence[OperatorModule]) -> OperatorModule:
        from datarax.operators.composite_operator import (
            CompositeOperatorConfig,
            CompositeOperatorModule,
            CompositionStrategy,
        )

        conditions = [lambda data: jnp.sum(data["tokens"]) > 10] * len(children)
        extra: dict[str, Any] = {}
        if strategy == "BRANCHING":
            extra["router"] = lambda data: jnp.int32(jnp.sum(data["tokens"]) % len(children))
        elif strategy == "CONDITIONAL_PARALLEL":
            extra["conditions"] = conditions
            extra["merge_strategy"] = "concat"
        else:
            extra["conditions"] = conditions
        config = CompositeOperatorConfig(strategy=CompositionStrategy[strategy], **extra)
        return CompositeOperatorModule(config, operators=children)

    @pytest.mark.parametrize(
        "strategy", ["BRANCHING", "CONDITIONAL_SEQUENTIAL", "CONDITIONAL_PARALLEL"]
    )
    def test_a_strategy_aligns_a_branch_only_mark(self, strategy: str) -> None:
        # CONDITIONAL_PARALLEL keeps its last child's state and marks, so the writer comes last
        # there; BRANCHING routes each record to one child.
        children = {
            "BRANCHING": [_Writes("masked"), _Keeps()],
            "CONDITIONAL_SEQUENTIAL": [_Writes("masked")],
            "CONDITIONAL_PARALLEL": [_Keeps(), _Writes("masked")],
        }[strategy]
        out = self._composite(strategy, children)(_batch())
        assert out.masked["tokens"].shape == (B, L)

    @pytest.mark.parametrize(
        "strategy", ["BRANCHING", "CONDITIONAL_SEQUENTIAL", "CONDITIONAL_PARALLEL"]
    )
    def test_a_strategy_names_itself_in_the_refusal(self, strategy: str) -> None:
        children = (
            [_Writes("score"), _Keeps()]
            if strategy != "CONDITIONAL_SEQUENTIAL"
            else [_Writes("score")]
        )
        with pytest.raises(TypeError, match=rf"^{strategy}: child 0 \(_Writes\) writes state"):
            self._composite(strategy, children)(_batch())


class TestWrittenFields:
    """Which entries a child wrote, by array identity while tracing."""

    def test_a_returned_array_is_no_write_and_a_new_one_is(self) -> None:
        def probe(a: jax.Array, b: jax.Array) -> tuple[Any, ...]:
            received = {"a": a, "group": {"b": b}}
            returned = {"a": a, "group": {"b": b * 1}, "c": a + 1}
            return tuple(sorted(operator_module.written_fields(received, returned)))

        found: list[Any] = []
        jax.make_jaxpr(lambda a, b: found.append(probe(a, b)) or a)(jnp.ones(2), jnp.ones(2))
        assert found == [(("c",), ("group", "b"))]

    def test_nothing_returned_unchanged_is_a_write(self) -> None:
        def probe(a: jax.Array) -> Any:
            received = {"a": a, "group": {"b": a}}
            return operator_module.written_fields(received, dict(received))

        found: list[Any] = []
        jax.make_jaxpr(lambda a: found.append(probe(a)) or a)(jnp.ones(2))
        assert found == [frozenset()]


class TestJaxForwarding:
    """jax forwards an untouched array through cond, switch, vmap and scan: written_fields' base."""

    @pytest.mark.parametrize("transform", ["cond", "switch", "vmap", "scan"])
    def test_an_untouched_operand_comes_back_as_itself_and_a_changed_one_does_not(
        self, transform: str
    ) -> None:
        def run(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
            if transform == "cond":
                return jax.lax.cond(True, lambda x, y: (x, y * 2), lambda x, y: (x, y), a, b)
            if transform == "switch":
                branches = [lambda x, y: (x, y * 2), lambda x, y: (x, y)]
                return jax.lax.switch(0, branches, a, b)
            if transform == "vmap":
                return jax.vmap(lambda x, y: (x, y * 2))(a, b)
            _, out = jax.lax.scan(lambda c, xy: (c, (xy[0], xy[1] * 2)), None, (a, b))
            return out

        seen: dict[str, bool] = {}

        def traced(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
            out = run(a, b)
            seen["untouched"] = out[0] is a
            seen["changed"] = out[1] is not b
            return out

        jax.make_jaxpr(traced)(jnp.ones(3), jnp.ones(3))
        assert seen == {"untouched": True, "changed": True}


@pytest.mark.parametrize("graph", [True, False])
class TestUnderTransforms:
    """The alignment traces with the wrapper, once, and is gone from a cached call."""

    def test_aligns_while_tracing_and_never_on_a_cached_call(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch).masked["tokens"]), graph=graph)
        op = _WRAPPERS["Selector (switch)"]("masked")
        _Writes.calls = 0
        with counter.expect(new_traces=1):
            for _ in range(6):
                apply(op, _batch())
        traced = _Writes.calls
        assert traced > 0
        with counter.expect(new_traces=1):  # a new batch size traces again, and aligns again
            apply(op, _batch(8))
        assert _Writes.calls == 2 * traced

    def test_rebuilt_wrappers_share_one_trace_and_one_graphdef(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch).masked["tokens"]), graph=graph)
        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(_WRAPPERS["Probabilistic (cond)"]("masked"), _batch())
        first = _WRAPPERS["Probabilistic (cond)"]("masked")
        second = _WRAPPERS["Probabilistic (cond)"]("masked")
        assert nnx.graphdef(first, graph=graph) == nnx.graphdef(second, graph=graph)

    def test_a_batch_carrying_another_mark_traces_again(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch).masked["tokens"]), graph=graph)
        op = _WRAPPERS["Probabilistic (cond)"]("masked")
        with counter.expect(new_traces=2):  # the different-spec control
            apply(op, _batch())
            apply(op, _batch(masked={"other": np.zeros(B, bool)}))

    def test_nnx_vmap_and_nnx_scan_align(self, graph: bool) -> None:
        op = _WRAPPERS["Selector (switch)"]("masked")
        stacked = batch_ops.stack([_batch(), _batch()])
        out = nnx.vmap(lambda o, b: o(b).masked["tokens"], in_axes=(None, 0), graph=graph)(
            op, stacked
        )
        assert out.shape == (2, B, L)

        def body(o: OperatorModule, total: jax.Array, b: Batch) -> jax.Array:
            return total + jnp.sum(o(b).masked["tokens"])

        scan = nnx.scan(body, in_axes=(None, nnx.Carry, 0), out_axes=nnx.Carry, graph=graph)
        assert scan(op, jnp.int32(0), stacked).shape == ()

    def test_gradients_flow_through_an_aligned_wrapper(self, graph: bool) -> None:
        class _Scales(OperatorModule):
            def __init__(self) -> None:
                super().__init__(OperatorConfig(stochastic=False))
                self.scale = nnx.Param(jnp.asarray(3.0))

            def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
                return element.replace(weight=element.weight * self.scale[...])

        selector = SelectorOperator(
            SelectorOperatorConfig(), operators=[_Scales(), _Scales()], rngs=nnx.Rngs(0)
        )

        # The selector holds an RNG key, which is no gradient target: split off the Params in the
        # mode under test (tree mode refuses a DiffState) and differentiate those.
        graphdef, params, rest = nnx.split(selector, nnx.Param, ..., graph=graph)

        def loss(params: Any, batch: Batch) -> jax.Array:
            return jnp.sum(nnx.merge(graphdef, params, rest)(batch).weights)

        grads = jax.jit(jax.grad(loss))(params, _batch())
        # Every record takes one child, which scales its weight 1 by that child's scale: the
        # loss is the sum of the chosen scales, so the two children's gradients sum to B.
        total = sum(float(leaf) for leaf in jax.tree.leaves(grads))
        assert total == pytest.approx(B)
