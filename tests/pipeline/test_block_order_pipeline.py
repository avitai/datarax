"""A pipeline built with a block order serves each epoch in that order, every record once.

``Pipeline(..., shuffle=True, block_order=BlockOrder(...))`` names each epoch's records with the
block order of the epoch's key (:mod:`datarax.core.block_order`): the host stage, its read threads
and the compiled session all name them through ``batch_records`` from the epoch plan, which carries
the order. A record keeps its draws, a resume is exact, and the resume fingerprint records the
order, so a state saved under another order is refused.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.core.block_order import BlockOrder
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, known_length, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.core.prng import NAMING_PRNG_IMPL
from datarax.operators.modality.image import BrightnessOperator, BrightnessOperatorConfig
from datarax.pipeline import Pipeline
from datarax.pipeline.epochs import EpochPlan
from datarax.pipeline.run_configuration import run_configuration
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.host_plans import reading_with


_ORDER = BlockOrder(block_size=3, blocks_per_buffer=2)
_LENGTH, _BATCH = 50, 5


def _memory(length: int = _LENGTH) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(), {"x": np.arange(length * 2, dtype=np.float32).reshape(length, 2)}
    )


def _pipeline(
    source: DataSourceModule | None = None,
    *,
    block_order: BlockOrder | None = _ORDER,
    shuffle: bool = True,
    num_epochs: int | None = 2,
    threads: int = 1,
    stages: list[nnx.Module] | None = None,
    batch_size: int = _BATCH,
) -> Pipeline:
    return Pipeline(
        source=_memory() if source is None else source,
        stages=[] if stages is None else stages,
        batch_size=batch_size,
        rngs=nnx.Rngs(7),
        shuffle=shuffle,
        num_epochs=num_epochs,
        block_order=block_order,
        host_resources=None if threads == 1 else reading_with(threads),
    )


def _indices(batches: Iterable[Batch]) -> list[int]:
    return [int(v) for batch in batches for v in from_words(np.asarray(batch.indices))]


def _epoch_names(pipe: Pipeline, epoch: int) -> list[int]:
    """The records of epoch ``epoch`` as the block order of the epoch's key names them."""
    base = jax.random.wrap_key_data(
        jnp.asarray(pipe._epoch_key_base.get_value()),  # noqa: SLF001 - the epoch key's base
        impl=NAMING_PRNG_IMPL,
    )
    key = jax.random.fold_in(base, epoch)
    length = known_length(pipe.source)
    assert length is not None
    names = pipe.source.record_indices_at(0, length, key, block=pipe.block_order)
    return [int(v) for v in from_words(np.asarray(names))]


class TestTheOrderServed:
    def test_each_epoch_is_the_block_order_of_its_key(self) -> None:
        pipe = _pipeline()
        served = _indices(pipe.raw_batches())
        assert served == _epoch_names(pipe, 0) + _epoch_names(pipe, 1)

    def test_every_record_once_per_epoch_and_epochs_differ(self) -> None:
        served = _indices(_pipeline().raw_batches())
        first, second = served[:_LENGTH], served[_LENGTH:]
        assert sorted(first) == sorted(second) == list(range(_LENGTH))
        assert first != second

    def test_batches_read_from_few_blocks(self) -> None:
        """A batch of 5 inside a buffer of 6 records reads from at most 2 blocks of 3."""
        served = np.asarray(_indices(_pipeline(num_epochs=1).raw_batches()))
        for start in range(0, _LENGTH, 6):
            assert len(np.unique(served[start : start + 6] // 3)) <= 2

    @pytest.mark.parametrize("threads", [2, 4])
    def test_read_threads_serve_the_same_names(self, threads: int) -> None:
        assert _indices(_pipeline(threads=threads).raw_batches()) == _indices(
            _pipeline().raw_batches()
        )

    def test_without_a_block_order_the_global_order_is_served(self) -> None:
        pipe = _pipeline(block_order=None)
        served = _indices(pipe.raw_batches())
        assert served != _indices(_pipeline().raw_batches())
        assert sorted(served[:_LENGTH]) == list(range(_LENGTH))

    def test_the_compiled_session_names_what_the_host_stage_names(self) -> None:
        host = _indices(itertools.islice(_pipeline(num_epochs=None).raw_batches(), 12))
        stepped = _pipeline(num_epochs=None)
        steps = [stepped.step() for _ in range(12)]
        assert _indices(iter(steps)) == host
        scanned = _pipeline(num_epochs=None)
        indices = scanned.scan(lambda batch: batch.indices, length=12)
        assert [int(v) for v in from_words(np.asarray(indices).reshape(-1, 2))] == host


class TestResume:
    @pytest.mark.parametrize("threads", [1, 2, 4])
    @pytest.mark.parametrize("cut", [1, 7, 13])
    def test_a_resume_after_any_batch_is_exact(self, threads: int, cut: int) -> None:
        whole = _indices(_pipeline().raw_batches())
        pipe = _pipeline(threads=threads)
        batches = iter(pipe.raw_batches())
        head = [next(batches) for _ in range(cut)]
        state = pipe.get_state()
        pipe.close()
        resumed = _pipeline(threads=threads)
        resumed.set_state(state)
        assert _indices(iter(head)) + _indices(resumed.raw_batches()) == whole

    def test_the_fingerprint_records_the_block_order(self) -> None:
        assert run_configuration(_pipeline())["order"] == {
            "kind": "block",
            "block_size": 3,
            "blocks_per_buffer": 2,
        }
        assert run_configuration(_pipeline(block_order=None))["order"] == {"kind": "global"}

    @pytest.mark.parametrize(
        ("saved", "restored"),
        [
            (None, _ORDER),
            (_ORDER, None),
            (_ORDER, BlockOrder(block_size=4, blocks_per_buffer=2)),
            (_ORDER, BlockOrder(block_size=3, blocks_per_buffer=3)),
        ],
        ids=["global-into-block", "block-into-global", "other-block-size", "other-buffer"],
    )
    def test_a_state_saved_under_another_order_is_refused(
        self, saved: BlockOrder | None, restored: BlockOrder | None
    ) -> None:
        pipe = _pipeline(block_order=saved)
        next(iter(pipe.raw_batches()))
        state = pipe.get_state()
        pipe.close()
        with pytest.raises(ValueError, match="order"):
            _pipeline(block_order=restored).set_state(state)


def _brightness() -> BrightnessOperator:
    return BrightnessOperator(
        BrightnessOperatorConfig(
            brightness_range=(-0.5, 0.5), field_key="x", stochastic=True, stream_name="b"
        ),
        rngs=nnx.Rngs(b=3),
    )


def _per_record(pipe: Pipeline) -> dict[tuple[int, int], bytes]:
    out = {}
    for batch in pipe:
        names = from_words(np.asarray(batch.indices))
        for row, (name, epoch) in enumerate(zip(names, np.asarray(batch.epochs), strict=True)):
            out[(int(epoch), int(name))] = np.asarray(batch["x"][row]).tobytes()
    return out


def test_a_record_s_draws_do_not_depend_on_the_order() -> None:
    """A record's key is built from its index and epoch, so reordering moves it, not its draws."""
    by_block = _per_record(_pipeline(stages=[_brightness()]))
    by_global = _per_record(_pipeline(block_order=None, stages=[_brightness()]))
    assert by_block == by_global
    assert len(by_block) == 2 * _LENGTH


class _Stream(DataSourceModule):
    """A stream with a length: indices it reports are ids, so no order applies to them."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: a stream's ids."""
        return RecordIdentity.STREAM_IDS

    def __init__(self) -> None:
        super().__init__(StructuralConfig())

    def __len__(self) -> int:
        return 10


class _Identity(nnx.Module):
    def __call__(self, batch: Batch) -> Batch:
        return batch


class TestRefusals:
    def test_a_block_order_without_shuffle_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"block_order.*shuffle"):
            _pipeline(shuffle=False)

    def test_a_block_order_over_a_stream_is_refused(self) -> None:
        with pytest.raises(ValueError, match="INDEXED"):
            _pipeline(_Stream())

    def test_an_epoch_plan_without_a_length_refuses_a_block_order(self) -> None:
        with pytest.raises(ValueError, match="length"):
            EpochPlan(
                length=None, batch_size=4, drop_last=False, num_epochs=None, block_order=_ORDER
            )

    @pytest.mark.parametrize("build", ["from_dag", "from_arrays"])
    def test_every_constructor_takes_the_block_order(self, build: str) -> None:
        if build == "from_dag":
            pipe = Pipeline.from_dag(
                source=_memory(),
                nodes={"a": _Identity()},
                edges={"a": []},
                sink="a",
                batch_size=_BATCH,
                rngs=nnx.Rngs(7),
                shuffle=True,
                block_order=_ORDER,
            )
        else:
            pipe = Pipeline.from_arrays(
                {"x": np.arange(_LENGTH, dtype=np.float32)},
                batch_size=_BATCH,
                seed=7,
                shuffle=True,
                block_order=_ORDER,
            )
        assert pipe.epoch_plan.block_order == _ORDER


class TestCompiles:
    def test_pipelines_built_alike_compile_the_block_naming_once(self) -> None:
        """One naming program per structure and plan; the global order's plan is another."""

        def run(order: BlockOrder | None) -> None:
            pipe = _pipeline(block_order=order, batch_size=8)
            list(pipe.raw_batches())
            pipe.close()

        jax.clear_caches()
        with compiled_programs() as programs:
            run(_ORDER)
            run(_ORDER)
            run(BlockOrder(block_size=3, blocks_per_buffer=2))
        # 100 records in batches of 8: the full batches' program and the short final batch's.
        assert sum(str(program).startswith("jit(_names") for program in programs) == 2
        with compiled_programs() as programs:
            run(None)
        assert sum(str(program).startswith("jit(_names") for program in programs) == 2
