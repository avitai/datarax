"""The pipeline owns the shuffle: ``Pipeline(shuffle=...)`` decides the order, sources apply it.

A source's ``record_indices_at(start, size, key)`` serves the order ``key`` selects, and the
sequential order when ``key`` is ``None``; the pipeline passes its epoch key exactly when it was
built with ``shuffle=True``. Moving the flag from the source to the pipeline changes no order:
the fixture holds the records every path served (iteration, ``step()``, ``scan``) on datarax
``611bf97``, when the flag lived on ``MemorySourceConfig``, and the same pipelines built the new
way serve them bit for bit. Its provenance is stored in the file.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.pipeline.pipeline import Pipeline
from datarax.sources import source_ops
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.source_ops import resolve_wrapped_indices
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig
from tests.test_common.compiles import expect_first_call_compiles


_ORACLE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pipeline"
    / "served_records_before_the_shuffle_moved.npz"
)
_PATHS = ("iter", "step", "scan")
_MIX_LENGTHS = (5, 9)


def _columns(length: int) -> dict[str, np.ndarray]:
    return {"x": np.arange(length, dtype=np.float32)}


def _oracle_cases() -> list[str]:
    with np.load(_ORACLE) as oracle:
        return [str(case) for case in oracle["cases"]]


@dataclass(frozen=True)
class _Case:
    """One recorded case: ``family|shuffle|drop_last|length|batch_size|seed``."""

    family: str
    shuffle: bool
    drop_last: bool
    length: int
    batch_size: int
    seed: int

    @classmethod
    def parse(cls, name: str) -> _Case:
        family, shuffle, drop_last, length, batch_size, seed = name.split("|")
        return cls(
            family, shuffle == "1", drop_last == "1", int(length), int(batch_size), int(seed)
        )


def _build(case: _Case, tmp_path: Path) -> Pipeline:
    """The case's pipeline as it is built now: the shuffle flag on the pipeline."""
    if case.family == "arrays":
        return Pipeline.from_arrays(
            _columns(case.length),
            seed=case.seed,
            shuffle=case.shuffle,
            batch_size=case.batch_size,
            drop_last=case.drop_last,
            num_epochs=3,
        )
    if case.family == "memory":
        source: DataSourceModule = MemorySource(MemorySourceConfig(), data=_columns(case.length))
    elif case.family == "mix":
        children = [MemorySource(MemorySourceConfig(), data=_columns(n)) for n in _MIX_LENGTHS]
        source = MixDataSourcesNode(
            MixDataSourcesConfig(num_sources=2, weights=(0.3, 0.7)),
            children,
            rngs=nnx.Rngs(case.seed),
        )
    else:
        path = tmp_path / "records.npy"
        np.save(path, np.arange(case.length, dtype=np.float32))
        source = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path)))
    return Pipeline(
        source=source,
        stages=[],
        rngs=nnx.Rngs(case.seed),
        shuffle=case.shuffle,
        batch_size=case.batch_size,
        drop_last=case.drop_last,
        num_epochs=3,
    )


def _concatenated(batches: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "indices": np.concatenate([indices for indices, _ in batches]).astype(np.uint64),
        "epochs": np.concatenate([epochs for _, epochs in batches]).astype(np.int32),
        "sizes": np.asarray([len(indices) for indices, _ in batches], np.int32),
    }


def _named(batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    return from_words(np.asarray(batch.indices)), np.asarray(batch.epochs)


def _scan_fn(batch: Batch) -> tuple[jax.Array, jax.Array]:
    return jnp.asarray(batch.indices), jnp.asarray(batch.epochs)


def _served(make: Callable[[], Pipeline]) -> dict[str, dict[str, np.ndarray]]:
    pipe = make()
    count = len(pipe)
    served = {"iter": _concatenated([_named(batch) for batch in pipe])}
    pipe = make()
    served["step"] = _concatenated([_named(pipe.step()) for _ in range(count)])
    indices, epochs = make().scan(_scan_fn, length=count)
    served["scan"] = _concatenated(
        [(from_words(np.asarray(indices[k])), np.asarray(epochs[k])) for k in range(count)]
    )
    return served


def test_the_oracle_was_recorded_where_the_flag_lived_on_the_source() -> None:
    with np.load(_ORACLE) as oracle:
        provenance = json.loads(str(oracle["provenance"]))
    assert provenance["datarax_revision"].startswith("611bf97")
    assert provenance["worktree_clean"] is True
    assert provenance["x64"] is False
    families = {case.split("|")[0] for case in _oracle_cases()}
    assert families == {"memory", "arrays", "mix", "disk"}


@pytest.mark.parametrize("name", _oracle_cases())
def test_every_path_serves_the_records_recorded_before_the_flag_moved(
    name: str, tmp_path: Path
) -> None:
    """OWN-0930-BITID: orders below ``2**31`` stay bit-identical through moving the flag."""
    case = _Case.parse(name)
    served = _served(lambda: _build(case, tmp_path))
    with np.load(_ORACLE) as oracle:
        for path in _PATHS:
            for field in ("indices", "epochs", "sizes"):
                np.testing.assert_array_equal(
                    served[path][field], oracle[f"{name}|{path}|{field}"], err_msg=f"{path} {field}"
                )


class TestTheSourceKeepsNoShuffle:
    """One owner: no source config, factory or source carries a shuffle flag or seed."""

    def test_memory_source_config_refuses_shuffle(self) -> None:
        with pytest.raises(TypeError, match="shuffle"):
            MemorySourceConfig(shuffle=True)  # type: ignore[call-arg]

    def test_eager_configs_refuse_shuffle_and_seed(self) -> None:
        from datarax.sources.hf_source import HFEagerConfig
        from datarax.sources.tfds_source import TFDSEagerConfig

        with pytest.raises(TypeError, match="seed"):
            TFDSEagerConfig(name="mnist", split="train", seed=1)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="shuffle"):
            TFDSEagerConfig(name="mnist", split="train", shuffle=True)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="shuffle"):
            HFEagerConfig(name="mnist", split="train", shuffle=True)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="seed"):
            HFEagerConfig(name="mnist", split="train", seed=1)  # type: ignore[call-arg]

    @pytest.mark.parametrize("factory", ["from_tfds", "from_hf"])
    def test_factories_take_no_shuffle_or_seed(self, factory: str) -> None:
        import datarax.sources as sources

        parameters = inspect.signature(getattr(sources, factory)).parameters
        assert "shuffle" not in parameters
        assert "seed" not in parameters

    def test_in_memory_sources_take_no_rngs(self) -> None:
        """Nothing in an in-memory source is random, so it takes no ``rngs``."""
        from datarax.sources.hf_source import HFEagerSource
        from datarax.sources.tfds_source import TFDSEagerSource

        for source_class in (MemorySource, TFDSEagerSource, HFEagerSource):
            assert "rngs" not in inspect.signature(source_class.__init__).parameters
        source = MemorySource(MemorySourceConfig(), _columns(4))
        assert source.rngs is None

    def test_no_source_reports_or_sets_a_random_order(self) -> None:
        from datarax.sources._source_base import EagerSourceBase

        for source_class in (MemorySource, EagerSourceBase):
            for name in ("is_random_order", "set_random_order", "_host_shuffle_seed"):
                assert not hasattr(source_class, name), f"{source_class.__name__}.{name}"

    def test_the_position_shuffle_helper_is_gone(self) -> None:
        assert not hasattr(source_ops, "shuffled_index_for_position")
        assert not hasattr(source_ops, "validate_seed_range")

    def test_resolve_wrapped_indices_shuffles_iff_given_a_key(self) -> None:
        assert "is_random_order" not in inspect.signature(resolve_wrapped_indices).parameters
        key = jax.random.key(3)
        ordered = resolve_wrapped_indices(0, 10, 10, None)
        shuffled = resolve_wrapped_indices(0, 10, 10, key)
        np.testing.assert_array_equal(from_words(ordered), np.arange(10, dtype=np.uint64))
        np.testing.assert_array_equal(np.sort(from_words(shuffled)), np.arange(10, dtype=np.uint64))
        assert not np.array_equal(from_words(shuffled), np.arange(10))


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Sized(DataSourceModule):
    """A source with a length and nothing else: the default ``record_indices_at``."""

    def __init__(self, length: int) -> None:
        super().__init__(_Config())
        self.rows = length

    def __len__(self) -> int:
        return self.rows


class _Unsized(DataSourceModule):
    """A source without a length: sequential positions only."""

    def __init__(self) -> None:
        super().__init__(_Config())


def _memory(length: int = 23, *, workers: int = 1, shard: int | None = None) -> MemorySource:
    return MemorySource(MemorySourceConfig(num_workers=workers, shard_id=shard), _columns(length))


def _mixed() -> MixDataSourcesNode:
    return MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)), [_memory(9), _memory(14)]
    )


_KEYED_SOURCES: dict[str, Callable[[], DataSourceModule]] = {
    "default": lambda: _Sized(23),
    "memory": _memory,
    "memory-worker": lambda: _memory(workers=3, shard=1),
    "mixed": _mixed,
}
_ORDERED_SOURCES = {name: make for name, make in _KEYED_SOURCES.items() if name != "mixed"}


class TestTheKeySelectsTheOrder:
    """``record_indices_at(start, size, None)`` is sequential; with a key, the keyed order."""

    @pytest.mark.parametrize("name", sorted(_ORDERED_SOURCES))
    def test_without_a_key_the_order_is_sequential(self, name: str) -> None:
        source = _ORDERED_SOURCES[name]()
        served = from_words(source.record_indices_at(20, 6, None))
        expected = from_words(resolve_wrapped_indices(20, 6, 23, None, **_layout(name)))
        np.testing.assert_array_equal(served, expected)

    @pytest.mark.parametrize("name", sorted(_ORDERED_SOURCES))
    def test_with_a_key_the_order_is_the_keyed_permutation(self, name: str) -> None:
        source = _ORDERED_SOURCES[name]()
        key = jax.random.key(11)
        served = from_words(source.record_indices_at(20, 6, key))
        expected = from_words(resolve_wrapped_indices(20, 6, 23, key, **_layout(name)))
        np.testing.assert_array_equal(served, expected)
        assert not np.array_equal(served, from_words(source.record_indices_at(20, 6, None)))

    def test_a_source_without_a_length_has_no_order_to_shuffle(self) -> None:
        source = _Unsized()
        np.testing.assert_array_equal(
            from_words(source.record_indices_at(5, 3, None)), np.asarray([5, 6, 7], np.uint64)
        )
        with pytest.raises(ValueError, match="no length"):
            source.record_indices_at(5, 3, jax.random.key(0))

    def test_streaming_disk_shuffles_when_the_pipeline_does(self, tmp_path: Path) -> None:
        """A capability it lacked while the flag lived on source configs."""
        path = tmp_path / "records.npy"
        np.save(path, np.arange(10, dtype=np.float32))

        def epochs(shuffle: bool) -> list[np.ndarray]:
            pipe = Pipeline(
                source=StreamingDiskSource(StreamingDiskSourceConfig(path=str(path))),
                stages=[],
                batch_size=5,
                num_epochs=2,
                rngs=nnx.Rngs(2),
                shuffle=shuffle,
            )
            served = np.concatenate([_named(batch)[0] for batch in pipe])
            return [served[:10], served[10:]]

        for epoch in epochs(False):
            np.testing.assert_array_equal(epoch, np.arange(10, dtype=np.uint64))
        shuffled = epochs(True)
        for epoch in shuffled:
            np.testing.assert_array_equal(np.sort(epoch), np.arange(10, dtype=np.uint64))
        assert not np.array_equal(shuffled[0], np.arange(10))
        assert not np.array_equal(shuffled[0], shuffled[1])

    def test_a_pipeline_over_mix_without_shuffle_is_refused_at_its_first_pull(self) -> None:
        pipe = Pipeline(source=_mixed(), stages=[], batch_size=4, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError, match=r"Pipeline\(shuffle=True\)"):
            next(iter(pipe))


def _layout(name: str) -> dict[str, int]:
    return {"num_workers": 3, "shard_id": 1} if name == "memory-worker" else {}


class TestRecordIndicesUnderTransforms:
    """Section 7: ``record_indices_at`` with and without a key under jit, vmap and scan."""

    @pytest.mark.parametrize("keyed", [False, True], ids=["ordered", "keyed"])
    @pytest.mark.parametrize("name", sorted(_KEYED_SOURCES))
    def test_one_compile_per_layout_serves_every_start(self, name: str, keyed: bool) -> None:
        if name == "mixed" and not keyed:
            pytest.skip("a mix draws from the key whatever the order (C5c)")
        source = _KEYED_SOURCES[name]()
        starts = (0, 5, 17, 40)
        keys = [jax.random.key(seed) if keyed else None for seed in range(len(starts))]
        expected = [source.record_indices_at(s, 6, k) for s, k in zip(starts, keys, strict=True)]
        if keyed:
            names = jax.jit(lambda start, key: source.record_indices_at(start, 6, key))
            arguments = [(jnp.int32(s), k) for s, k in zip(starts, keys, strict=True)]
        else:
            names = jax.jit(lambda start: source.record_indices_at(start, 6, None))
            arguments = [(jnp.int32(s),) for s in starts]
        with expect_compiles(1):
            jax.block_until_ready(names(*arguments[0]))
        with expect_compiles(0):
            served = [names(*argument) for argument in arguments]
        for got, want in zip(served, expected, strict=True):
            assert got.dtype == jnp.uint32
            assert got.shape == (6, 2)
            np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("name", sorted(_KEYED_SOURCES))
    def test_vmap_over_keys_equals_the_loop_over_keys(self, name: str) -> None:
        """The pipeline names a crossing batch by vmapping over its epochs' keys."""
        source = _KEYED_SOURCES[name]()
        keys = jax.random.split(jax.random.key(4), 3)
        starts = jnp.asarray([0, 0, 7], jnp.int32)
        batched = jax.jit(jax.vmap(lambda s, k: source.record_indices_at(s, 6, k)))(starts, keys)
        for row, start, key in zip(batched, starts, keys, strict=True):
            np.testing.assert_array_equal(row, source.record_indices_at(int(start), 6, key))

    @pytest.mark.parametrize("keyed", [False, True], ids=["ordered", "keyed"])
    @pytest.mark.parametrize("name", sorted(_KEYED_SOURCES))
    def test_scan_over_starts_equals_the_loop(self, name: str, keyed: bool) -> None:
        if name == "mixed" and not keyed:
            pytest.skip("a mix draws from the key whatever the order (C5c)")
        source = _KEYED_SOURCES[name]()
        key = jax.random.key(5) if keyed else None
        starts = jnp.arange(0, 30, 6, dtype=jnp.int32)

        @jax.jit
        def scanned(starts: jax.Array) -> jax.Array:
            def body(carry: None, start: jax.Array) -> tuple[None, jax.Array]:
                return carry, source.record_indices_at(start, 6, key)

            return jax.lax.scan(body, None, starts)[1]

        served = scanned(starts)
        for step, start in enumerate(np.asarray(starts)):
            np.testing.assert_array_equal(
                served[step], source.record_indices_at(int(start), 6, key)
            )


def _pipeline(*, shuffle: bool, drop_last: bool, length: int, batch: int) -> Pipeline:
    return Pipeline(
        source=MemorySource(MemorySourceConfig(), _columns(length)),
        stages=[],
        batch_size=batch,
        num_epochs=2,
        drop_last=drop_last,
        rngs=nnx.Rngs(0),
        shuffle=shuffle,
    )


# (length, batch): 12/4 never crosses an epoch; 10/4 crosses under drop_last=False.
_LAYOUTS = [(12, 4), (10, 4)]


@pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
@pytest.mark.parametrize("drop_last", [False, True], ids=["keep", "drop"])
@pytest.mark.parametrize(("length", "batch"), _LAYOUTS, ids=["aligned", "crossing"])
class TestThePipelineUnderTransforms:
    """Section 7: the pipeline's compiled paths, per (shuffle, drop_last, crossing)."""

    def test_a_session_compiles_once_and_once_more_for_a_short_final_batch(
        self, shuffle: bool, drop_last: bool, length: int, batch: int
    ) -> None:
        pipe = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        batches = len(pipe)
        short = int(not drop_last and (2 * length) % batch != 0)
        session = iter(pipe)
        with expect_first_call_compiles("jit(session_step)"):
            jax.block_until_ready(next(session).indices)
        with expect_compiles(0):
            for _ in range(batches - 1 - short):
                jax.block_until_ready(next(session).indices)
        with expect_compiles(short):
            remaining = [jax.block_until_ready(b.indices) for b in session]
        assert len(remaining) == short

    def test_a_jitted_step_compiles_once(
        self, shuffle: bool, drop_last: bool, length: int, batch: int
    ) -> None:
        pipe = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        step = nnx.jit(lambda p: p.step())
        with expect_compiles(1):
            first = step(pipe)
        with expect_compiles(0):
            rest = [step(pipe) for _ in range(3)]
        reference = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        expected = [_named(reference.step()) for _ in range(4)]
        for got, want in zip([first, *rest], expected, strict=True):
            np.testing.assert_array_equal(_named(got)[0], want[0])
            np.testing.assert_array_equal(_named(got)[1], want[1])

    def test_scan_compiles_once_and_equals_its_steps(
        self, shuffle: bool, drop_last: bool, length: int, batch: int
    ) -> None:
        pipe = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        # The scan builds its program and, eagerly, the step numbers it scans over.
        with expect_first_call_compiles("jit(iota)", "jit(scan)"):
            first = pipe.scan(_scan_fn, length=3)
        with expect_compiles(0):
            second = pipe.scan(_scan_fn, length=3)
        reference = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        expected = [_named(reference.step()) for _ in range(6)]
        indices = np.concatenate([np.asarray(first[0]), np.asarray(second[0])])
        epochs = np.concatenate([np.asarray(first[1]), np.asarray(second[1])])
        for k, (want_indices, want_epochs) in enumerate(expected):
            np.testing.assert_array_equal(from_words(indices[k]), want_indices)
            np.testing.assert_array_equal(epochs[k], want_epochs)

    def test_a_tree_mode_split_and_merge_steps_as_the_pipeline_does(
        self, shuffle: bool, drop_last: bool, length: int, batch: int
    ) -> None:
        pipe = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        reference = _pipeline(shuffle=shuffle, drop_last=drop_last, length=length, batch=batch)
        graphdef, state = nnx.split(pipe, graph=False)
        merged = nnx.merge(graphdef, state)
        for _ in range(3):
            got, want = _named(merged.step()), _named(reference.step())
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_array_equal(got[1], want[1])
