"""Tests for step-addressed checkpoints of Checkpointable objects."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from hypothesis import given, settings, strategies as st
from substrax.checkpoint import CheckpointNotFoundError, upgrade_checkpoints

from datarax.checkpoint import (
    ITERATOR_STATE_FORMAT2,
    IteratorCheckpoint,
    validate_restore_compatibility,
)
from datarax.pipeline import Pipeline
from datarax.pipeline.iteration import PipelineIterator
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.typing import CheckpointableIterator


class SimpleIterator(CheckpointableIterator):
    """Walks an array and carries its position, a seed and a PRNG key in its state."""

    def __init__(self, data: jax.Array, start_idx: int = 0) -> None:
        self.data = data
        self.idx = start_idx
        self.max_idx = len(data)
        self.epoch = 0
        self.key = jax.random.key(7)

    def __next__(self) -> jax.Array:
        if self.idx >= self.max_idx:
            raise StopIteration
        value = self.data[self.idx]
        self.idx += 1
        return value

    def __iter__(self) -> "SimpleIterator":
        return self

    def get_state(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "max_idx": self.max_idx,
            "epoch": self.epoch,
            "key": self.key,
            "data": self.data,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.idx = state["idx"]
        self.max_idx = state["max_idx"]
        self.epoch = state["epoch"]
        self.key = state["key"]
        self.data = state["data"]


class IdentityIterator(CheckpointableIterator):
    """Iterator exposing Grain-style identity fields for restore validation."""

    def __init__(self, *, data_source_repr, sampler_repr, shard_count, worker_count):
        self.position = 0
        self.data_source_repr = data_source_repr
        self.sampler_repr = sampler_repr
        self.shard_count = shard_count
        self.worker_count = worker_count

    def __next__(self):
        self.position += 1
        return self.position

    def __iter__(self):
        return self

    def get_state(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "data_source_repr": self.data_source_repr,
            "sampler_repr": self.sampler_repr,
            "shard_count": self.shard_count,
            "worker_count": self.worker_count,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.position = state["position"]
        self.data_source_repr = state["data_source_repr"]
        self.sampler_repr = state["sampler_repr"]
        self.shard_count = state["shard_count"]
        self.worker_count = state["worker_count"]


class NotADict:
    """A Checkpointable whose get_state breaks the contract."""

    def get_state(self):
        return [1, 2, 3]

    def set_state(self, state):
        del state


class Stateless:
    """A Checkpointable with nothing to save."""

    def get_state(self):
        return {}

    def set_state(self, state):
        del state


@pytest.fixture
def data() -> jax.Array:
    return jnp.arange(100)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Iterator[IteratorCheckpoint]:
    with IteratorCheckpoint(tmp_path, max_to_keep=10) as ckpt:
        yield ckpt


class TestSaveAndRestore:
    def test_restore_continues_where_the_save_left_off(self, checkpoint, data):
        iterator = SimpleIterator(data)
        for _ in range(10):
            next(iterator)

        path = checkpoint.save(iterator, step=1)
        assert Path(path).exists()

        fresh = SimpleIterator(data)
        checkpoint.restore(fresh, step=1)

        assert fresh.idx == 10
        assert int(next(fresh)) == 10

    def test_python_and_key_leaves_round_trip(self, checkpoint, data):
        iterator = SimpleIterator(data)
        iterator.epoch = 3
        iterator.key = jax.random.key(99)
        checkpoint.save(iterator, step=0)

        fresh = SimpleIterator(data)
        checkpoint.restore(fresh, step=0)

        assert fresh.epoch == 3
        assert jnp.array_equal(
            jax.random.key_data(fresh.key), jax.random.key_data(jax.random.key(99))
        )

    def test_restore_defaults_to_the_latest_step(self, checkpoint, data):
        iterator = SimpleIterator(data)
        for step in (1, 2, 3):
            for _ in range(10):
                next(iterator)
            checkpoint.save(iterator, step=step)

        second = SimpleIterator(data)
        checkpoint.restore(second, step=2)
        latest = SimpleIterator(data)
        checkpoint.restore(latest)

        assert second.idx == 20
        assert latest.idx == 30

    def test_metadata_is_recorded_beside_the_state(self, checkpoint, data):
        checkpoint.save(SimpleIterator(data), step=4, metadata={"run": "a", "note": "mid"})

        metadata = checkpoint.store.read_metadata(4)

        assert metadata.extra == {"run": "a", "note": "mid"}
        assert metadata.items == ("data_iterator",)
        assert metadata.metrics == {}

    def test_metadata_cannot_shadow_a_record_field(self, checkpoint, data):
        """The record's own fields, ``epoch`` among them, are not free metadata keys."""
        with pytest.raises(ValueError, match="epoch"):
            checkpoint.save(SimpleIterator(data), step=4, metadata={"epoch": 2})

        assert not checkpoint.has_checkpoint()

    def test_epoch_is_the_record_field(self, checkpoint, data):
        checkpoint.save(SimpleIterator(data), step=4, epoch=2, metadata={"note": "mid"})

        metadata = checkpoint.store.read_metadata(4)

        assert metadata.epoch == 2
        assert metadata.extra == {"note": "mid"}

    def test_save_if_due_records_the_epoch(self, checkpoint, data):
        checkpoint.save_if_due(SimpleIterator(data), 6, interval=3, epoch=1)

        assert checkpoint.store.read_metadata(6).epoch == 1

    def test_the_state_is_the_data_iterator_item(self, checkpoint, data):
        """The store's format names the item, so any format-3 reader finds the state."""
        iterator = SimpleIterator(data)
        checkpoint.save(iterator, step=1)

        restored = checkpoint.store.restore(1).items["data_iterator"]

        assert restored["idx"] == 0
        assert restored["max_idx"] == len(data)

    def test_steps_are_listed_oldest_first(self, checkpoint, data):
        iterator = SimpleIterator(data)
        assert not checkpoint.has_checkpoint()
        assert checkpoint.all_steps() == []

        for step in (1, 3, 5):
            checkpoint.save(iterator, step=step)

        assert checkpoint.all_steps() == [1, 3, 5]
        assert checkpoint.latest_step() == 5
        assert checkpoint.has_checkpoint()

    def test_max_to_keep_bounds_the_retained_steps(self, tmp_path, data):
        with IteratorCheckpoint(tmp_path, max_to_keep=2) as ckpt:
            for step in range(4):
                ckpt.save(SimpleIterator(data), step=step)
            assert ckpt.all_steps() == [2, 3]


class TestErrors:
    def test_restore_without_checkpoints_raises(self, checkpoint, data):
        with pytest.raises(ValueError, match="No checkpoints found"):
            checkpoint.restore(SimpleIterator(data))

    def test_restore_of_a_missing_step_raises(self, checkpoint, data):
        checkpoint.save(SimpleIterator(data), step=1)
        with pytest.raises(CheckpointNotFoundError, match="step 7"):
            checkpoint.restore(SimpleIterator(data), step=7)

    def test_state_that_is_not_a_dict_is_rejected(self, checkpoint):
        with pytest.raises(TypeError, match="must return a dict"):
            checkpoint.save(NotADict(), step=0)  # type: ignore[arg-type]

    def test_empty_state_is_rejected(self, checkpoint):
        with pytest.raises(ValueError, match="nothing to checkpoint"):
            checkpoint.save(Stateless(), step=0)  # type: ignore[arg-type]

    def test_non_positive_interval_is_rejected(self, checkpoint, data):
        with pytest.raises(ValueError, match="interval must be positive"):
            checkpoint.save_if_due(SimpleIterator(data), 0, interval=0)


class TestSaveIfDue:
    @given(
        step=st.integers(min_value=0, max_value=10_000),
        interval=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=60, deadline=None)
    def test_saves_iff_step_is_a_multiple_of_interval(self, tmp_path_factory, step, interval):
        directory = tmp_path_factory.mktemp("due")
        with IteratorCheckpoint(directory) as ckpt:
            saved = ckpt.save_if_due(SimpleIterator(jnp.arange(4)), step, interval=interval)

            assert (saved is not None) == (step % interval == 0)
            assert ckpt.all_steps() == ([step] if step % interval == 0 else [])


class TestRestoreValidation:
    """Restore-time validation of sampler/source repr and shard/worker counts."""

    @staticmethod
    def _iterator(**overrides):
        base = {
            "data_source_repr": "ArrayRecordSourceModule(paths=['a'])",
            "sampler_repr": "IndexSampler(seed=0)",
            "shard_count": 4,
            "worker_count": 2,
        }
        base.update(overrides)
        return IdentityIterator(**base)

    def _save_reference(self, checkpoint):
        source = self._iterator()
        for _ in range(3):
            next(source)
        checkpoint.save(source, step=1)

    def test_restore_succeeds_when_identity_matches(self, checkpoint):
        self._save_reference(checkpoint)
        target = self._iterator()
        checkpoint.restore(target, step=1)
        assert target.position == 3

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("data_source_repr", "ArrayRecordSourceModule(paths=['b'])"),
            ("sampler_repr", "IndexSampler(seed=99)"),
            ("shard_count", 8),
            ("worker_count", 16),
        ],
    )
    def test_restore_rejects_a_mismatched_identity_field(self, checkpoint, field, value):
        self._save_reference(checkpoint)
        target = self._iterator(**{field: value})
        with pytest.raises(ValueError, match=field):
            checkpoint.restore(target, step=1)
        assert target.position == 0, "a rejected restore must leave the target untouched"

    def test_validation_ignores_fields_only_one_side_has(self):
        validate_restore_compatibility({"position": 1}, {"position": 9, "shard_count": 4})


FORMAT2_FIXTURE = Path(__file__).with_name("fixtures") / "format2" / "iterator_state"
FORMAT2_STEP = 6
_MAKE_FIXTURE = "python3 scripts/write_format2_fixture.py tests/checkpoint/fixtures/format2"
if not FORMAT2_FIXTURE.is_dir():
    raise RuntimeError(
        f"the format-2 iterator checkpoint is missing under {FORMAT2_FIXTURE}; it is generated, "
        f"never committed. Write it first: {_MAKE_FIXTURE}"
    )


def _fixture_pipeline() -> Pipeline:
    """The pipeline ``scripts/make_format2_iterator_fixture.py`` saved its iterator from."""
    source = MemorySource(
        MemorySourceConfig(shuffle=True),
        data={"x": jnp.arange(16, dtype=jnp.float32)},
        rngs=nnx.Rngs(0, shuffle=0),
    )
    return Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0))


class TestFormat2Checkpoints:
    """A root datarax 0.1.11 wrote (substrax format 2) restores and upgrades."""

    def test_the_layout_names_the_iterator_item(self):
        assert ITERATOR_STATE_FORMAT2.items_of({"position": 3}) == {
            "data_iterator": {"position": 3}
        }
        assert ITERATOR_STATE_FORMAT2.template_of({"data_iterator": {"position": 0}}) == {
            "position": 0
        }

    def test_restore_reads_the_old_root(self):
        iterator = iter(_fixture_pipeline())
        assert isinstance(iterator, PipelineIterator)

        with IteratorCheckpoint(FORMAT2_FIXTURE) as checkpoint:
            assert checkpoint.all_steps() == [FORMAT2_STEP]
            checkpoint.restore(iterator, step=FORMAT2_STEP)
            metadata = checkpoint.store.read_metadata(
                FORMAT2_STEP, legacy_layout=ITERATOR_STATE_FORMAT2
            )

        state = iterator.get_state()
        assert (int(state["position"]), int(state["epoch"])) == (8, 1)
        assert metadata.extra["run"] == "fixture"
        assert metadata.epoch == 1
        assert metadata.items == ("data_iterator",)

    def test_upgrade_writes_a_format_3_root(self, tmp_path):
        destination = tmp_path / "upgraded"

        steps = upgrade_checkpoints(
            FORMAT2_FIXTURE, destination, legacy_layout=ITERATOR_STATE_FORMAT2
        )

        assert steps == [FORMAT2_STEP]
        iterator = iter(_fixture_pipeline())
        assert isinstance(iterator, PipelineIterator)
        with IteratorCheckpoint(destination) as checkpoint:
            checkpoint.restore(iterator)
            assert checkpoint.store.read_metadata(FORMAT2_STEP).items == ("data_iterator",)
        assert int(iterator.get_state()["position"]) == 8
