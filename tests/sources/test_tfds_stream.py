"""The TFDS stream: a TFRecord copy read without TensorFlow, named by ``tfds_id`` (C5b).

``TFDSStreamingSource`` reads a split TFDS prepared as TFRecord (TFDS's default format) by an offset
index of its frame headers, checks each frame's CRCs and decodes each record with TFDS's NumPy
decoder, so TensorFlow never enters the process. It names each record ``(shard, offset)``, the id
TFDS reports as ``tfds_id`` (``STREAM_IDS``). With the pipeline's key a pass is ordered as TFDS's
documented training read orders it: the shard files in a keyed order, interleaved 16 at a time in
blocks of 16, then tf.data's buffer shuffle, its picks drawn from a generator keyed by
``fold_in(key, pass)``. A copy prepared as ArrayRecord is refused, naming the eager source.
``from_tfds`` picks the source by the format the copy is prepared in.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx
from substrax.testing import run_python

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch, PADDING_INDEX
from datarax.core.prng import key_words
from datarax.pipeline import Pipeline
from datarax.pipeline.epochs import EpochPlan, Run
from datarax.pipeline.host_stage import RunUnits
from datarax.sources import (
    from_tfds,
    tfds_source,
    TFDSEagerSource,
    TFDSStreamingConfig,
    TFDSStreamingSource,
)
from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common.streams import (
    graph_definitions_across_a_pass,
    non_array_state_leaves,
    second_pulls_after_a_tree_round_trip,
)
from tests.test_common.tfds_fixture import FIXTURE, IMAGE_SHAPE, TFDSFixture, TRAIN_RECORDS


pytestmark = pytest.mark.tfds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_SECONDS = 300.0

# tf.data's own read of the TFRecord copy with tfds_id, files in order: the oracle of names.
_TFDATA_ORACLE = """
import hashlib, json, sys
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
builder = tfds.builder(sys.argv[1], data_dir=sys.argv[2])
config = tfds.ReadConfig(add_tfds_id=True, interleave_cycle_length=1)
rows = [
    [r["tfds_id"].decode(), hashlib.sha256(r["image"].tobytes()).hexdigest(), int(r["label"]),
     r["name"].decode()]
    for r in tfds.as_numpy(builder.as_dataset(split=sys.argv[3], read_config=config))
]
print(json.dumps(rows))
"""


def _oracle(fixture: TFDSFixture, split: str = "train") -> list[list[Any]]:
    result = run_python(
        _TFDATA_ORACLE,
        FIXTURE,
        str(fixture.tfrecord),
        split,
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    return json.loads(result.check().stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def oracle(tfds_fixture: TFDSFixture) -> list[list[Any]]:
    return _oracle(tfds_fixture)


def _stream(fixture: TFDSFixture, split: str = "train", **kwargs: Any) -> TFDSStreamingSource:
    return TFDSStreamingSource(
        TFDSStreamingConfig(name=FIXTURE, split=split, data_dir=str(fixture.tfrecord), **kwargs)
    )


def _pass(
    source: TFDSStreamingSource, size: int = 6, key: jax.Array | None = None
) -> tuple[list[Batch], list[Any]]:
    batches, provenance = [], []
    while True:
        batch, sidecar = source.get_batch(size, key=key, with_provenance=True)
        if not batch.batch_size:
            return batches, provenance
        batches.append(batch)
        provenance.extend(sidecar)


def _tfds_ids(source: TFDSStreamingSource, batches: list[Batch]) -> list[str]:
    words = np.concatenate([np.asarray(b.indices) for b in batches])
    return [f"{source.shard_files[int(hi)]}__{int(lo)}" for hi, lo in words]


def _rows(
    source: TFDSStreamingSource, batches: list[Batch], provenance: list[Any]
) -> list[list[Any]]:
    images = np.concatenate([b["image"] for b in batches])
    labels = np.concatenate([b["label"] for b in batches])
    return [
        [tfds_id, hashlib.sha256(image.tobytes()).hexdigest(), int(label), record["name"].decode()]
        for tfds_id, image, label, record in zip(
            _tfds_ids(source, batches), images, labels, provenance, strict=True
        )
    ]


class TestNamesAndRecords:
    def test_records_and_names_equal_tf_data_s_tfds_id_read(
        self, tfds_fixture: TFDSFixture, oracle: list[list[Any]]
    ) -> None:
        source = _stream(tfds_fixture)

        batches, provenance = _pass(source)

        assert source.record_identity is RecordIdentity.STREAM_IDS
        assert _rows(source, batches, provenance) == oracle

    def test_a_slice_names_its_records_as_the_full_split_does(
        self, tfds_fixture: TFDSFixture, oracle: list[list[Any]]
    ) -> None:
        source = _stream(tfds_fixture, "train[10:20]")

        batches, provenance = _pass(source)

        assert len(source) == 10
        assert _rows(source, batches, provenance) == oracle[10:20]

    def test_a_record_is_a_host_batch_row_with_its_text_beside_it(
        self, tfds_fixture: TFDSFixture, oracle: list[list[Any]]
    ) -> None:
        batch, provenance = _stream(tfds_fixture).get_batch(4, with_provenance=True)

        assert batch["image"].shape == (4, *IMAGE_SHAPE)
        assert batch["image"].dtype == np.uint8
        assert batch["label"].dtype == np.int64
        assert batch["meta"]["score"].shape == (4, 2)
        assert "name" not in batch
        assert [p["name"].decode() for p in provenance] == [row[3] for row in oracle[:4]]
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))

    def test_an_offset_past_the_word_is_refused_naming_the_shard(
        self, tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tfds_source, "_LARGEST_OFFSET", 3)

        with pytest.raises(ValueError, match=r"train.tfrecord-00000-of-00001.*offset 19"):
            _stream(tfds_fixture)  # refused when built, before any record is read

    def test_supervised_keys_and_key_filters(self, tfds_fixture: TFDSFixture) -> None:
        supervised = _stream(tfds_fixture, as_supervised=True).get_batch(2)
        included = _stream(tfds_fixture, include_keys={"label"}).get_batch(2)

        assert set(supervised.data) == {"image", "label"}
        assert set(included.data) == {"label"}

    def test_the_spec_is_the_first_record_s_array_part(self, tfds_fixture: TFDSFixture) -> None:
        spec = _stream(tfds_fixture).element_spec()

        assert spec["image"] == jax.ShapeDtypeStruct(IMAGE_SHAPE, np.uint8)
        assert spec["label"] == jax.ShapeDtypeStruct((), np.int32)
        assert "name" not in spec


class TestTheOrder:
    def test_without_a_key_every_pass_reads_the_files_in_order(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = _stream(tfds_fixture)

        first, second = _pass(source)[0], _pass(source)[0]

        words = np.concatenate([np.asarray(b.indices) for b in first])
        assert _tfds_ids(source, first) == _tfds_ids(source, second)
        assert source.shard_files[int(words[0, 0])].endswith("train.tfrecord-00000-of-00001")
        assert len(set(words[:, 0].tolist())) == 1
        assert words[:, 1].tolist() == list(range(TRAIN_RECORDS))
        assert [int(e) for b in second for e in b.epochs] == [1] * TRAIN_RECORDS

    def test_the_key_orders_each_pass_reproducibly(self, tfds_fixture: TFDSFixture) -> None:
        key = jax.random.key(11)

        def passes(seed_key: jax.Array) -> list[list[str]]:
            source = _stream(tfds_fixture, shuffle_buffer_size=8)
            return [_tfds_ids(source, _pass(source, key=seed_key)[0]) for _ in range(3)]

        orders = passes(key)

        assert orders == passes(key)
        assert passes(jax.random.key(12)) != orders
        assert len({tuple(order) for order in orders}) == 3
        assert all(sorted(order) == sorted(orders[0]) for order in orders)
        assert len(orders[0]) == TRAIN_RECORDS

    def test_the_stream_config_has_no_shuffle_or_preparation_field(self) -> None:
        for field in (
            "shuffle",
            "try_gcs",
            "download_and_prepare_kwargs",
            "beam_num_workers",
            "prefetch_buffer",
            "local_files_only",
        ):
            with pytest.raises(TypeError):
                TFDSStreamingConfig(name=FIXTURE, split="train", **{field: True})  # type: ignore[arg-type]


class TestProvenanceByIdentity:
    def test_provenance_by_index_equals_the_sidecar_of_the_pull(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        batches, sidecar = _pass(source, key=jax.random.key(2))

        looked_up = [p for b in batches for p in source.provenance(b.indices)]

        assert [dict(p) for p in looked_up] == [dict(p) for p in sidecar]

    def test_record_keys_are_the_batch_s_indices(self, tfds_fixture: TFDSFixture) -> None:
        source = _stream(tfds_fixture)
        batch = source.get_batch(3)

        np.testing.assert_array_equal(source.record_keys(batch), batch.indices)

    def test_a_lookup_on_a_fresh_stream_indexes_only_the_files_it_names(
        self, tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indexed: list[str] = []
        build = tfds_source._record_index

        def counting(path: str) -> Any:
            indexed.append(Path(path).name)
            return build(path)

        monkeypatch.setattr(tfds_source, "_record_index", counting)
        source = _stream(tfds_fixture)  # the train split
        test_shard = next(i for i, f in enumerate(source.shard_files) if "-test.tfrecord" in f)

        (record,) = source.provenance(np.asarray([[test_shard, 1]], np.uint32))

        assert indexed == [source.shard_files[test_shard]]
        assert isinstance(record["name"], bytes)

    @pytest.mark.parametrize(
        ("words", "match"),
        [
            (np.asarray([PADDING_INDEX]), "padding"),
            (np.asarray([[0, TRAIN_RECORDS]], np.uint32), "offset"),
            (np.asarray([[99, 0]], np.uint32), "shard"),
        ],
        ids=["padding", "past-the-shard", "no-such-shard"],
    )
    def test_ids_naming_no_record_are_refused(
        self, tfds_fixture: TFDSFixture, words: np.ndarray, match: str
    ) -> None:
        with pytest.raises(IndexError, match=match):
            _stream(tfds_fixture).provenance(words)


def test_every_pass_decodes_in_the_pipeline_s_batch_size(
    tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded: list[int] = []
    decode = tfds_source._decoded_batch  # noqa: SLF001 - the one decode of a TFDS stream

    def watched(features: Any, frames: list[Any], kept: Any) -> Any:
        decoded.append(len(frames))
        return decode(features, frames, kept)

    monkeypatch.setattr(tfds_source, "_decoded_batch", watched)
    pipeline = Pipeline(
        source=_stream(tfds_fixture), stages=[], batch_size=6, rngs=nnx.Rngs(0), num_epochs=3
    )

    served = [batch.batch_size for batch in pipeline]

    assert TRAIN_RECORDS % 6 != 0
    # A pass's last records are completed from the next pass's head (stream_batches), so three
    # passes of 20 records serve ten full batches; each is decoded once, after the declared
    # spec's read of one record.
    assert served == [6] * 10
    assert decoded == [1] + [6] * 10


class TestNnxHygiene:
    """The TFDS stream keeps its position and dataset objects out of NNX state (brief T11)."""

    def test_no_variable_holds_a_python_value(self, tfds_fixture: TFDSFixture) -> None:
        source = _stream(tfds_fixture)
        _pass(source)
        source.provenance(np.asarray([[0, 0]], np.uint32))

        assert non_array_state_leaves(source) == []

    def test_the_graph_definition_does_not_move_as_the_stream_advances(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        before, after = graph_definitions_across_a_pass(_stream(tfds_fixture), 6)

        assert before == after
        assert hash(before) == hash(after)

    def test_a_tree_mode_split_and_merge_round_trip_keeps_reading(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        merged, twin = second_pulls_after_a_tree_round_trip(lambda: _stream(tfds_fixture), 6)

        assert merged[0][:, 1].tolist() == list(range(6, 12))
        for ours, theirs in zip(merged, twin, strict=True):
            np.testing.assert_array_equal(ours, theirs)


_DAMAGED = tfds_source.DamagedRecordError
_TRAIN_SHARD = r"train\.tfrecord-00000-of-00001"


class TestDamagedFrames:
    """A damaged frame is refused naming its file and record, as tf.data's TFRecord reader does.

    tf.data checks each frame's masked CRC32C, of the length and of the payload, and reports a
    corrupted record; the stream checks the same two and a frame cut short.
    """

    @staticmethod
    def _damaged(
        fixture: TFDSFixture, tmp_path: Path, damage: Callable[[bytearray, list[int]], bytes]
    ) -> TFDSStreamingSource:
        shutil.copytree(fixture.tfrecord / FIXTURE, tmp_path / FIXTURE)
        path = _train_shard(tmp_path)
        data = bytearray(path.read_bytes())
        path.write_bytes(damage(data, [start for start, _ in _frames(bytes(data))]))
        return TFDSStreamingSource(
            TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tmp_path))
        )

    def test_a_payload_failing_its_crc_is_refused(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        def flip_a_payload_byte(data: bytearray, starts: list[int]) -> bytes:
            data[starts[3] + 12 + 5] ^= 0xFF
            return bytes(data)

        source = self._damaged(tfds_fixture, tmp_path, flip_a_payload_byte)

        with pytest.raises(_DAMAGED, match=_TRAIN_SHARD + r".*record 3.*CRC"):
            _pass(source)
        shard = source.shard_files.index(_train_name(source))
        with pytest.raises(_DAMAGED, match=r"record 3.*CRC"):
            source.provenance(np.asarray([[shard, 3]], np.uint32))

    def test_a_length_failing_its_crc_is_refused(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        def flip_a_length_bit(data: bytearray, starts: list[int]) -> bytes:
            data[starts[2]] ^= 0x01
            return bytes(data)

        source = self._damaged(tfds_fixture, tmp_path, flip_a_length_bit)

        with pytest.raises(_DAMAGED, match=_TRAIN_SHARD + r".*record 2.*CRC"):
            source.get_batch(4)

    def test_a_file_cut_short_is_refused(self, tfds_fixture: TFDSFixture, tmp_path: Path) -> None:
        source = self._damaged(tfds_fixture, tmp_path, lambda data, _: bytes(data[:-10]))

        with pytest.raises(_DAMAGED, match=_TRAIN_SHARD + r".*record 19.*short"):
            source.get_batch(4)


def _train_name(source: TFDSStreamingSource) -> str:
    return next(name for name in source.shard_files if "-train.tfrecord" in name)


class TestTheCopyTheStreamReads:
    def test_an_array_record_copy_is_refused_naming_the_eager_source(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        config = TFDSStreamingConfig(
            name=FIXTURE, split="train", data_dir=str(tfds_fixture.array_record)
        )

        with pytest.raises(FileNotFoundError, match="TFDSEagerSource") as refused:
            TFDSStreamingSource(config)
        assert "array_record" in str(refused.value)
        assert "host stage" in str(refused.value)

    def test_a_copy_not_prepared_is_refused_naming_the_preparing_call(self, tmp_path: Path) -> None:
        config = TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError, match="file_format='tfrecord'"):
            TFDSStreamingSource(config)

    def test_from_tfds_picks_the_source_by_the_prepared_format(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        eager = from_tfds(FIXTURE, "train", data_dir=str(tfds_fixture.array_record))
        stream = from_tfds(FIXTURE, "train", data_dir=str(tfds_fixture.tfrecord))

        assert isinstance(eager, TFDSEagerSource)
        assert isinstance(stream, TFDSStreamingSource)
        assert stream.get_batch(2).batch_size == 2

    def test_from_tfds_takes_no_eager_argument(self, tfds_fixture: TFDSFixture) -> None:
        with pytest.raises(TypeError):
            from_tfds(FIXTURE, "train", eager=False, data_dir=str(tfds_fixture.tfrecord))  # type: ignore[call-arg]

    def test_from_tfds_over_nothing_prepared_names_the_preparing_call(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="file_format='array_record'"):
            from_tfds(FIXTURE, "train", data_dir=str(tmp_path))


# Builds a stream over the TFRecord copy, reads a shuffled pass, looks provenance up and runs a
# pipeline over it; prints whether TensorFlow is in the process after each stage.
_STREAM_WITHOUT_TENSORFLOW = """
import json, sys
if "--import-tensorflow" in sys.argv:
    import tensorflow
import jax
from flax import nnx
seen = {}
from datarax.pipeline import Pipeline
from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource
source = TFDSStreamingSource(TFDSStreamingConfig(name=NAME, split="train", data_dir=sys.argv[1]))
seen["construct"] = "tensorflow" in sys.modules
batch = source.get_batch(8, key=jax.random.key(0))
seen["read"] = "tensorflow" in sys.modules
source.provenance(batch.indices)
seen["provenance"] = "tensorflow" in sys.modules
source.reset()
served = sum(b.batch_size for b in Pipeline(source=source, stages=[], batch_size=4,
                                             rngs=nnx.Rngs(0), shuffle=True, num_epochs=2))
seen["pipeline"] = "tensorflow" in sys.modules
print(json.dumps({"served": served, "seen": seen}))
"""


@pytest.mark.parametrize("control", [False, True], ids=["stream", "control-imports-tensorflow"])
def test_the_stream_never_imports_tensorflow(tfds_fixture: TFDSFixture, control: bool) -> None:
    result = run_python(
        _STREAM_WITHOUT_TENSORFLOW.replace("NAME", repr(FIXTURE)),
        str(tfds_fixture.tfrecord),
        *(["--import-tensorflow"] if control else []),
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )

    report = json.loads(result.check().stdout.strip().splitlines()[-1])
    assert report["served"] == 2 * TRAIN_RECORDS
    assert set(report["seen"].values()) == {control}


# The order of the fixture's train split at buffer 8 and batch 4 (record offsets in its one shard),
# for keys 0 and 11, passes 0 and 1: the order decided for TFDS streams ("C5b stream order",
# OWN-1002-STREAM-ID-ORDER keeps it bit for bit). Recorded from datarax 11b7aff.
_DECIDED_ORDER = {
    (0, 0): [5, 4, 8, 9, 11, 3, 7, 10, 13, 16, 17, 0, 18, 19, 2, 12, 1, 14, 6, 15],
    (0, 1): [3, 8, 4, 5, 9, 6, 7, 14, 11, 13, 12, 1, 15, 19, 10, 16, 18, 2, 17, 0],
    (11, 0): [6, 2, 3, 0, 5, 10, 13, 14, 15, 11, 8, 16, 9, 12, 1, 19, 4, 7, 18, 17],
    (11, 1): [0, 7, 4, 5, 11, 10, 2, 14, 6, 8, 1, 9, 17, 13, 3, 12, 16, 19, 18, 15],
}


class _CountingReads:
    """Every ``_pread`` call: the file it reads, how many bytes and from where."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, int, int]] = []
        read = tfds_source._pread

        def counting(file: Any, size: int, position: int) -> bytes:
            self.calls.append((Path(file.name).name, size, position))
            return read(file, size, position)

        monkeypatch.setattr(tfds_source, "_pread", counting)


def _train_shard(data_dir: Path) -> Path:
    (path,) = (data_dir / FIXTURE).glob("*/*-train.tfrecord-*")
    return path


def _frames(data: bytes) -> list[tuple[int, int]]:
    """Each TFRecord frame's start and payload length, read here from the frames."""
    frames, position = [], 0
    while position < len(data):
        (length,) = struct.unpack("<Q", data[position : position + 8])
        frames.append((position, length))
        position += 12 + length + 4
    return frames


def _payload_lengths(fixture: TFDSFixture) -> list[int]:
    """Each record's payload length in the fixture's train shard."""
    return [length for _, length in _frames(_train_shard(fixture.tfrecord).read_bytes())]


class TestThePassDatasetForWorkers:
    """The pass is a Grain dataset that worker processes split without changing its order.

    Every worker computes the same order over record ids from the offset index the parent built,
    and reads the payloads of its own batches only (OWN-1002-STREAM-ID-ORDER).
    """

    @staticmethod
    def _batches(dataset: Any) -> list[tuple[list[int], bytes]]:
        return [
            (ids.tolist(), hashlib.sha256(columns["image"].tobytes()).digest())
            for columns, _, ids, _ in dataset
        ]

    @pytest.mark.parametrize(("seed", "pass_index"), sorted(_DECIDED_ORDER))
    def test_the_order_is_the_decided_order(
        self, tfds_fixture: TFDSFixture, seed: int, pass_index: int
    ) -> None:
        dataset = _stream(tfds_fixture, shuffle_buffer_size=8).pass_dataset(
            pass_index, key_words(jax.random.key(seed)), 4
        )

        served = [int(i) & 0xFFFFFFFF for _, _, ids, _ in dataset for i in ids]

        assert served == _DECIDED_ORDER[(seed, pass_index)]

    @pytest.mark.parametrize("slices", [1, 2, 4, 8])
    def test_slices_interleaved_round_robin_are_the_unsliced_pass(
        self, tfds_fixture: TFDSFixture, slices: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        key = key_words(jax.random.key(4))
        whole = self._batches(source.pass_dataset(1, key, 2))
        decoded: list[int] = []
        decode = tfds_source._decoded_batch

        def counting(features: Any, frames: Any, read: Any) -> Any:
            decoded.append(len(frames))
            return decode(features, frames, read)

        monkeypatch.setattr(tfds_source, "_decoded_batch", counting)
        parts = []
        for i in range(slices):
            dataset = source.pass_dataset(1, key, 2)
            dataset.set_slice(slice(i, None, slices))
            parts.append(self._batches(dataset))
        interleaved = [
            part[j] for j in range(max(map(len, parts))) for part in parts if j < len(part)
        ]

        assert interleaved == whole
        assert len(decoded) == len(whole)  # each batch decoded by one slice only

    @pytest.mark.parametrize("slices", [2, 4, 8])
    def test_each_slice_reads_the_payloads_of_its_own_records_only(
        self, tfds_fixture: TFDSFixture, slices: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        key = key_words(jax.random.key(4))
        source.pass_dataset(0, key, 2)  # the parent builds the offset index here, once
        reads = _CountingReads(monkeypatch)
        for i in range(slices):
            dataset = source.pass_dataset(0, key, 2)
            dataset.set_slice(slice(i, None, slices))
            reads.calls.clear()
            served = [int(r) & 0xFFFFFFFF for _, _, ids, _ in dataset for r in ids]

            assert len(reads.calls) == len(served)  # one read per own record, no header reads
            lengths = _payload_lengths(tfds_fixture)
            # Each read is a record's payload and its 4-byte CRC, nothing more.
            assert sum(size for _, size, _ in reads.calls) == sum(lengths[o] + 4 for o in served)

    def test_building_the_offset_index_reads_the_frame_headers_only(
        self, tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _stream(tfds_fixture)
        reads = _CountingReads(monkeypatch)

        source.pass_dataset(0, None, 2)

        assert {size for _, size, _ in reads.calls} == {12}
        assert len(reads.calls) == TRAIN_RECORDS + 1  # each header, then the end of the file

    def test_a_sequential_slice_is_refused(self, tfds_fixture: TFDSFixture) -> None:
        dataset = _stream(tfds_fixture).pass_dataset(0, None, 2)

        with pytest.raises(ValueError, match="strides"):
            dataset.set_slice(slice(0, None, 2), sequential_slice=True)

    def test_the_pass_dataset_pickles_and_reads_the_same_batches(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        import cloudpickle  # noqa: PLC0415 - Grain's process prefetch pickles with it

        dataset = _stream(tfds_fixture, shuffle_buffer_size=8).pass_dataset(
            0, key_words(jax.random.key(9)), 4
        )
        copy = cloudpickle.loads(cloudpickle.dumps(dataset))

        assert self._batches(copy) == self._batches(dataset)

    def test_iterators_started_together_in_threads_read_the_same_batches(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        dataset = _stream(tfds_fixture, shuffle_buffer_size=8).pass_dataset(
            0, key_words(jax.random.key(9)), 4
        )
        reference = self._batches(dataset)

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda _: self._batches(dataset), range(8)))

        assert all(result == reference for result in results)


def _run_units(
    length: int,
    batch_size: int,
    *,
    start: tuple[int, int] = (0, 0),
    passes: int = 3,
    drop_last: bool = False,
    chunk: int = 1,
) -> RunUnits:
    """A run of ``passes`` passes from ``(position, pass)``, cut into the host stage's units."""
    plan = EpochPlan(length=length, batch_size=batch_size, drop_last=drop_last, num_epochs=passes)
    run = Run(plan=plan, position=start[0], epoch=start[1], end_epoch=passes)
    return RunUnits(run=run, chunk=chunk)


class TestRunDataset:
    """A run's passes as one Grain dataset of decoded batches, numbered from where the run starts.

    One dataset spans the run, so a worker process slicing it starts once per run, and its batches
    are numbered from the run's first, so ``k`` slices interleaved from worker 0 serve the run's
    order wherever it resumed.
    """

    @staticmethod
    def _rows(dataset: Any) -> list[tuple[list[int], list[int], bytes]]:
        return [
            (
                [int(i) for i in ids],
                [int(e) for e in epochs],
                hashlib.sha256(columns["image"].tobytes()).digest(),
            )
            for columns, _, ids, epochs in dataset
        ]

    @staticmethod
    def _pass_ids(source: TFDSStreamingSource, key: Any, pass_index: int) -> list[int]:
        return [int(i) for _, _, ids, _ in source.pass_dataset(pass_index, key, 64) for i in ids]

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_the_run_is_its_passes_cut_into_batches_by_the_plan(
        self, tfds_fixture: TFDSFixture, drop_last: bool
    ) -> None:
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        key = key_words(jax.random.key(4))
        served = self._rows(
            source.run_dataset(_run_units(TRAIN_RECORDS, 6, drop_last=drop_last), key)
        )
        expected_ids: list[int] = []
        expected_epochs: list[int] = []
        for pass_index in range(3):
            ids = self._pass_ids(source, key, pass_index)
            if drop_last:
                ids = ids[: len(ids) // 6 * 6]
            expected_ids += ids
            expected_epochs += [pass_index] * len(ids)
        assert [i for ids, _, _ in served for i in ids] == expected_ids
        assert [e for _, epochs, _ in served for e in epochs] == expected_epochs
        assert [len(ids) for ids, _, _ in served][:-1] == [6] * (len(served) - 1)

    @pytest.mark.parametrize("slices", [2, 3, 8])
    def test_slices_interleaved_from_the_first_serve_the_run_each_batch_decoded_once(
        self, tfds_fixture: TFDSFixture, slices: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        key = key_words(jax.random.key(4))
        whole = self._rows(source.run_dataset(_run_units(TRAIN_RECORDS, 4), key))
        decoded: list[int] = []
        decode = tfds_source._decoded_batch

        def counting(features: Any, frames: Any, read: Any) -> Any:
            decoded.append(len(frames))
            return decode(features, frames, read)

        monkeypatch.setattr(tfds_source, "_decoded_batch", counting)
        parts = []
        for i in range(slices):
            dataset = source.run_dataset(_run_units(TRAIN_RECORDS, 4), key)
            dataset.set_slice(slice(i, None, slices))
            parts.append(self._rows(dataset))
        interleaved = [
            part[j] for j in range(max(map(len, parts))) for part in parts if j < len(part)
        ]
        assert interleaved == whole
        assert len(decoded) == len(whole)

    @pytest.mark.parametrize("slices", [2, 3])
    @pytest.mark.parametrize("resume_at", [1, 5, 7])
    def test_a_resumed_run_sliced_serves_the_rest_in_order_reading_no_skipped_payload(
        self,
        tfds_fixture: TFDSFixture,
        slices: int,
        resume_at: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resumed at a batch not divisible by ``k``, the slices still interleave from worker 0."""
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        key = key_words(jax.random.key(7))
        units = _run_units(TRAIN_RECORDS, 4)
        whole = self._rows(source.run_dataset(units, key))
        position, epoch = 0, 0
        for ordinal in range(resume_at):
            batch = units.run.batch(ordinal)
            assert batch is not None
            position, epoch = units.run.plan.advance(*batch)
        resumed = _run_units(TRAIN_RECORDS, 4, start=(position, epoch))
        source.run_dataset(resumed, key)  # the parent builds the offset index here, once
        reads = _CountingReads(monkeypatch)
        parts = []
        for i in range(slices):
            dataset = source.run_dataset(resumed, key)
            dataset.set_slice(slice(i, None, slices))
            parts.append(self._rows(dataset))
        interleaved = [
            part[j] for j in range(max(map(len, parts))) for part in parts if j < len(part)
        ]
        assert interleaved == whole[resume_at:]
        assert len(reads.calls) == sum(len(ids) for ids, _, _ in whole[resume_at:])

    def test_the_run_dataset_pickles_and_reads_the_same_batches(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        import cloudpickle  # noqa: PLC0415 - Grain's process prefetch pickles with it

        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        dataset = source.run_dataset(_run_units(TRAIN_RECORDS, 4), key_words(jax.random.key(1)))
        copy = cloudpickle.loads(cloudpickle.dumps(dataset))
        assert self._rows(copy) == self._rows(dataset)


class TestThroughTheHostStage:
    def test_raw_batches_serve_the_run_dataset_s_batches(self, tfds_fixture: TFDSFixture) -> None:
        source = _stream(tfds_fixture, shuffle_buffer_size=8)
        pipe = Pipeline(
            source=source, stages=[], batch_size=6, rngs=nnx.Rngs(2), shuffle=True, num_epochs=2
        )
        served = list(pipe.raw_batches())
        key = key_words(pipe._epoch_key_base[...])  # noqa: SLF001
        expected = [
            [int(i) for i in ids]
            for _, _, ids, _ in source.run_dataset(_run_units(TRAIN_RECORDS, 6, passes=2), key)
        ]
        names = [
            [(int(hi) << 32) | int(lo) for hi, lo in np.asarray(batch.indices)] for batch in served
        ]
        assert names == expected

    def test_units_are_read_ahead_of_the_consumer_on_a_thread(
        self, tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decode runs on the stream's read thread, ahead of the consumer, as an indexed read does.

        At run creation the consumer reads one record for the declared spec; every unit is then
        decoded on the read thread, which fills the read buffer while the consumer holds a batch.
        """
        decode = tfds_source._decoded_batch  # noqa: SLF001 - the one decode of a TFDS stream
        threads: list[str] = []
        decoded = threading.Condition()

        def watched(features: Any, frames: list[Any], kept: Any) -> Any:
            batch = decode(features, frames, kept)
            with decoded:
                threads.append(threading.current_thread().name)
                decoded.notify_all()
            return batch

        monkeypatch.setattr(tfds_source, "_decoded_batch", watched)
        pipe = Pipeline(
            source=_stream(tfds_fixture), stages=[], batch_size=4, rngs=nnx.Rngs(0), num_epochs=None
        )
        batches = iter(pipe.raw_batches())
        next(batches)
        consumer = threading.current_thread().name
        with decoded:  # the spec's record, the batch taken, and the read buffer's two units
            decoded.wait_for(lambda: len(threads) >= 4, timeout=30)
            seen = list(threads)
        pipe.close()
        assert len(seen) >= 4, seen
        assert seen[0] == consumer
        assert consumer not in seen[1:], seen

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_resume_after_any_batch_is_exact_and_replays_no_payload(
        self, tfds_fixture: TFDSFixture, drop_last: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``get_state`` after every batch of a 3-pass shuffled run, read ahead; the rest resumes.

        A resumed run reads the payloads of the records it serves and of the spec's one record,
        never those before the saved position.
        """

        def build() -> Pipeline:
            return Pipeline(
                source=_stream(tfds_fixture, shuffle_buffer_size=8),
                stages=[],
                batch_size=6,
                rngs=nnx.Rngs(5),
                shuffle=True,
                drop_last=drop_last,
                num_epochs=3,
            )

        def rows(batch: Batch) -> tuple[list[tuple[int, int]], list[int], bytes]:
            return (
                [(int(hi), int(lo)) for hi, lo in np.asarray(batch.indices)],
                [int(e) for e in np.asarray(batch.epochs)],
                hashlib.sha256(np.asarray(batch["image"]).tobytes()).digest(),
            )

        whole = [rows(b) for b in build().raw_batches()]
        assert len(whole) == (9 if drop_last else 10)  # 3 passes of 20 records, 6 a batch
        payloads: list[int] = []
        payload = tfds_source._payload  # noqa: SLF001 - a record's payload read

        def counted(file: Any, path: str, index: Any, offset: int) -> bytes:
            payloads.append(offset)
            return payload(file, path, index, offset)

        monkeypatch.setattr(tfds_source, "_payload", counted)
        for done in range(len(whole) + 1):
            pipe = build()
            batches = iter(pipe.raw_batches())
            served = [rows(next(batches)) for _ in range(done)]
            state = pipe.get_state()
            pipe.close()
            resumed = build()
            resumed.set_state(state)
            assert resumed.batches_left() == len(whole) - done
            payloads.clear()
            rest = [rows(b) for b in resumed.raw_batches()]
            assert served + rest == whole
            assert len(payloads) == 1 + sum(len(names) for names, _, _ in rest)

    @pytest.mark.parametrize("ending", ["exhausted", "close", "break"])
    def test_no_read_thread_is_left(self, tfds_fixture: TFDSFixture, ending: str) -> None:
        pipe = Pipeline(
            source=_stream(tfds_fixture), stages=[], batch_size=4, rngs=nnx.Rngs(0), num_epochs=2
        )
        batches = pipe.raw_batches()
        if ending == "exhausted":
            list(batches)
        else:
            next(iter(batches))
            if ending == "close":
                pipe.close()
            else:
                del batches, pipe
                gc.collect()
        threads = []
        for _ in range(100):
            threads = [t.name for t in threading.enumerate() if "grain" in t.name.lower()]
            if not threads:
                break
            threading.Event().wait(0.05)
        assert threads == []
