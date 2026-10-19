"""Decoded copy format 2: a ragged field stored as one values file and int64 offsets.

A copy of a source holding ragged fields keeps each fixed field as ``column-k.npy`` of
``(N, *shape)``, written in place, and each ragged field as ``column-k.values.npy`` of
``(T, *inner)`` beside ``column-k.offsets.npy``, int64 ``(N + 1,)`` from 0: record ``r`` is
``values[offsets[r]:offsets[r + 1]]``, the layout of Megatron-LM's indexed dataset and levanter's
``JaggedArrayStore``. Workers decode a unit and return its ragged values, which the training
process appends in unit order, so the copy is the same whoever decoded it. The manifest is
written last and its content digest covers the values and the offsets; a copy of format 1 is
refused, naming the rebuild. The copy reads a record's length from its offsets without decoding,
and keeps the pack plans planned over it, one directory per planning parameters.
"""

from __future__ import annotations

import errno
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.host_resources import available_cpus
from datarax.core.index_words import to_words
from datarax.core.pack_plan import PackPlan, plan_rows
from datarax.pipeline import materialize
from datarax.sources import open_decoded_copy, StreamingDiskSource
from datarax.sources.decoded_copy import _files
from datarax.sources.decoded_copy._layout import MANIFEST
from datarax.sources.decoded_copy._plans import copy_plan
from tests.test_common.ragged_sources import (
    FIELDS,
    IdentifiedRaggedSource,
    MisdeclaredRaggedSource,
    RAGGED,
    ragged_records,
    reads,
)
from tests.test_common.small_filesystem import run_on_small_filesystem
from tests.test_common.streams import RecordStream
from tests.test_common.worker_reads import png_source, resources, write_png_records


_RECORDS = 37
_UNIT = 4
PROCESS_COUNTS = tuple(k for k in (2, 3) if k <= available_cpus())
_UNPACKED = "is ragged.*pack records with segment ids.*pad the field"


def _words(rows: Any) -> np.ndarray:
    return to_words(np.asarray(rows, np.uint64))


def _source(**options: Any) -> IdentifiedRaggedSource:
    return IdentifiedRaggedSource(ragged_records(_RECORDS), **options)


def _copy(source: Any, directory: Path, workers: int = 0, **options: Any) -> StreamingDiskSource:
    copy = materialize(
        source,
        host_resources=resources(workers) if workers else None,
        directory=directory,
        unit_records=_UNIT,
        **options,
    )
    assert isinstance(copy, StreamingDiskSource)
    return copy


def _files_of(directory: Path) -> dict[str, str]:
    """Each column and provenance file's SHA-256, by name."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.name.endswith((".npy", ".array_record"))
    }


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST).read_text())


def _same_reads(copy: Any, source: Any, rows: Any) -> None:
    words = _words(rows)
    for field in FIELDS:
        values, offsets = copy.get_ragged(words, field)
        want_values, want_offsets = source.get_ragged(words, field)
        assert values.dtype == want_values.dtype, field
        assert values.shape == want_values.shape, field
        assert values.tobytes() == want_values.tobytes(), field
        np.testing.assert_array_equal(offsets, want_offsets)
        np.testing.assert_array_equal(copy.lengths(words, field), source.lengths(words, field))


class TestARaggedCopyReadsBackRecordForRecord:
    def test_every_field_reads_as_the_source_reads_it(self, tmp_path: Path) -> None:
        source = _source()
        copy = _copy(source, tmp_path / "copy")

        assert sorted(copy.ragged_fields) == sorted(RAGGED)
        assert sorted(copy.field_names) == sorted(FIELDS)
        assert len(copy) == _RECORDS
        everything = np.arange(_RECORDS)
        for rows in (everything, everything[::-1], [5, 5, 0], [36], []):
            _same_reads(copy, source, rows)
        assert copy.provenance(_words(everything)) == source.provenance(_words(everything))

    def test_a_ragged_field_is_a_values_file_and_int64_offsets_from_zero(
        self, tmp_path: Path
    ) -> None:
        source = _source()
        directory = tmp_path / "copy"
        _copy(source, directory)
        manifest = _manifest(directory)

        assert manifest["format"] == 2
        entries = {entry["name"]: entry for entry in manifest["columns"]}
        for name in RAGGED:
            entry = entries[name]
            assert re.fullmatch(r"column-\d+\.values\.npy", entry["file"])
            assert re.fullmatch(r"column-\d+\.offsets\.npy", entry["offsets"]["file"])
            offsets = np.load(directory / entry["offsets"]["file"])
            values = np.load(directory / entry["file"])
            want_values, want_offsets = source.get_ragged(_words(np.arange(_RECORDS)), name)
            assert offsets.dtype == np.int64
            assert offsets.shape == (_RECORDS + 1,)
            np.testing.assert_array_equal(offsets, want_offsets)
            assert values.shape == want_values.shape
            assert values.tobytes() == want_values.tobytes()
            assert entry["shape"] == list(want_values.shape)
        for name in ("label", "pair"):
            assert re.fullmatch(r"column-\d+\.npy", entries[name]["file"])
            assert entries[name]["offsets"] is None
            assert np.load(directory / entries[name]["file"]).shape[0] == _RECORDS

    def test_a_source_without_a_ragged_field_keeps_format_1_s_columns(self, tmp_path: Path) -> None:
        records = [{key: r[key] for key in ("label", "pair", "name")} for r in ragged_records(9)]
        source = IdentifiedRaggedSource(records)
        directory = tmp_path / "copy"
        copy = _copy(source, directory)
        manifest = _manifest(directory)

        assert copy.ragged_fields == ()
        assert sorted(p.name for p in directory.glob("*.npy")) == ["column-0.npy", "column-1.npy"]
        assert all(entry["offsets"] is None for entry in manifest["columns"])
        batch = copy.get_batch(_words(np.arange(9)))
        np.testing.assert_array_equal(batch.data["label"], np.arange(9))
        assert batch.data["pair"].shape == (9, 3)

    @pytest.mark.skipif(not PROCESS_COUNTS, reason="fewer than two CPUs: no worker processes")
    @pytest.mark.parametrize("workers", PROCESS_COUNTS)
    def test_workers_build_the_same_files_as_the_calling_thread(
        self, tmp_path: Path, workers: int
    ) -> None:
        log = tmp_path / "reads.log"
        _copy(_source(), tmp_path / "one")
        _copy(_source(log=log), tmp_path / "workers", workers=workers)

        assert _files_of(tmp_path / "workers") == _files_of(tmp_path / "one")
        assert _manifest(tmp_path / "workers")["content"] == _manifest(tmp_path / "one")["content"]
        assert len({pid for pid, _, _ in reads(log)}) > 1  # units were decoded by workers


class TestTheManifestIsWrittenLast:
    def test_a_crash_after_the_values_before_the_manifest_leaves_no_readable_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "copy"

        def crash(*args: Any, **kwargs: Any) -> None:
            raise OSError("the writer died")

        with monkeypatch.context() as patched:
            patched.setattr(_files.json, "dump", crash)
            with pytest.raises(OSError, match="the writer died"):
                _copy(_source(), directory)

        assert list(directory.glob("column-*.values.npy"))  # the values were renamed
        assert not (directory / MANIFEST).exists()
        with pytest.raises(FileNotFoundError, match="incomplete"):
            open_decoded_copy(directory)
        rebuilt = _copy(_source(), directory)
        _same_reads(rebuilt, _source(), np.arange(_RECORDS))

    @pytest.mark.slow
    def test_a_full_disk_refuses_the_append_naming_it_and_leaves_no_copy(
        self, tmp_path: Path
    ) -> None:
        """A tmpfs of 1 MiB: 2 MiB of ragged values end in ENOSPC, raised, not a ``SIGBUS``.

        A fixed column is reserved before a row is written because a memory map written on a
        full disk kills its writer; a ragged field is appended by ``write``, which raises.
        """
        mount = tmp_path / "small"
        mount.mkdir()
        run = run_on_small_filesystem(
            mount,
            f"""
            import numpy as np
            from pathlib import Path
            from datarax.pipeline import materialize
            from tests.test_common.ragged_sources import IdentifiedRaggedSource
            records = [
                {{"tokens": np.arange(8192 + i, dtype=np.int32), "label": np.int64(i)}}
                for i in range(64)
            ]
            copy = Path({str(mount / "copy")!r})
            try:
                materialize(IdentifiedRaggedSource(records), directory=copy, unit_records=8)
                print("written")
            except OSError as error:
                print("refused", error.errno, (copy / "manifest.json").exists(), error)
            """,
        )

        assert run.returncode == 0, run.stdout + run.stderr  # no signal ended the writer
        words = run.stdout.split()
        assert words[:3] == ["refused", str(errno.ENOSPC), "False"], run.stdout + run.stderr
        assert "cannot append ragged field 'tokens'" in run.stdout, run.stdout


class TestTheContentDigest:
    def test_one_changed_ragged_value_changes_it(self, tmp_path: Path) -> None:
        records = ragged_records(_RECORDS)
        changed = ragged_records(_RECORDS)
        row = next(r for r, record in enumerate(changed) if len(record["tokens"]))
        changed[row]["tokens"][0] += 1
        _copy(IdentifiedRaggedSource(records), tmp_path / "a")
        _copy(IdentifiedRaggedSource(changed, seed=1), tmp_path / "b")

        assert _manifest(tmp_path / "a")["content"] != _manifest(tmp_path / "b")["content"]

    def test_moving_a_value_between_records_changes_it(self, tmp_path: Path) -> None:
        records = ragged_records(_RECORDS)
        moved = ragged_records(_RECORDS)
        row = next(r for r, record in enumerate(moved) if len(record["tokens"]) > 1)
        moved[row + 1]["tokens"] = np.concatenate(
            [moved[row]["tokens"][-1:], moved[row + 1]["tokens"]]
        )
        moved[row]["tokens"] = moved[row]["tokens"][:-1]
        for key in ("tokens",):
            joined = np.concatenate([r[key] for r in moved])
            assert joined.tobytes() == np.concatenate([r[key] for r in records]).tobytes()
        moved[row + 1]["meta"]["mask"] = np.concatenate(
            [moved[row]["meta"]["mask"][-1:], moved[row + 1]["meta"]["mask"]]
        )
        moved[row]["meta"]["mask"] = moved[row]["meta"]["mask"][:-1]
        _copy(IdentifiedRaggedSource(records), tmp_path / "a")
        _copy(IdentifiedRaggedSource(moved, seed=1), tmp_path / "b")

        assert _manifest(tmp_path / "a")["content"] != _manifest(tmp_path / "b")["content"]

    @staticmethod
    def _change(directory: Path, part: str, position: int) -> None:
        entry = next(e for e in _manifest(directory)["columns"] if e["name"] == "tokens")
        path = directory / (entry["file"] if part == "values" else entry["offsets"]["file"])
        stored = np.load(path, mmap_mode="r+")
        stored[position] += 1
        stored.flush()
        del stored

    @pytest.mark.parametrize(("part", "position"), [("values", -1), ("offsets", _RECORDS // 2)])
    def test_verifying_refuses_a_changed_values_or_offsets_file(
        self, tmp_path: Path, part: str, position: int
    ) -> None:
        directory = tmp_path / "copy"
        _copy(_source(), directory)
        self._change(directory, part, position)

        open_decoded_copy(directory)  # opening without a full read checks sizes only
        with pytest.raises(ValueError, match="content digest"):
            open_decoded_copy(directory, verify=True)

    def test_opening_refuses_a_last_offset_unlike_the_values(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        _copy(_source(), directory)
        self._change(directory, "offsets", -1)

        with pytest.raises(ValueError, match=r"'tokens'.*last offset is \d+, but its values hold"):
            open_decoded_copy(directory)


def test_a_format_1_copy_is_refused_naming_the_rebuild(tmp_path: Path) -> None:
    directory = tmp_path / "copy"
    _copy(_source(), directory)
    manifest = _manifest(directory)
    manifest["format"] = 1
    (directory / MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"format 1, and this datarax reads format 2: rebuild it"):
        open_decoded_copy(directory)
    _copy(_source(), directory)  # materialize rebuilds it
    assert _manifest(directory)["format"] == 2


def test_lengths_come_from_the_offsets_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "reads.log"
    source = _source(log=log)
    copy = _copy(source, tmp_path / "copy")
    decoded = len(reads(log))
    assert decoded  # positive control: the build read the source

    def no_values(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("planning read a record's values")

    monkeypatch.setattr(StreamingDiskSource, "get_ragged", no_values)
    monkeypatch.setattr(StreamingDiskSource, "get_batch", no_values)
    every = _words(np.arange(_RECORDS))
    plan = plan_rows(
        length=_RECORDS,
        records_at=lambda start, stop: np.arange(start, stop, dtype=np.uint64),
        lengths_of=lambda records: {"tokens": copy.lengths(to_words(records), "tokens")},
        row_lengths={"tokens": 16},
        max_segments=4,
        window=8,
        directory=tmp_path / "plan",
    )

    assert len(reads(log)) == decoded
    assert plan.length == _RECORDS
    np.testing.assert_array_equal(copy.lengths(every, "tokens"), source.lengths(every, "tokens"))


class TestPackPlansLiveInTheCopy:
    def _plan(self, copy: StreamingDiskSource, calls: list[Path]):
        def plan(directory: Path, row_length: int) -> PackPlan:
            calls.append(directory)
            return plan_rows(
                length=len(copy),
                records_at=lambda start, stop: np.arange(start, stop, dtype=np.uint64),
                lengths_of=lambda records: {"tokens": copy.lengths(to_words(records), "tokens")},
                row_lengths={"tokens": row_length},
                max_segments=4,
                window=8,
                directory=directory,
            )

        return plan

    def test_a_plan_is_reused_for_the_same_parameters_and_rebuilt_for_others(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        copy = _copy(_source(), directory)
        calls: list[Path] = []
        plan = self._plan(copy, calls)

        def planned(row_length: int) -> PackPlan:
            return copy_plan(
                directory,
                parameters={"row_lengths": {"tokens": row_length}, "max_segments": 4, "window": 8},
                plan=lambda target: plan(target, row_length),
            )

        first = planned(16)
        again = planned(16)
        other = planned(24)

        assert len(calls) == 2
        assert again.directory == first.directory
        assert other.directory != first.directory
        assert first.directory.parent == directory
        assert re.fullmatch(r"plan-[0-9a-f]{64}", first.directory.name)
        np.testing.assert_array_equal(again.records, first.records)
        np.testing.assert_array_equal(again.row_starts, first.row_starts)
        assert dict(other.row_lengths) == {"tokens": 24}
        assert not list(directory.glob("*.partial"))

    def test_an_unfinished_plan_is_planned_again(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        copy = _copy(_source(), directory)
        calls: list[Path] = []
        plan = self._plan(copy, calls)
        parameters = {"row_lengths": {"tokens": 16}, "max_segments": 4, "window": 8}

        def dies(target: Path) -> PackPlan:
            plan(target, 16)
            raise OSError("the planner died")

        with pytest.raises(OSError, match="the planner died"):
            copy_plan(directory, parameters=parameters, plan=dies)
        finished = copy_plan(directory, parameters=parameters, plan=lambda target: plan(target, 16))

        assert len(calls) == 2
        assert finished.length == _RECORDS
        assert not list(directory.glob("*.partial"))

    def test_a_rebuilt_copy_holds_no_old_plan(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        copy = _copy(_source(), directory)
        calls: list[Path] = []
        plan = self._plan(copy, calls)
        copy_plan(
            directory,
            parameters={"row_lengths": {"tokens": 16}, "max_segments": 4, "window": 8},
            plan=lambda target: plan(target, 16),
        )
        assert list(directory.glob("plan-*"))

        _copy(_source(), directory, rebuild=True)

        assert not list(directory.glob("plan-*"))


class TestRefusals:
    def test_a_ragged_source_into_ram_is_refused_naming_the_copy(self) -> None:
        with pytest.raises(NotImplementedError, match=r"ragged.*directory="):
            materialize(_source())

    def test_a_ragged_copy_refuses_every_unpacked_read(self, tmp_path: Path) -> None:
        copy = _copy(_source(), tmp_path / "copy")
        words = _words(np.arange(3))

        for read in (
            lambda: copy.get_batch(words),
            lambda: copy.get_records(jnp.asarray(words)),
            copy.element_spec,
        ):
            with pytest.raises(ValueError, match=_UNPACKED):
                read()

    def test_a_field_differing_in_length_that_the_source_names_fixed_is_refused(
        self, tmp_path: Path
    ) -> None:
        source = MisdeclaredRaggedSource(ragged_records(_RECORDS))

        with pytest.raises(ValueError, match=r"'meta/mask'.*names it fixed"):
            _copy(source, tmp_path / "copy")


class TestFieldNames:
    def test_a_ragged_source_names_every_field(self) -> None:
        assert sorted(_source().field_names) == sorted(FIELDS)

    def test_an_indexed_source_without_the_ragged_read_refuses_it(self, tmp_path: Path) -> None:
        source = png_source(write_png_records(tmp_path, 4, shards=1))

        with pytest.raises(NotImplementedError, match="does not read ragged records"):
            _ = source.field_names

    def test_a_stream_refuses_it_naming_why(self) -> None:
        with pytest.raises(TypeError, match="is a stream"):
            _ = RecordStream({"x": np.arange(4)}).field_names
