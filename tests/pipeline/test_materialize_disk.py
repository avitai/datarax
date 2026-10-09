"""``materialize(..., directory=...)``: a decoded copy written in place, reused, rebuilt safely.

Workers write their units' rows straight into the copy's reserved ``.partial`` files (shared
memory maps of one local file), and send back only each unit's provenance and digest: no row
crosses ``/dev/shm`` and no worker flushes a map; the training process fsyncs each file once.
The files are the same whatever wrote them. A complete copy of the same source is returned as it
is, wherever the directory was moved; a changed source, a copy without its manifest, or
``rebuild=True`` writes it again, its manifest unlinked first, so a reader opening it meanwhile is
refused while one that mapped the old files keeps reading them. Processes building one copy at
once build it once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - the tests start their own builder processes
import sys
import textwrap
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from datarax.core.host_resources import available_cpus
from datarax.core.index_words import to_words
from datarax.pipeline import host_workers, materialization, materialize
from datarax.sources import from_tfds, open_decoded_copy, StreamingDiskSource
from datarax.sources.decoded_copy._layout import MANIFEST
from tests.test_common.decoded_copies import every_record, write_copy
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.tfds_golden import digests, EAGER_BUILD
from tests.test_common.worker_reads import (
    logged,
    png_source,
    PngDecoder,
    resources,
    write_png_records,
)


_RECORDS = 30
_UNIT = 4
_FLUSH_SPY = Path(__file__).resolve().parents[1] / "test_common" / "flush_spy"
PROCESS_COUNTS = tuple(k for k in (2, 3) if k <= available_cpus())
COUNTS = (0, 1, *PROCESS_COUNTS)
_BARRIER_SECONDS = 20.0
"""How long a builder's first decode waits for the others (see the four-builder test)."""


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    return write_png_records(tmp_path_factory.mktemp("png"), _RECORDS, shards=2)


def _files(directory: Path) -> dict[str, str]:
    """Each column and provenance file's SHA-256, by name."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.name.endswith((".npy", ".array_record"))
    }


def _content(directory: Path) -> str:
    return json.loads((directory / MANIFEST).read_text())["content"]["digest"]


def _same(left: Any, right: Any) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


class _Crossing:
    """What crosses back from the workers: each element, seen in the training process."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.elements: list[Any] = []
        start = materialization.in_workers

        def seen(dataset: Any, **kwargs: Any) -> Any:
            return start(dataset, **kwargs).map(self._keep)

        monkeypatch.setattr(materialization, "in_workers", seen)

    def _keep(self, element: Any) -> Any:
        self.elements.append(element)
        return element


class TestWritingInPlace:
    @pytest.mark.parametrize("workers", COUNTS)
    def test_the_files_are_the_same_whoever_writes_them(
        self, png_paths: list[str], tmp_path: Path, workers: int
    ) -> None:
        reference = tmp_path / "reference"
        source = png_source(png_paths)
        write_copy(source, reference, unit_records=_UNIT)

        copy = materialize(
            source,
            host_resources=resources(workers) if workers else None,
            directory=tmp_path / "copy",
            unit_records=_UNIT,
        )

        assert isinstance(copy, StreamingDiskSource)
        assert _files(tmp_path / "copy") == _files(reference)
        assert _content(tmp_path / "copy") == _content(reference)
        _same(every_record(copy)[0], every_record(source)[0])

    @pytest.mark.skipif(not PROCESS_COUNTS, reason="fewer than two CPUs: no worker processes")
    def test_workers_send_back_no_rows_only_provenance_and_a_digest(
        self, png_paths: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        crossing = _Crossing(monkeypatch)

        materialize(
            png_source(png_paths),
            host_resources=resources(PROCESS_COUNTS[0]),
            directory=tmp_path / "copy",
            unit_records=_UNIT,
        )

        assert len(crossing.elements) == -(-_RECORDS // _UNIT) - 1
        for unit, provenance, digest in crossing.elements:
            leaves = jax.tree.leaves((unit, provenance, digest))
            assert not any(isinstance(leaf, np.ndarray) for leaf in leaves)
            assert len(digest) == 32
            size = host_workers.pickled_bytes((unit, provenance, digest))
            assert size <= host_workers.pickled_bytes(provenance) + 128

    @pytest.mark.skipif(not PROCESS_COUNTS, reason="fewer than two CPUs: no worker processes")
    @pytest.mark.skipif(
        getattr(sys, "_is_gil_enabled", lambda: True)() is False,
        reason="free-threaded Python runs mp_prefetch on threads: no worker to spy on",
    )
    def test_no_worker_flushes_a_map(
        self, png_paths: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = tmp_path / "flushes.log"
        monkeypatch.setenv("DATARAX_TEST_FLUSH_LOG", str(log))
        monkeypatch.setenv(
            "PYTHONPATH", f"{_FLUSH_SPY}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        )
        decoded = tmp_path / "decoded.log"

        materialize(
            png_source(png_paths, PngDecoder(log=str(decoded))),
            host_resources=resources(PROCESS_COUNTS[0]),
            directory=tmp_path / "copy",
            unit_records=_UNIT,
        )

        workers = {pid for pid, _ in logged(decoded)} - {os.getpid()}
        assert workers, "the workers decoded"
        assert not log.exists() or log.read_text() == ""

    def test_positive_control_the_spy_sees_a_flush_in_a_spawned_process(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "flushes.log"
        script = (
            "import numpy as np, sys; "
            "m = np.memmap(sys.argv[1], mode='w+', dtype='u1', shape=(4,)); m.flush()"
        )
        env = {
            **os.environ,
            "DATARAX_TEST_FLUSH_LOG": str(log),
            "PYTHONPATH": f"{_FLUSH_SPY}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }

        subprocess.run(  # nosec B603 - fixed arguments
            [sys.executable, "-c", script, str(tmp_path / "map")], env=env, check=True, timeout=60
        )

        assert log.read_text().split()

    @pytest.mark.tfds
    @pytest.mark.parametrize("workers", COUNTS)
    def test_a_tfds_split_s_copy_opens_through_from_tfds(
        self, tfds_fixture: TFDSFixture, tmp_path: Path, workers: int
    ) -> None:
        data_dir = str(tfds_fixture.array_record)
        source = from_tfds(FIXTURE, "train", data_dir=data_dir, in_memory=False)

        materialize(
            source,
            host_resources=resources(workers) if workers else None,
            directory=tmp_path / "copy",
            unit_records=_UNIT,
        )

        copy = from_tfds(
            FIXTURE, "train", data_dir=data_dir, in_memory=False, decoded_dir=str(tmp_path / "copy")
        )
        _, columns, names = EAGER_BUILD["train"]
        data, provenance = every_record(copy)
        assert digests(data) == columns
        assert [p["name"].decode() for p in provenance] == names

    @pytest.mark.skipif(not PROCESS_COUNTS, reason="fewer than two CPUs: no worker processes")
    @pytest.mark.parametrize("ram", [False, True])
    def test_a_unit_that_cannot_cross_stops_the_build_naming_it(
        self, png_paths: list[str], tmp_path: Path, ram: bool
    ) -> None:
        source = png_source(png_paths, PngDecoder(unpicklable=13))

        with pytest.raises(Exception, match=r"unit 3"):  # noqa: B017, PT011 - Grain re-raises it
            materialize(
                source,
                host_resources=resources(PROCESS_COUNTS[0]),
                directory=None if ram else tmp_path / "copy",
                unit_records=_UNIT,
            )
        if not ram:
            with pytest.raises(FileNotFoundError, match="no manifest"):
                open_decoded_copy(tmp_path / "copy")


class TestTheDurabilityOrderWithWorkers:
    @pytest.mark.skipif(not PROCESS_COUNTS, reason="fewer than two CPUs: no worker processes")
    def test_files_are_fsynced_after_the_last_unit_arrives_and_the_manifest_last(
        self, png_paths: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not Path("/proc/self/fd").is_dir():
            pytest.skip("names an fsynced descriptor through /proc, which this platform lacks")
        log: list[tuple[str, str]] = []
        fsync, rename, replace = os.fsync, os.rename, os.replace
        arrival = materialization.check_arrival

        def spied_fsync(fd: int) -> None:
            log.append(("fsync", Path(os.readlink(f"/proc/self/fd/{fd}")).name))
            fsync(fd)

        def spied_rename(src: Any, dst: Any) -> None:
            log.append(("rename", Path(src).name))
            rename(src, dst)

        def spied_replace(src: Any, dst: Any) -> None:
            log.append(("replace", Path(dst).name))
            replace(src, dst)

        def spied_arrival(expected: int, unit: int) -> None:
            log.append(("unit", str(unit)))
            arrival(expected, unit)

        monkeypatch.setattr(os, "fsync", spied_fsync)
        monkeypatch.setattr(os, "rename", spied_rename)
        monkeypatch.setattr(os, "replace", spied_replace)
        monkeypatch.setattr(materialization, "check_arrival", spied_arrival)

        materialize(
            png_source(png_paths),
            host_resources=resources(PROCESS_COUNTS[0]),
            directory=tmp_path / "copy",
            unit_records=_UNIT,
        )

        last_unit = max(i for i, (kind, _) in enumerate(log) if kind == "unit")
        partials = [n for kind, n in log if kind == "fsync" and n.endswith(".partial")]
        assert len(partials) == len(set(partials)) == 3, partials
        for name in partials:
            assert last_unit < log.index(("fsync", name)) < log.index(("rename", name))
        assert log.index(("fsync", "manifest.json.tmp")) < log.index(("replace", MANIFEST))
        assert log[-1] == ("fsync", "copy"), "the directory is fsynced last"


class TestReuseAndRebuild:
    def _build(self, paths: list[str], directory: Path, **options: Any) -> Any:
        return materialize(
            png_source(paths, options.pop("decoder", None)),
            directory=directory,
            unit_records=_UNIT,
            **options,
        )

    def test_a_complete_matching_copy_is_returned_without_decoding_or_locking(
        self, png_paths: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "copy"
        self._build(png_paths, directory)
        before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in directory.iterdir()}
        locked: list[Path] = []
        lock = materialization.CopyLock

        def counted(path: Any) -> Any:
            locked.append(Path(path))
            return lock(path)

        monkeypatch.setattr(materialization, "CopyLock", counted)
        decoded = tmp_path / "decoded.log"

        again = self._build(png_paths, directory, decoder=PngDecoder(log=str(decoded)))

        assert logged(decoded) == []
        assert locked == []
        assert {
            p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in directory.iterdir()
        } == before
        _same(every_record(again)[0], every_record(png_source(png_paths))[0])

    def test_a_moved_copy_is_still_reused(self, png_paths: list[str], tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        self._build(png_paths, directory)
        moved = tmp_path / "moved"
        shutil.move(directory, moved)
        decoded = tmp_path / "decoded.log"

        self._build(png_paths, moved, decoder=PngDecoder(log=str(decoded)))

        assert logged(decoded) == []

    def test_a_changed_source_is_rebuilt_naming_what_changed(
        self, png_paths: list[str], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        directory = tmp_path / "copy"
        self._build(png_paths, directory)
        other = write_png_records(tmp_path / "other", _RECORDS + 5, shards=2)

        with caplog.at_level(logging.INFO, logger=materialization.__name__):
            rebuilt = self._build(other, directory)

        assert "source/records" in caplog.text
        assert len(rebuilt) == _RECORDS + 5

    @pytest.mark.parametrize("why", ["no manifest", "rebuild"])
    def test_an_incomplete_copy_or_a_request_rebuilds_into_fresh_files(
        self, png_paths: list[str], tmp_path: Path, why: str
    ) -> None:
        directory = tmp_path / "copy"
        self._build(png_paths, directory)
        inodes = {p.name: p.stat().st_ino for p in directory.iterdir() if p.suffix == ".npy"}
        if why == "no manifest":
            (directory / MANIFEST).unlink()
        decoded = tmp_path / "decoded.log"

        self._build(
            png_paths, directory, decoder=PngDecoder(log=str(decoded)), rebuild=why == "rebuild"
        )

        assert len(logged(decoded)) == _RECORDS
        after = {p.name: p.stat().st_ino for p in directory.iterdir() if p.suffix == ".npy"}
        assert after.keys() == inodes.keys()
        assert all(after[name] != inodes[name] for name in inodes)

    def test_a_reader_keeps_the_old_rows_and_one_opening_mid_rebuild_is_refused(
        self, png_paths: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "copy"
        old = self._build(png_paths, directory)
        held = every_record(old)[0]  # maps the old files
        refused: list[str] = []
        finish = materialization.finish_copy

        def finishing(*args: Any, **kwargs: Any) -> None:
            try:
                open_decoded_copy(directory)
            except FileNotFoundError as error:
                refused.append(str(error))
            finish(*args, **kwargs)

        monkeypatch.setattr(materialization, "finish_copy", finishing)
        other = write_png_records(tmp_path / "other", _RECORDS, shards=3)

        self._build(other, directory, rebuild=True)

        assert refused and "no manifest.json" in refused[0]
        _same(every_record(old)[0], held)

    def test_four_processes_on_one_empty_directory_decode_once(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        """Four builders at once: the lock lets one decode; the others wait and reuse its copy.

        The source's decode forces them to overlap: each builder's first decode waits until all
        four have reached it or ``_BARRIER_SECONDS`` pass. Without the lock all four reach it
        (more than one decodes); with it only the first does, waits the timeout out, and builds
        while the others wait on the lock. The timeout covers four interpreters starting and
        importing JAX and datarax on a loaded machine, a few seconds each.
        """
        directory = tmp_path / "copy"
        decoded = tmp_path / "decoded.log"
        arrivals = tmp_path / "arrivals"
        arrivals.mkdir()
        script = textwrap.dedent(
            f"""
            import json, sys
            from datarax.pipeline import materialize
            from tests.test_common.worker_reads import png_source, BarrierDecoder
            from tests.test_common.decoded_copies import every_record
            decoder = BarrierDecoder(
                log={str(decoded)!r}, arrivals={str(arrivals)!r}, parties=4,
                timeout={_BARRIER_SECONDS},
            )
            source = png_source({png_paths!r}, decoder)
            copy = materialize(source, directory={str(directory)!r}, unit_records={_UNIT})
            data = every_record(copy)[0]
            print(json.dumps([int(x) for x in data["label"]]))
            """
        )
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        runs = [
            subprocess.Popen(  # nosec B603 - fixed arguments
                [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env
            )
            for _ in range(4)
        ]
        outputs = [run.communicate(timeout=300)[0] for run in runs]

        assert [run.returncode for run in runs] == [0, 0, 0, 0]
        assert len(list(arrivals.iterdir())) == 1, "one builder reached the decode"
        assert len({pid for pid, _ in logged(decoded)}) == 1, "one process decoded"
        assert sorted(label for _, label in logged(decoded)) == list(range(_RECORDS))
        assert len(set(outputs)) == 1
        assert json.loads(outputs[0]) == list(range(_RECORDS))


def test_a_copy_reads_its_provenance_by_index(png_paths: list[str], tmp_path: Path) -> None:
    copy = materialize(png_source(png_paths), directory=tmp_path / "copy", unit_records=_UNIT)
    words = to_words(np.array([29, 0, 7], dtype=np.uint64))

    assert [p["name"] for p in copy.provenance(words)] == ["record-29", "record-0", "record-7"]
