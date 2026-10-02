"""Tests for scripts/prepare_example_datasets.py, which fills CI's cache of example datasets."""

from __future__ import annotations

import ast
import hashlib
import re
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from tests.scripts.script_loader import load_script, REPO_ROOT
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


_KERAS_CIFAR10 = re.compile(r"from keras\.datasets import cifar10\b")
_TFDS_READERS = {"TFDSEagerConfig", "TFDSStreamingConfig", "from_tfds"}
_BLOB = bytes(range(256)) * 4


@pytest.fixture(scope="module")
def prepare() -> ModuleType:
    return load_script("prepare_example_datasets")


def _dataset_name(call: ast.Call) -> str | None:
    """The literal dataset name a TFDS source config or factory call is given, if any."""
    named = [keyword.value for keyword in call.keywords if keyword.arg == "name"]
    candidates = named or call.args[:1]
    if candidates and isinstance(candidates[0], ast.Constant):
        return str(candidates[0].value)
    return None


def _tfds_datasets_the_examples_read() -> set[str]:
    """Every TFDS dataset an example script reads through a datarax TFDS source."""
    names: set[str] = set()
    for path in sorted((REPO_ROOT / "examples").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else None
            callee = node.func.id if isinstance(node.func, ast.Name) else callee
            if callee in _TFDS_READERS and (name := _dataset_name(node)) is not None:
                names.add(name)
    return names


def test_every_tfds_dataset_the_examples_read_is_prepared(prepare: ModuleType) -> None:
    """The examples read prepared ArrayRecord copies, and the script prepares exactly those."""
    read = _tfds_datasets_the_examples_read()

    assert read == {"cifar10", "fashion_mnist", "mnist"}
    assert set(prepare.TFDS_DATASETS) == read


def test_the_keras_cifar10_loader_the_examples_use_is_prepared(prepare: ModuleType) -> None:
    """CIFAR-10 is served from a slow host, so the keras loader of it is cached in CI too."""
    examples = sorted((REPO_ROOT / "examples").rglob("*.py"))

    assert any(_KERAS_CIFAR10.search(path.read_text(encoding="utf-8")) for path in examples)
    assert "cifar10" in prepare.KERAS_DATASETS


def test_a_tfds_dataset_is_prepared_as_array_record(
    prepare: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ArrayRecord is the format the eager TFDS source reads without TensorFlow."""
    import tensorflow_datasets as tfds

    calls: list[tuple[str, dict[str, object]]] = []

    class _Builder:
        def download_and_prepare(self, **kwargs: object) -> None:
            calls.append(("download_and_prepare", kwargs))

    def builder(name: str, **kwargs: object) -> _Builder:
        calls.append((name, kwargs))
        return _Builder()

    monkeypatch.setattr(tfds, "builder", builder)

    prepare.prepare_tfds_dataset("mnist", tmp_path)

    assert calls[0] == ("mnist", {"file_format": "array_record"})
    assert calls[1][0] == "download_and_prepare"
    assert calls[1][1]["download_dir"] == tmp_path


def test_the_cache_key_names_the_format_it_holds() -> None:
    """A cache of another format is never restored as this one."""
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    keys = {
        step["with"]["key"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/cache")
        and "tensorflow_datasets" in str(step.get("with", {}).get("path", ""))
    }

    assert keys == {
        "example-datasets-array_record-${{ hashFiles('scripts/prepare_example_datasets.py') }}"
    }


@pytest.mark.tfds
def test_a_copy_in_another_format_is_listed_for_replacement(
    prepare: ModuleType, tfds_fixture: TFDSFixture
) -> None:
    stale = prepare.stale_copies([FIXTURE], data_dir=str(tfds_fixture.tfrecord))

    assert stale == {FIXTURE: tfds_fixture.tfrecord / FIXTURE / "1.0.0"}


@pytest.mark.tfds
def test_an_array_record_copy_or_none_is_not_listed(
    prepare: ModuleType, tfds_fixture: TFDSFixture, tmp_path: Path
) -> None:
    assert prepare.stale_copies([FIXTURE], data_dir=str(tfds_fixture.array_record)) == {}
    assert prepare.stale_copies(["mnist", FIXTURE], data_dir=str(tmp_path)) == {}


def test_removing_a_stale_copy_deletes_that_version_and_nothing_else(
    prepare: ModuleType, tmp_path: Path
) -> None:
    version = tmp_path / "mnist" / "3.0.1"
    for kept in (
        tmp_path / "mnist" / "3.0.0",
        tmp_path / "nsynth" / "2.3.3",
        tmp_path / "downloads",
    ):
        kept.mkdir(parents=True)
        (kept / "keep.txt").write_text("kept")
    version.mkdir(parents=True)
    (version / "mnist-train.tfrecord-00000-of-00001").write_bytes(b"x")

    prepare.remove_stale_copy(version)

    assert not version.exists()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("keep.txt")) == [
        "downloads/keep.txt",
        "mnist/3.0.0/keep.txt",
        "nsynth/2.3.3/keep.txt",
    ]


class _RangeHandler(BaseHTTPRequestHandler):
    """Serves ``_BLOB`` in ranges at /archive, redirects /moved there; /whole ignores ranges."""

    def do_HEAD(self) -> None:
        self._respond(include_body=False)

    def do_GET(self) -> None:
        self._respond(include_body=True)

    def _respond(self, *, include_body: bool) -> None:
        if self.path == "/moved":
            self.send_response(301)
            self.send_header("Location", "/archive")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body, status = _BLOB, 200
        requested = self.headers.get("Range")
        if self.path == "/archive" and requested:
            start, end = (int(bound) for bound in requested.removeprefix("bytes=").split("-"))
            body, status = _BLOB[start : end + 1], 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep test output quiet."""


@pytest.fixture
def server_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_resolve_url_follows_the_redirect(prepare: ModuleType, server_url: str) -> None:
    assert prepare.resolve_url(f"{server_url}/moved") == f"{server_url}/archive"


def test_fetch_range_returns_the_requested_bytes_through_a_redirect(
    prepare: ModuleType, server_url: str
) -> None:
    assert prepare.fetch_range(f"{server_url}/moved", 3, 9) == _BLOB[3:10]


def test_fetch_range_rejects_a_server_that_ignores_the_range(
    prepare: ModuleType, server_url: str
) -> None:
    with pytest.raises(ValueError, match="206"):
        prepare.fetch_range(f"{server_url}/whole", 3, 9)


def test_plan_ranges_covers_every_byte_once(prepare: ModuleType) -> None:
    assert prepare.plan_ranges(10, 4) == [(0, 3), (4, 7), (8, 9)]
    assert prepare.plan_ranges(8, 4) == [(0, 3), (4, 7)]


def _archive(prepare: ModuleType, url: str, blob: bytes) -> object:
    return prepare.Archive(url=url, size=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def _serve_blobs(blobs: dict[str, bytes]) -> object:
    def fetch(url: str, start: int, end: int) -> bytes:
        return blobs[url][start : end + 1]

    return fetch


def test_fetch_archives_shares_one_connection_limit_across_archives(
    prepare: ModuleType, tmp_path: Path
) -> None:
    """Every range of every archive shares one pool, so the host never sees more connections."""
    blobs = {"https://mirror.example/a": _BLOB, "https://mirror.example/b": _BLOB[:700]}
    lock = threading.Lock()
    active = 0
    most_active = 0

    def fetch(url: str, start: int, end: int) -> bytes:
        nonlocal active, most_active
        with lock:
            active += 1
            most_active = max(most_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return blobs[url][start : end + 1]

    downloads = [
        (_archive(prepare, "https://host.example/a", _BLOB), tmp_path / "a" / "a.tar.gz"),
        (_archive(prepare, "https://host.example/b", _BLOB[:700]), tmp_path / "b.tar.gz"),
    ]

    hosts = prepare.fetch_archives(
        downloads,
        fetch_range=fetch,
        resolve_url=lambda url: url.replace("host.example", "mirror.example"),
        connections=3,
        part_bytes=64,
    )

    assert (tmp_path / "a" / "a.tar.gz").read_bytes() == _BLOB
    assert (tmp_path / "b.tar.gz").read_bytes() == _BLOB[:700]
    assert most_active == 3
    assert hosts == {"host.example", "mirror.example"}


def test_fetch_archives_rejects_an_archive_whose_digest_differs(
    prepare: ModuleType, tmp_path: Path
) -> None:
    archive = prepare.Archive(url="https://host.example/a", size=len(_BLOB), sha256="0" * 64)

    with pytest.raises(ValueError, match="https://host.example/a"):
        prepare.fetch_archives(
            [(archive, tmp_path / "a.tar.gz")],
            fetch_range=_serve_blobs({"https://host.example/a": _BLOB}),
            resolve_url=lambda url: url,
            part_bytes=64,
        )


def test_fetch_archives_fetches_a_reset_range_again(
    prepare: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare, "RETRY_DELAY_SECONDS", 0.0)
    calls: list[int] = []

    def fetch(url: str, start: int, end: int) -> bytes:
        calls.append(start)
        if len(calls) == 1:
            raise ConnectionResetError("Connection reset by peer")
        return _BLOB[start : end + 1]

    prepare.fetch_archives(
        [(_archive(prepare, "https://host.example/a", _BLOB), tmp_path / "a.tar.gz")],
        fetch_range=fetch,
        resolve_url=lambda url: url,
        connections=1,
        part_bytes=256,
    )

    assert (tmp_path / "a.tar.gz").read_bytes() == _BLOB
    assert calls == [0, 0, 256, 512, 768]


def test_fetch_archives_gives_up_after_the_last_attempt(
    prepare: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare, "RETRY_DELAY_SECONDS", 0.0)
    calls: list[int] = []

    def fetch(url: str, start: int, end: int) -> bytes:
        calls.append(start)
        raise ConnectionResetError("Connection reset by peer")

    with pytest.raises(ConnectionResetError):
        prepare.fetch_archives(
            [(_archive(prepare, "https://host.example/a", _BLOB), tmp_path / "a.tar.gz")],
            fetch_range=fetch,
            resolve_url=lambda url: url,
            connections=1,
            part_bytes=len(_BLOB),
        )

    assert len(calls) == prepare.FETCH_ATTEMPTS


def test_host_guard_blocks_lookups_of_the_fetched_hosts_only(prepare: ModuleType) -> None:
    guard = prepare.host_guard({"www.cs.toronto.edu", "cave.cs.toronto.edu"})

    with pytest.raises(RuntimeError, match="cave.cs.toronto.edu"):
        guard("socket.getaddrinfo", ("cave.cs.toronto.edu", 443, 0, 0, 0))
    with pytest.raises(RuntimeError, match="www.cs.toronto.edu"):
        guard("socket.getaddrinfo", (b"www.cs.toronto.edu", 443, 0, 0, 0))
    assert guard("socket.getaddrinfo", ("storage.googleapis.com", 443, 0, 0, 0)) is None
    assert guard("open", ("cave.cs.toronto.edu", "r", 0)) is None


def test_tfds_archives_reads_the_checksums_tfds_registers(prepare: ModuleType) -> None:
    import tensorflow_datasets as tfds

    registered = tfds.builder_cls("cifar10").url_infos
    assert registered is not None

    archives = prepare.tfds_archives("cifar10")

    assert {archive.url for archive, _ in archives} == set(registered)
    for archive, filename in archives:
        info = registered[archive.url]
        assert (archive.size, archive.sha256, filename) == (
            int(info.size),
            info.checksum,
            info.filename,
        )


def test_keras_archive_path_is_where_get_file_looks_for_a_named_download(
    prepare: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KERAS_HOME", str(tmp_path / "keras-home"))
    assert prepare.keras_archive_path("cifar-10-batches-py-target") == (
        tmp_path / "keras-home" / "datasets" / "cifar-10-batches-py-target_archive"
    )

    monkeypatch.delenv("KERAS_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert prepare.keras_archive_path("cifar-10-batches-py-target") == (
        tmp_path / "home" / ".keras" / "datasets" / "cifar-10-batches-py-target_archive"
    )


def test_prepare_all_lists_the_stale_copies_first_and_replaces_each_before_preparing_it(
    prepare: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KERAS_HOME", str(tmp_path / "keras-home"))
    monkeypatch.setattr(prepare, "TFDS_DATASETS", ("cifar10", "mnist"))
    archives = {
        name: prepare.Archive(url=f"https://host.example/{name}.gz", size=1, sha256="0" * 64)
        for name in prepare.TFDS_DATASETS
    }
    stale = tmp_path / "data" / "mnist" / "3.0.1"
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(prepare, "stale_copies", lambda names: {"mnist": stale})
    monkeypatch.setattr(prepare, "tfds_archives", lambda name: [(archives[name], f"{name}.gz")])
    monkeypatch.setattr(
        prepare,
        "fetch_archives",
        lambda downloads: events.append(("fetch", list(downloads))) or {"host.example"},
    )
    monkeypatch.setattr(prepare, "forbid_hosts", lambda hosts: events.append(("forbid", hosts)))
    monkeypatch.setattr(prepare, "remove_stale_copy", lambda path: events.append(("remove", path)))
    monkeypatch.setattr(
        prepare,
        "prepare_tfds_dataset",
        lambda name, archive_dir: events.append(("tfds", name, archive_dir)),
    )
    monkeypatch.setattr(
        prepare, "prepare_keras_dataset", lambda name: events.append(("keras", name))
    )

    monkeypatch.setattr(
        prepare.LOGGER, "info", lambda message, *args: events.append(("log", message % args))
    )

    prepare.prepare_all(tmp_path / "work")

    keras = prepare.KERAS_DATASETS["cifar10"]
    work = tmp_path / "work" / "tfds"
    assert events[0] == ("log", f"Replacing these copies with ArrayRecord: {stale}")
    assert [event for event in events if event[0] != "log"] == [
        (
            "fetch",
            [
                (archives["cifar10"], work / "cifar10" / "cifar10.gz"),
                (archives["mnist"], work / "mnist" / "mnist.gz"),
                (keras.archive, prepare.keras_archive_path(keras.fname)),
            ],
        ),
        ("forbid", {"host.example"}),
        ("tfds", "cifar10", work / "cifar10"),
        ("remove", stale),
        ("tfds", "mnist", work / "mnist"),
        ("keras", "cifar10"),
    ]
