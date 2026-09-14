"""Tests for scripts/prepare_example_datasets.py, which fills CI's cache of example datasets."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script, REPO_ROOT


_TFDS_CIFAR10 = re.compile(r'Config\(\s*name="cifar10"|tfds\.(?:load|builder)\("cifar10"')
_KERAS_CIFAR10 = re.compile(r"from keras\.datasets import cifar10\b")
_BLOB = bytes(range(256)) * 4


@pytest.fixture(scope="module")
def prepare() -> ModuleType:
    return load_script("prepare_example_datasets")


def _cifar10_loaders() -> set[tuple[str, str]]:
    """The CIFAR-10 loaders the examples call, found in their sources."""
    loaders: set[tuple[str, str]] = set()
    for path in sorted((REPO_ROOT / "examples").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if _TFDS_CIFAR10.search(text):
            loaders.add(("tfds", "cifar10"))
        if _KERAS_CIFAR10.search(text):
            loaders.add(("keras", "cifar10"))
    return loaders


def test_every_cifar10_loader_the_examples_use_is_prepared(prepare: ModuleType) -> None:
    """CIFAR-10 is served from a slow host, so each loader of it an example uses is cached in CI."""
    loaders = _cifar10_loaders()
    prepared = {("tfds", name) for name in prepare.TFDS_DATASETS} | {
        ("keras", name) for name in prepare.KERAS_DATASETS
    }

    assert loaders == {("tfds", "cifar10"), ("keras", "cifar10")}
    assert loaders <= prepared


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


def test_prepare_all_fetches_then_forbids_the_hosts_then_runs_the_loaders(
    prepare: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KERAS_HOME", str(tmp_path / "keras-home"))
    tfds_archive = prepare.Archive(url="https://host.example/tfds.tar.gz", size=1, sha256="0" * 64)
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(prepare, "tfds_archives", lambda name: [(tfds_archive, "tfds.tar.gz")])
    monkeypatch.setattr(
        prepare,
        "fetch_archives",
        lambda downloads: events.append(("fetch", list(downloads))) or {"host.example"},
    )
    monkeypatch.setattr(prepare, "forbid_hosts", lambda hosts: events.append(("forbid", hosts)))
    monkeypatch.setattr(
        prepare,
        "prepare_tfds_dataset",
        lambda name, archive_dir: events.append(("tfds", name, archive_dir)),
    )
    monkeypatch.setattr(
        prepare, "prepare_keras_dataset", lambda name: events.append(("keras", name))
    )

    prepare.prepare_all(tmp_path / "work")

    keras = prepare.KERAS_DATASETS["cifar10"]
    assert events == [
        (
            "fetch",
            [
                (tfds_archive, tmp_path / "work" / "tfds" / "cifar10" / "tfds.tar.gz"),
                (keras.archive, prepare.keras_archive_path(keras.fname)),
            ],
        ),
        ("forbid", {"host.example"}),
        ("tfds", "cifar10", tmp_path / "work" / "tfds" / "cifar10"),
        ("keras", "cifar10"),
    ]
