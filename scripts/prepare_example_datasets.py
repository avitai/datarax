#!/usr/bin/env python3
"""Download and prepare the example datasets that CI caches for the long-running example tier.

CIFAR-10 is served from https://www.cs.toronto.edu, which sends each connection about 100 KB/s
and resets a client's connections beyond eight. The loaders the examples call, TFDS and keras,
each download over a single connection, which takes about 25 minutes per archive. This script
fetches the archives itself, in byte ranges over one shared pool of eight connections, verifies
each archive's SHA-256, and places it where its loader reuses a downloaded archive; the loaders
then only extract and prepare. Once the archives are fetched, a lookup of their hosts fails the
run, so a loader that would download again cannot slow it silently.

Usage:
    uv run python scripts/prepare_example_datasets.py
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from substrax.runtime import configure_entry_point_logging


LOGGER = logging.getLogger(__name__)

# The host resets a client's connections beyond eight; eight concurrent ranges finish without one.
HOST_CONNECTION_LIMIT = 8
PART_BYTES = 4 * 2**20
FETCH_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60.0

type RangeFetcher = Callable[[str, int, int], bytes]
type AuditHook = Callable[[str, tuple[object, ...]], None]


@dataclass(frozen=True, slots=True, kw_only=True)
class Archive:
    """An archive a dataset loader downloads, with the size and SHA-256 the loader verifies."""

    url: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class KerasArchive:
    """The archive a ``keras.datasets`` loader downloads and the ``fname`` it gives ``get_file``."""

    archive: Archive
    fname: str


# TFDS builders the examples load; their archives come from TFDS's registered checksums.
TFDS_DATASETS = ("cifar10",)

# keras.datasets modules the examples load, as their load_data() calls get_file.
KERAS_DATASETS = {
    "cifar10": KerasArchive(
        archive=Archive(
            url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            size=170_498_071,
            sha256="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        ),
        fname="cifar-10-batches-py-target",
    ),
}


def tfds_archives(name: str) -> list[tuple[Archive, str]]:
    """The archives a TFDS builder downloads, with the file names it reuses from ``manual_dir``.

    Args:
        name: The TFDS builder name.

    Returns:
        Each registered archive and the file name TFDS looks for.
    """
    import tensorflow_datasets as tfds

    url_infos = tfds.builder_cls(name).url_infos or {}
    return [
        (Archive(url=url, size=int(info.size), sha256=info.checksum), info.filename or "")
        for url, info in url_infos.items()
    ]


def keras_archive_path(fname: str) -> Path:
    """Where ``keras.utils.get_file(fname=..., extract=True)`` keeps the downloaded archive.

    Args:
        fname: The ``fname`` the loader passes, without an extension.

    Returns:
        The archive path under ``$KERAS_HOME``, or ``~/.keras`` when it is unset.
    """
    home = Path(os.environ["KERAS_HOME"]) if "KERAS_HOME" in os.environ else Path.home() / ".keras"
    return home / "datasets" / f"{fname}_archive"


def plan_ranges(size: int, part_bytes: int) -> list[tuple[int, int]]:
    """Split ``size`` bytes into inclusive byte ranges of at most ``part_bytes``.

    Args:
        size: Total number of bytes.
        part_bytes: Largest range length.

    Returns:
        ``(first, last)`` byte offsets of each range, in order.
    """
    return [(start, min(start + part_bytes, size) - 1) for start in range(0, size, part_bytes)]


def resolve_url(url: str) -> str:
    """Follow ``url``'s redirects once, so each range request goes to the serving host.

    Args:
        url: The archive's registered URL.

    Returns:
        The URL that serves the archive.
    """
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # nosec B310
        return response.url


def fetch_range(url: str, first: int, last: int) -> bytes:
    """Fetch bytes ``first`` through ``last`` of ``url``.

    Args:
        url: The archive URL.
        first: Offset of the first byte.
        last: Offset of the last byte.

    Returns:
        The requested bytes.

    Raises:
        ValueError: If the server does not answer with the partial content requested.
    """
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # nosec B310
        body = response.read()
        status = response.status
    if status != 206 or len(body) != last - first + 1:
        raise ValueError(
            f"{url} answered bytes {first}-{last} with HTTP {status} and {len(body)} bytes; "
            "expected 206 Partial Content"
        )
    return body


def _fetch_with_retries(fetch: RangeFetcher, url: str, first: int, last: int) -> bytes:
    """Fetch one range, fetching it again after a dropped connection until the last attempt."""
    for attempt in range(1, FETCH_ATTEMPTS):
        try:
            return fetch(url, first, last)
        except (OSError, http.client.HTTPException) as error:
            LOGGER.warning("Bytes %d-%d of %s failed (%s); retrying", first, last, url, error)
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    return fetch(url, first, last)


def _fetch_part(fetch: RangeFetcher, url: str, path: Path, first: int, last: int) -> int:
    """Fetch one range and write it at its offset in ``path``; returns the bytes written."""
    body = _fetch_with_retries(fetch, url, first, last)
    with path.open("r+b") as file:
        os.pwrite(file.fileno(), body, first)
    return len(body)


def _sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archives(
    downloads: Sequence[tuple[Archive, Path]],
    *,
    fetch_range: RangeFetcher = fetch_range,
    resolve_url: Callable[[str], str] = resolve_url,
    connections: int = HOST_CONNECTION_LIMIT,
    part_bytes: int = PART_BYTES,
) -> set[str]:
    """Fetch archives in byte ranges over one pool of connections and verify their digests.

    Args:
        downloads: Each archive and the path to write it to.
        fetch_range: Fetches an inclusive byte range of a URL.
        resolve_url: Returns the URL that serves an archive after redirects.
        connections: Most ranges fetched at once, across every archive.
        part_bytes: Largest range fetched in one request.

    Returns:
        The host names contacted, registered and redirected.

    Raises:
        ValueError: If an archive's SHA-256 is not the one its loader verifies.
    """
    hosts: set[str] = set()
    parts: list[tuple[int, str, Path, int, int]] = []
    for index, (archive, path) in enumerate(downloads):
        served_url = resolve_url(archive.url)
        hosts |= {urlsplit(archive.url).hostname or "", urlsplit(served_url).hostname or ""}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            file.truncate(archive.size)
        ranges = plan_ranges(archive.size, part_bytes)
        parts += [(index, served_url, path, first, last) for first, last in ranges]

    started = time.perf_counter()
    fetched = [0] * len(downloads)
    logged_quarters = [0] * len(downloads)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = {pool.submit(_fetch_part, fetch_range, *part[1:]): part[0] for part in parts}
        for future in as_completed(futures):
            index = futures[future]
            fetched[index] += future.result()
            archive = downloads[index][0]
            quarter = 4 * fetched[index] // archive.size
            if quarter > logged_quarters[index]:
                logged_quarters[index] = quarter
                seconds = time.perf_counter() - started
                LOGGER.info(
                    "%s: %d%% of %.0f MiB after %.0f s",
                    archive.url,
                    25 * quarter,
                    archive.size / 2**20,
                    seconds,
                )

    for archive, path in downloads:
        digest = _sha256(path)
        if digest != archive.sha256:
            raise ValueError(
                f"{archive.url} downloaded with SHA-256 {digest}; expected {archive.sha256}"
            )
    return hosts - {""}


def host_guard(hosts: Collection[str]) -> AuditHook:
    """An audit hook that fails any name lookup of ``hosts``.

    Args:
        hosts: Host names nothing may reach once their archives are fetched.

    Returns:
        A hook for :func:`sys.addaudithook`.
    """

    def guard(event: str, args: tuple[object, ...]) -> None:
        if event != "socket.getaddrinfo" or not args:
            return
        host = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if host in hosts:
            raise RuntimeError(
                f"A loader tried to reach {host} after its archive was fetched; the archive is not "
                "where the loader looks for it"
            )

    return guard


def forbid_hosts(hosts: Collection[str]) -> None:
    """Fail every later name lookup of ``hosts`` in this process.

    Args:
        hosts: Host names nothing may reach from now on.
    """
    sys.addaudithook(host_guard(frozenset(hosts)))


def prepare_tfds_dataset(name: str, archive_dir: Path) -> None:
    """Prepare a TFDS dataset from archives already in ``archive_dir``.

    Args:
        name: The TFDS builder name.
        archive_dir: Directory holding the builder's archives under their registered file names;
            TFDS also extracts there, so only the prepared dataset lands in its data directory.
    """
    import tensorflow_datasets as tfds

    tfds.disable_progress_bar()
    tfds.builder(name).download_and_prepare(
        download_dir=archive_dir,
        download_config=tfds.download.DownloadConfig(manual_dir=archive_dir),
    )


def prepare_keras_dataset(name: str) -> None:
    """Load a keras dataset once, so it verifies and extracts the archive already in its cache.

    Args:
        name: The module name under ``keras.datasets``.
    """
    from keras import datasets

    getattr(datasets, name).load_data()


def _log_prepared(loader: str, name: str, started: float) -> None:
    """Log that a dataset is prepared and how long it took since ``started``."""
    LOGGER.info("Prepared %s dataset %s in %.0f s", loader, name, time.perf_counter() - started)


def prepare_all(work_dir: Path) -> None:
    """Fetch every archive, forbid their hosts, then prepare every dataset from them.

    Args:
        work_dir: Scratch directory for TFDS archives and extractions, which CI does not cache.
    """
    tfds_dirs = {name: work_dir / "tfds" / name for name in TFDS_DATASETS}
    downloads = [
        (archive, tfds_dirs[name] / filename)
        for name in TFDS_DATASETS
        for archive, filename in tfds_archives(name)
    ]
    downloads += [
        (keras.archive, keras_archive_path(keras.fname)) for keras in KERAS_DATASETS.values()
    ]

    forbid_hosts(fetch_archives(downloads))

    for name, archive_dir in tfds_dirs.items():
        started = time.perf_counter()
        prepare_tfds_dataset(name, archive_dir)
        _log_prepared("tfds", name, started)
    for name in KERAS_DATASETS:
        started = time.perf_counter()
        prepare_keras_dataset(name)
        _log_prepared("keras", name, started)


def main() -> None:
    """Prepare the example datasets."""
    configure_entry_point_logging()
    with tempfile.TemporaryDirectory() as work_dir:
        prepare_all(Path(work_dir))


if __name__ == "__main__":
    main()
