"""The descriptor helpers of ``tests.test_common.host_resources``, on every platform."""

from __future__ import annotations

import os
from pathlib import Path

from tests.test_common.host_resources import descriptor_path, open_descriptors


def test_a_descriptor_names_the_file_it_was_opened_on(tmp_path: Path) -> None:
    path = tmp_path / "column.partial"
    path.write_bytes(b"x")

    with path.open("rb") as handle:
        named = descriptor_path(handle.fileno())

    assert os.path.realpath(named) == os.path.realpath(path)


def test_an_open_file_is_held_and_a_closed_one_is_not(tmp_path: Path) -> None:
    held, closed = tmp_path / "held", tmp_path / "closed"
    held.write_bytes(b"x")
    closed.write_bytes(b"x")
    closed.open("rb").close()

    with held.open("rb"):
        found = open_descriptors([held, closed])

    assert found == [os.path.realpath(held)]
    assert open_descriptors([held]) == []
