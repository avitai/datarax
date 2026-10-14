"""A decoded copy's manifest: reading it, and checking a copy against what its source declares."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from datarax.sources._field_paths import by_field
from datarax.sources.decoded_copy._layout import COPY_FORMAT, MANIFEST, unit_count, unit_rows
from datarax.sources.decoded_copy._units import content_digest, unit_digest
from datarax.sources.streaming_disk_source import DiskColumn


def flat(value: Any) -> dict[str, Any]:
    """A JSON value's fields by name: mappings are levels, anything else (a list too) one value.

    An empty mapping has no field, so it differs from another mapping in that mapping's fields.
    """
    return by_field(value, is_leaf=lambda node: not isinstance(node, Mapping))


def as_json(identity: Mapping[str, Any]) -> Any:
    """An identity's ``source`` and ``decoder`` as JSON stores them (tuples become lists)."""
    return json.loads(
        json.dumps({"source": identity.get("source"), "decoder": identity.get("decoder")})
    )


def identity_differences(found: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """The fields in which two identities differ, as paths (``"decoder/pillow"``).

    Both are compared as JSON stores them, so a manifest's identity compares with a source's.

    Args:
        found: One identity (a copy's manifest, or a source's).
        expected: Another.

    Returns:
        The differing paths, sorted; empty when they match.
    """
    left = flat(as_json(found))
    right = flat(as_json(expected))
    return sorted(path for path in set(left) | set(right) if left.get(path) != right.get(path))


def read_manifest(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """A finished copy's manifest, refused when absent or of another format.

    Args:
        directory: The copy's directory.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If there is no manifest: the copy is incomplete.
        ValueError: If the copy is of another format, naming both.
    """
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"{directory} has no manifest.json, so the copy is incomplete (being written, or its "
            "writer stopped): write or rebuild it with datarax.pipeline.materialize"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != COPY_FORMAT:
        raise ValueError(
            f"{directory} is a decoded copy of format {manifest.get('format')}, and this datarax "
            f"reads format {COPY_FORMAT}: rebuild it"
        )
    return manifest


def verify_content(
    directory: Path, manifest: Mapping[str, Any], columns: Sequence[DiskColumn]
) -> None:
    """Refuse a copy whose columns' bytes do not give its content digest (a full read)."""
    length, unit_records = manifest["length"], manifest["unit_records"]
    maps = [
        np.memmap(c.file, dtype=np.dtype(c.dtype), mode="r", offset=c.offset, shape=c.shape)
        for c in columns
    ]
    digests = []
    for unit in range(unit_count(length, unit_records)):
        rows = unit_rows(unit, length, unit_records)
        digests.append(
            unit_digest(
                {c.name: m[rows.start : rows.stop] for c, m in zip(columns, maps, strict=True)}
            )
        )
    del maps
    found = content_digest(digests)
    if found != manifest["content"]["digest"]:
        raise ValueError(
            f"{directory}'s columns give the content digest {found}, not the "
            f"{manifest['content']['digest']} it was written with: a file was changed; rebuild it"
        )
