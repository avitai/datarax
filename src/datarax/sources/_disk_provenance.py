"""Provenance on disk: a record's strings and objects, one ArrayRecord record of UTF-8 JSON each.

:class:`DiskProvenance` names a file of them for a :class:`~datarax.sources.StreamingDiskSource`,
which reads a record's provenance back with :func:`decode_provenance`; a decoded copy writes them
with :func:`encode_provenance`. Text, bytes, numbers, bools and ``None`` are stored exactly, bytes
base64-encoded under ``"bytes"`` (:data:`PROVENANCE_ENCODING`).
"""

from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiskProvenance:
    """Records' provenance stored on disk: one ArrayRecord file, one encoded record per record.

    Each record is its strings and objects as UTF-8 JSON, ``bytes`` values base64-encoded under
    ``"bytes"`` (:func:`~datarax.sources.decoded_copy.encode_provenance`), in record order.

    Attributes:
        file: The ArrayRecord file.
        records: The records it holds, one per record of the source.
    """

    file: str
    records: int


PROVENANCE_ENCODING = "utf-8 json; bytes as base64 under 'bytes'"
"""How :class:`DiskProvenance` stores a record (:func:`encode_provenance`)."""


def encode_provenance(record: Mapping[str, Any]) -> bytes:
    """One record's provenance as UTF-8 JSON, ``bytes`` values base64-encoded under ``"bytes"``.

    Args:
        record: The record's strings and objects, by field path.

    Returns:
        The encoded record.

    Raises:
        TypeError: If a value is not text, bytes, a number, a bool or ``None``, which JSON holds
            exactly, naming the field.
    """
    values: dict[str, Any] = {}
    raw: dict[str, str] = {}
    for field, value in record.items():
        if isinstance(value, bytes):
            raw[field] = base64.b64encode(value).decode("ascii")
        elif value is None or isinstance(value, str | bool | int | float):
            values[field] = value
        else:
            raise TypeError(
                f"provenance field {field!r} holds a {type(value).__name__}, which a decoded "
                "copy cannot store exactly: it keeps text, bytes, numbers, bools and None"
            )
    return json.dumps({"values": values, "bytes": raw}, ensure_ascii=False).encode("utf-8")


def decode_provenance(encoded: bytes) -> dict[str, Any]:
    """A record's provenance as :func:`encode_provenance` stored it.

    Args:
        encoded: The stored record.

    Returns:
        The record's values by field path, ``bytes`` restored.
    """
    stored = json.loads(encoded.decode("utf-8"))
    raw = {field: base64.b64decode(text) for field, text in stored["bytes"].items()}
    return {**stored["values"], **raw}
