"""The decoded copy's modules, one responsibility each, and where each name lives.

``decoded_copy`` is a package whose one public name is ``open_decoded_copy``, what users call.
Its internal modules hold the copy's layout (``_layout``: file names, the column layout, units),
reading and writing a unit (``_units``), the files and the lock (``_files``) and the manifest
(``_manifest``); ``datarax.pipeline.materialization`` imports them directly. The provenance codec
is ``_disk_provenance``, below both the package and ``StreamingDiskSource``, which reads it.
"""

from __future__ import annotations

from types import ModuleType

import pytest

import datarax.sources
from datarax.sources import _disk_provenance, decoded_copy, streaming_disk_source
from datarax.sources.decoded_copy import _files, _layout, _manifest, _units


_HOMES: dict[ModuleType, tuple[str, ...]] = {
    _layout: (
        "COPY_FORMAT",
        "UNIT_RECORDS",
        "MANIFEST",
        "MANIFEST_TMP",
        "LOCK",
        "PROVENANCE",
        "PARTIAL",
        "COLUMN",
        "DIGEST",
        "LayoutColumn",
        "ColumnLayout",
        "unit_count",
        "unit_rows",
    ),
    _units: (
        "read_unit",
        "first_read",
        "source_layout",
        "write_unit",
        "refuse_other_fields",
        "checked",
        "unit_digest",
        "content_digest",
        "build_in_memory",
    ),
    _files: (
        "REFUSED_FILESYSTEMS",
        "ProvenanceWriter",
        "CopyLock",
        "refuse_unsupported_filesystem",
        "is_copy_file",
        "read_descriptor",
        "fsync_directory",
        "fsync_file",
        "clear_copy",
        "CopyFiles",
        "reserve",
        "create_column",
        "begin_copy",
        "finish_copy",
    ),
    _manifest: ("flat", "as_json", "identity_differences", "read_manifest", "verify_content"),
    _disk_provenance: (
        "DiskProvenance",
        "PROVENANCE_ENCODING",
        "encode_provenance",
        "decode_provenance",
    ),
}
_PLACED = [(module, name) for module, names in _HOMES.items() for name in names]


@pytest.mark.parametrize(
    ("module", "name"), _PLACED, ids=[f"{m.__name__.rsplit('.', 1)[1]}.{n}" for m, n in _PLACED]
)
def test_each_name_is_defined_in_its_module(module: ModuleType, name: str) -> None:
    """Names inside an internal module are spelled plainly; its leading underscore is enough."""
    assert name in vars(module)
    value = vars(module)[name]
    if isinstance(value, type) or callable(value):
        assert value.__module__ == module.__name__


def test_the_package_s_one_public_name_is_what_users_call() -> None:
    """It defines and exports one name; its internal modules hold the rest."""
    defined = {
        name
        for name, value in vars(decoded_copy).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == decoded_copy.__name__
    }
    assert defined == {"open_decoded_copy"}
    assert decoded_copy.__all__ == ["open_decoded_copy"]


def test_the_codec_has_one_home_below_the_disk_source() -> None:
    """``StreamingDiskSource`` reads provenance with it; the package writes it."""
    for name in _HOMES[_disk_provenance]:
        assert vars(streaming_disk_source).get(name) in (None, vars(_disk_provenance)[name])
    assert streaming_disk_source.decode_provenance is _disk_provenance.decode_provenance


def test_disk_provenance_is_still_exported_where_every_source_is() -> None:
    """Users pass it in a config, so it stays public (a public class from an internal module)."""
    assert datarax.sources.DiskProvenance is _disk_provenance.DiskProvenance
