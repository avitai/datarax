"""Decoded copies written on the calling thread, for the tests of the copy's format and reader.

:func:`write_copy` follows the writer's protocol of :mod:`datarax.sources.decoded_copy` in one
thread: the layout from the source's first record, the reserved ``.partial`` files, every unit
written with :func:`~datarax.sources.decoded_copy._units.write_unit` and its digest, the provenance
appended unit by unit, then :func:`~datarax.sources.decoded_copy._files.finish_copy`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from datarax.core.data_source import (
    DataSourceModule,
    IndexedHostRead,
    known_length,
    SourceIdentified,
)
from datarax.core.index_words import to_words
from datarax.sources.decoded_copy._files import begin_copy, finish_copy
from datarax.sources.decoded_copy._layout import ColumnLayout, UNIT_RECORDS
from datarax.sources.decoded_copy._units import source_layout, unit_digest, write_unit


IDENTITY: dict[str, Any] = {"source": {"kind": "test", "records": 0}, "decoder": {}}
"""An identity for a source that declares none (an in-memory test source)."""


needs_posix_fallocate = pytest.mark.skipif(
    not hasattr(os, "posix_fallocate"),
    reason=(
        f"a decoded copy reserves its files with os.posix_fallocate, which {sys.platform} lacks; "
        "test_without_posix_fallocate_a_copy_is_refused_naming_the_platform covers the refusal"
    ),
)
"""For a test that writes a decoded copy: the writer refuses a platform without the reservation."""


def write_copy(
    source: DataSourceModule,
    directory: Path,
    *,
    identity: Mapping[str, Any] | None = None,
    unit_records: int = UNIT_RECORDS,
) -> ColumnLayout:
    """Write ``source``'s decoded copy into ``directory`` on the calling thread.

    Args:
        source: An indexed source with the stateless host read.
        directory: Where the copy goes.
        identity: The identity the manifest records; the source's own when ``None``.
        unit_records: Records per unit.

    Returns:
        The copy's layout.
    """
    layout = source_layout(source)
    declared = cast(SourceIdentified, source).source_identity() if identity is None else identity
    digests: list[bytes] = []
    with begin_copy(directory, layout, unit_records=unit_records) as files:
        with files.provenance() as provenance:
            for unit in range(files.units):
                with files.rows(unit) as out:
                    records = write_unit(
                        source, files.unit_rows(unit), out, layout=layout, with_provenance=True
                    )
                    digests.append(unit_digest(out))
                provenance.write(records or ())
        finish_copy(files, unit_digests=digests, identity=declared)
    return layout


def every_record(source: DataSourceModule) -> tuple[Any, tuple[Mapping[str, Any], ...]]:
    """``source``'s records in order, read whole: its batch's data and its provenance.

    Args:
        source: An indexed source with the stateless host read.

    Returns:
        The data of one batch of every record, and their provenance.
    """
    words = to_words(np.arange(known_length(source) or 0, dtype=np.uint64))
    return cast(IndexedHostRead, source).get_batch(words).data, tuple(source.provenance(words))
