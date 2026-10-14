"""The in-memory TFDS build decodes through the one per-unit writer, bit for bit as it did before.

``TFDSEagerSource`` reads its split with the per-batch reader of the same split and filters and
writes every unit through :func:`~datarax.sources.decoded_copy._units.write_unit`, the writer a
decoded copy and a parallel build use, so there is one decode path. The columns and provenance it
holds equal a golden recorded from the build that read the whole split through TFDS's own reader
(datarax ``460c47a``, ``evidence/w2b/c5a5/impl/commit3/3b/golden_tfds_eager_at_460c47a.json``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tensorflow_datasets.core.data_sources import base as tfds_data_sources

from datarax.core.index_words import to_words
from datarax.sources import from_tfds, TFDSEagerConfig, TFDSEagerSource
from datarax.sources.array_record_source import ArrayRecordSourceModule
from datarax.sources.decoded_copy import _layout, _units
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.tfds_golden import digests, EAGER_BUILD


pytestmark = pytest.mark.tfds


def _eager(fixture: TFDSFixture, **config: Any) -> TFDSEagerSource:
    options = {"name": FIXTURE, "split": "train", "data_dir": str(fixture.array_record)}
    return TFDSEagerSource(TFDSEagerConfig(**{**options, **config}))


@pytest.mark.parametrize("case", list(EAGER_BUILD))
def test_the_build_equals_the_golden_of_the_whole_split_read(
    tfds_fixture: TFDSFixture, case: str
) -> None:
    config, columns, names = EAGER_BUILD[case]

    source = _eager(tfds_fixture, **config)

    assert digests(source.data) == columns
    words = to_words(np.arange(len(source), dtype=np.uint64))
    provenance = [dict(record) for record in source.provenance(words)]
    assert [record.get("name", b"").decode() for record in provenance if record] == names
    assert all(isinstance(record["name"], bytes) for record in provenance if record)


def test_one_writer_decodes_every_unit_and_tfds_reads_no_whole_split(
    tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[range] = []
    whole: list[int] = []
    write_unit = _units.write_unit
    read = tfds_data_sources.BaseDataSource.__getitems__

    def counting_writer(source: Any, rows: range, *args: Any, **kwargs: Any) -> Any:
        written.append(rows)
        return write_unit(source, rows, *args, **kwargs)

    def counting_read(self: Any, keys: Any) -> Any:
        whole.append(len(keys))
        return read(self, keys)

    monkeypatch.setattr(_units, "write_unit", counting_writer)
    monkeypatch.setattr(tfds_data_sources.BaseDataSource, "__getitems__", counting_read)

    source = _eager(tfds_fixture)

    units = -(-len(source) // _layout.UNIT_RECORDS)
    assert len(written) == units
    assert [row for rows in written for row in rows] == list(range(len(source)))
    assert whole == [], "TFDS's own reader decodes nothing"


def test_the_build_holds_what_the_per_batch_read_serves(tfds_fixture: TFDSFixture) -> None:
    eager = _eager(tfds_fixture)
    per_batch = from_tfds(
        FIXTURE, "train", data_dir=str(tfds_fixture.array_record), in_memory=False
    )

    words = to_words(np.arange(len(eager), dtype=np.uint64))
    assert isinstance(per_batch, ArrayRecordSourceModule)
    assert digests(per_batch.get_batch(words).data) == digests(eager.get_batch(words).data)
    assert per_batch.provenance(words) == eager.provenance(words)


def test_filters_leaving_no_feature_are_refused_before_decoding(
    tfds_fixture: TFDSFixture,
) -> None:
    with pytest.raises(ValueError, match="produced no records after loading and filtering"):
        _eager(tfds_fixture, include_keys={"absent"})
