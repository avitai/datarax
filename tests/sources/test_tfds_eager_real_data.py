"""The eager TFDS source over the real example datasets, prepared as ArrayRecord.

The datasets are the ones ``scripts/prepare_example_datasets.py`` prepares, in TFDS's data
directory (``TFDS_DATA_DIR``, else ``~/tensorflow_datasets``); CI's long-running job restores
them. The records must be bit for bit the ones the TensorFlow loader this source replaced read
from the TFRecord copies: the oracle below was recorded with that loader (datarax ``eb8486b``,
``TFDSEagerSource`` over ``tfds.load``) as each column's dtype, shape and SHA-256, the label also
as int64 values, and the CIFAR-10 record ids in order. The build is linear in the records and far
under C4c-6's measurement of that loader (384 records/s at 5,000).
"""

from __future__ import annotations

import hashlib
import statistics
import time

import jax.numpy as jnp
import numpy as np
import pytest

from datarax.sources import TFDSEagerConfig, TFDSEagerSource


pytestmark = [pytest.mark.tfds, pytest.mark.slow]

_ORACLE_RECORDS = 5000
# split -> column -> (stored dtype, shape, SHA-256 of the bytes as eb8486b held them, and of the
# values as int64 where the column is an integer).
_ORACLE = {
    ("mnist", "train[:5000]"): {
        "image": (
            "|u1",
            (5000, 28, 28, 1),
            "199e3a97f1c280a973b8f6871c76787969d873ba9cab50966c108cb50fdfe647",
        ),
        "label": (
            "<i4",
            (5000,),
            "e97ca64be869f8603a48e1ab9a8c66911c506563da64ec9db159831a7cb23055",
        ),
    },
    ("mnist", "test[:5000]"): {
        "image": (
            "|u1",
            (5000, 28, 28, 1),
            "79bf4fff2dd8bdff429fa34bccc28d3f4d1d8caa29c97a6e6952592e1df3a662",
        ),
        "label": (
            "<i4",
            (5000,),
            "bd37cf743e1fd5c1a49800297856938f65319871a7d899b724d714b5914fab5b",
        ),
    },
    ("cifar10", "train[:5000]"): {
        "image": (
            "|u1",
            (5000, 32, 32, 3),
            "018a97383fd7bcefcca1855bd31b7ccd3874cd553d948a8d5966bf12450630eb",
        ),
        "label": (
            "<i4",
            (5000,),
            "c0dd01b0e57c38ebee11d45ed371a579e7cefd7cb90cf9c5af0f1549b594e9b2",
        ),
    },
}
# SHA-256 of the 5,000 CIFAR-10 ids of train[:5000], newline-joined, in order.
_CIFAR10_IDS = "826f1dad26a11a20249e95c71c6f1532d857fb35e648694f9758d4c5c1747679"

# C4c-6 measured the TensorFlow loader at 13.0 s for 5,000 CIFAR-10 records.
_C4C6_RECORDS_PER_SECOND = 384
_BUILD_SIZES = (5000, 10000, 20000)
_REPEATS = 3


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.mark.parametrize(("name", "split"), sorted(_ORACLE))
def test_the_records_are_bit_identical_to_the_tensorflow_loader(name: str, split: str) -> None:
    source = TFDSEagerSource(
        TFDSEagerConfig(name=name, split=split, include_keys={"image", "label"})
    )

    expected = _ORACLE[(name, split)]
    image_dtype, image_shape, image_hash = expected["image"]
    image = source.data["image"]
    assert (image.dtype.str, image.shape) == (image_dtype, image_shape)
    assert _sha256(image) == image_hash
    # The label is held as TFDS stores it (int64) and equals the old loader's values; on the
    # device, with 64-bit types off, it is the int32 the old loader held.
    label = source.data["label"]
    _, label_shape, label_hash = expected["label"]
    assert (label.dtype, label.shape) == (np.dtype(np.int64), label_shape)
    assert _sha256(label) == label_hash
    assert jnp.asarray(label).dtype == jnp.int32


def test_the_cifar10_ids_are_the_provenance_in_the_old_order() -> None:
    source = TFDSEagerSource(TFDSEagerConfig(name="cifar10", split=f"train[:{_ORACLE_RECORDS}]"))

    ids = [record["id"] for record in source._provenance.value]
    assert len(ids) == _ORACLE_RECORDS
    assert ids[:3] == [b"train_16399", b"train_01680", b"train_47917"]
    assert hashlib.sha256(b"\n".join(ids)).hexdigest() == _CIFAR10_IDS
    assert "id" not in source.data


def _build_seconds(records: int) -> float:
    config = TFDSEagerConfig(name="cifar10", split=f"train[:{records}]")
    started = time.perf_counter()
    source = TFDSEagerSource(config)
    seconds = time.perf_counter() - started
    assert len(source) == records
    return seconds


def test_the_build_is_linear_in_the_records_and_under_the_c4c6_bound() -> None:
    """R55: CIFAR-10 at 5k/10k/20k records, three interleaved repeats, medians.

    Linear: at four times the records the time per record is under twice what it is at 5,000,
    where any term quadratic in the records would make it four times (the loader this replaced
    grew from 2.6 ms per record at 5,000 to over 20 ms at 50,000, C4c-6). Each build reads more
    records per second than C4c-6 measured for that loader at 5,000.
    """
    _build_seconds(100)  # opens TFDS and the decoder once, outside the measurement
    seconds: dict[int, list[float]] = {size: [] for size in _BUILD_SIZES}
    for _ in range(_REPEATS):
        for size in _BUILD_SIZES:
            seconds[size].append(_build_seconds(size))
    median = {size: statistics.median(values) for size, values in seconds.items()}

    per_record = {size: median[size] / size for size in _BUILD_SIZES}
    assert per_record[_BUILD_SIZES[-1]] < 2 * per_record[_BUILD_SIZES[0]], seconds
    for size in _BUILD_SIZES:
        assert size / median[size] > _C4C6_RECORDS_PER_SECOND, seconds
