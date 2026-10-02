"""T15: the TFDS stream's order mixes at least as well as tf.data's training read (C5b).

The reference is recorded by :mod:`tests.test_common.stream_mixing` in a TensorFlow process of its
own: TFDS's documented training read (``shuffle_files=True`` with ``ReadConfig(shuffle_seed)``, then
``shuffle(W)``) over datasets of 1, 4 and 20 TFRecord shards, two passes, buffers of a hundredth and
a tenth of the records, ten seeds. The stream reads the same files with the pipeline's key of each
seed, and no measure (per-batch label TV on a class-sorted layout, batch-mates recurring across
passes, displacement, shards among the first records) may be worse than the reference's beyond the
seed spread. A failure stops the step for the order to be revisited; the margin is never widened. A
buffer of every record is not compared (see :func:`~tests.test_common.stream_mixing.buffer_sizes`);
over twenty salted sets of ten seeds the criterion held in every cell compared.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import numpy as np
import pytest
from substrax.testing import run_python

from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource
from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common import stream_mixing as mixing


pytestmark = [pytest.mark.tfds, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECORDS = 2000
_SEEDS = 10
_SHARDS = (1, 4, 20)
_PREPARE_SECONDS = 1800.0


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("stream_mixing")
    run_python(
        "from tests.test_common.stream_mixing import main; main()",
        str(root),
        str(_RECORDS),
        str(_SEEDS),
        ",".join(map(str, _SHARDS)),
        timeout=_PREPARE_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    ).check()
    return root, json.loads((root / mixing.REFERENCE_FILE).read_text())


def _stream_passes(
    root: Path, shards: int, size: int, seed: int, shard_sizes: list[int]
) -> list[np.ndarray]:
    source = TFDSStreamingSource(
        TFDSStreamingConfig(
            name=mixing.NAME.format(shards=shards),
            split="train",
            data_dir=str(root),
            shuffle_buffer_size=size,
        )
    )
    files = source.shard_files  # one split, so its files in name order are its files in order
    start = dict(zip(files, np.cumsum([0, *shard_sizes])[:-1].tolist(), strict=True))
    key = jax.random.key(seed)
    passes = []
    for _ in range(mixing.PASSES):
        positions = []
        while (batch := source.get_batch(256, key=key)).batch_size:
            for shard, offset in np.asarray(batch.indices):
                positions.append(start[files[int(shard)]] + int(offset))
        passes.append(np.asarray(positions))
    return passes


@pytest.mark.parametrize("shards", _SHARDS)
def test_the_stream_mixes_no_worse_than_tf_data(reference: tuple[Path, dict], shards: int) -> None:
    root, recorded = reference
    dataset = recorded["datasets"][str(shards)]
    report = {}
    for size in mixing.buffer_sizes(_RECORDS):
        ours = mixing.summary(
            [
                mixing.measures(
                    _stream_passes(root, shards, size, seed, dataset["sizes"]), dataset["sizes"]
                )
                for seed in range(_SEEDS)
            ]
        )
        theirs = mixing.summary(
            [
                mixing.measures(
                    [np.asarray(p) for p in dataset["orders"][f"{size}:{seed}"]], dataset["sizes"]
                )
                for seed in range(_SEEDS)
            ]
        )
        report[size] = (ours, theirs, mixing.worse_measures(ours, theirs))

    assert all(not worse for _, _, worse in report.values()), report
