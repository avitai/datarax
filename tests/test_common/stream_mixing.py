"""T15: how well the TFDS stream's order mixes, against tf.data's training read over the same files.

The reference is TFDS's documented training read: ``as_dataset(split, shuffle_files=True,
read_config=ReadConfig(shuffle_seed=seed))`` then ``shuffle(W, seed)``, over several passes. It is
recorded in a process of its own, since it needs TensorFlow:

    python -m tests.test_common.stream_mixing <directory> <records> <seeds> <shards,...>

which prepares, for each shard count, a dataset of ``records`` records (an integer ``x``) as
TFRecord with that many shards, and writes the positions tf.data serves for :data:`PASSES` passes
at each buffer size of :func:`buffer_sizes` and each seed. A position is the record's place in the
files read in order. The stream's order is read the same way, from the ids it names records by.

The measures, per pass and over seeds (:func:`measures`):

* ``tv_sorted``: the mean per-batch label total-variation distance from the label distribution,
  on a class-sorted layout (record ``p`` has label ``p * CLASSES // records``), the case where a
  stream's order has to break a grouped layout; lower mixes better.
* ``mates``: per record, how many of its pass-0 batch-mates are its batch-mates again in pass 1;
  lower means batches recur less across epochs.
* ``displacement``: the mean ``|served position - file position|``; higher mixes more.
* ``first_shards``: the number of shards among the first ``records / shards`` records served;
  higher mixes more.

The stream's order passes when no measure is worse than the reference's by more than the seed
spread: twice the larger of the two standard deviations over seeds (:func:`worse_measures`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PASSES = 2
CLASSES = 10
BATCH = 32
REFERENCE_FILE = "tfdata_orders.json"
NAME = "datarax_stream_mixing_s{shards}"
LOWER_IS_BETTER = ("tv_sorted", "mates")
HIGHER_IS_BETTER = ("displacement", "first_shards")


def buffer_sizes(records: int) -> tuple[int, ...]:
    """The buffer sizes measured: a hundredth and a tenth of the records.

    A buffer of every record is not measured: there both reads serve a uniform permutation of the
    pass, so the two orders are draws of one distribution and the criterion could only fire by
    chance (it did once, at 10,000 records and 20 shards, on a low draw of tf.data's first ten
    seeds). Below that the order's mixing is what its design gives, and that is what is compared.
    """
    return (records // 100, records // 10)


def tv_sorted(order: np.ndarray, records: int) -> float:
    """Mean per-batch label total-variation distance on the class-sorted layout."""
    labels = (np.asarray(order) * CLASSES) // records
    batches = len(order) // BATCH
    counts = np.zeros((batches, CLASSES))
    rows = np.repeat(np.arange(batches), BATCH)
    np.add.at(counts, (rows, labels[: batches * BATCH]), 1)
    return float((0.5 * np.abs(counts / BATCH - 1 / CLASSES).sum(axis=1)).mean())


def mates(first: np.ndarray, second: np.ndarray) -> float:
    """Per record, how many of its batch-mates in ``first`` are its batch-mates in ``second``."""
    size = len(first)
    batch_of = np.empty((2, size), dtype=np.int64)
    batch_of[0, first] = np.arange(size) // BATCH
    batch_of[1, second] = np.arange(size) // BATCH
    _, together = np.unique(batch_of[0] * (size // BATCH + 1) + batch_of[1], return_counts=True)
    return float((together * (together - 1)).sum() / size)


def measures(passes: list[np.ndarray], shard_sizes: list[int]) -> dict[str, float]:
    """The four measures of one seed's passes (each a permutation of the file positions).

    Args:
        passes: The positions served in each pass, :data:`PASSES` of them.
        shard_sizes: The records in each shard, in file order.

    Returns:
        The measures, each averaged over the passes where it is per pass.

    Raises:
        ValueError: If a pass is not a permutation of the positions.
    """
    records = int(sum(shard_sizes))
    shard_of = np.repeat(np.arange(len(shard_sizes)), shard_sizes)
    for order in passes:
        if not np.array_equal(np.sort(order), np.arange(records)):
            raise ValueError("a pass does not serve every record exactly once")
    first = records // len(shard_sizes)
    return {
        "tv_sorted": float(np.mean([tv_sorted(order, records) for order in passes])),
        "mates": mates(passes[0], passes[1]),
        "displacement": float(
            np.mean([np.abs(np.arange(records) - order).mean() for order in passes])
        ),
        "first_shards": float(
            np.mean([len(np.unique(shard_of[order[:first]])) for order in passes])
        ),
    }


def summary(per_seed: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Each measure's mean and standard deviation over seeds."""
    return {
        name: (
            float(np.mean([m[name] for m in per_seed])),
            float(np.std([m[name] for m in per_seed])),
        )
        for name in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER)
    }


def worse_measures(
    stream: dict[str, tuple[float, float]], reference: dict[str, tuple[float, float]]
) -> list[str]:
    """The measures on which the stream mixes worse than the reference beyond the seed spread."""
    worse = []
    for name, (mean, spread) in stream.items():
        reference_mean, reference_spread = reference[name]
        margin = 2 * max(spread, reference_spread)
        sign = 1 if name in LOWER_IS_BETTER else -1
        if sign * (mean - reference_mean) > margin:
            worse.append(name)
    return worse


def prepare_reference(root: Path, records: int, seeds: int, shard_counts: list[int]) -> None:
    """Prepare each dataset as TFRecord and record tf.data's training-read orders; imports TF.

    Args:
        root: The directory to write the datasets and :data:`REFERENCE_FILE` in.
        records: Records per dataset.
        seeds: Seeds per buffer size.
        shard_counts: The shard counts, one dataset each.
    """
    import tensorflow as tf  # noqa: PLC0415 - the reference is tf.data's, in its own process

    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds  # noqa: PLC0415

    reference: dict[str, Any] = {"records": records, "seeds": seeds, "datasets": {}}
    for shards in shard_counts:
        builder = _builder(tfds, shards, records)(data_dir=root, file_format="tfrecord")
        builder.download_and_prepare(
            download_config=tfds.download.DownloadConfig(num_shards=shards)
        )
        instructions = builder.info.splits["train"].file_instructions
        names = [Path(i.filename).name for i in instructions]
        sizes = [i.examples_in_shard for i in instructions]
        start = dict(zip(names, np.cumsum([0, *sizes])[:-1].tolist(), strict=True))
        orders: dict[str, list[list[int]]] = {}
        for size in buffer_sizes(records):
            for seed in range(seeds):
                dataset = builder.as_dataset(
                    split="train",
                    shuffle_files=True,
                    read_config=tfds.ReadConfig(add_tfds_id=True, shuffle_seed=seed),
                ).shuffle(size, seed=seed, reshuffle_each_iteration=True)
                ids = dataset.map(lambda record: record["tfds_id"])
                orders[f"{size}:{seed}"] = [
                    [
                        start[t.decode().rsplit("__", 1)[0]] + int(t.decode().rsplit("__", 1)[1])
                        for t in tfds.as_numpy(ids)
                    ]
                    for _ in range(PASSES)
                ]
        reference["datasets"][str(shards)] = {"sizes": sizes, "orders": orders}
    (root / REFERENCE_FILE).write_text(json.dumps(reference))


def _builder(tfds: Any, shards: int, records: int) -> Any:
    class Fixture(tfds.core.GeneratorBasedBuilder):
        """``records`` integers, the dataset of one shard count."""

        name = NAME.format(shards=shards)
        VERSION = tfds.core.Version("1.0.0")

        def _info(self) -> Any:
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict(
                    {"x": tfds.features.Tensor(shape=(), dtype=np.int64)}
                ),
            )

        def _split_generators(self, dl_manager: Any) -> Any:
            del dl_manager
            return {"train": self._generate_examples()}

        def _generate_examples(self) -> Any:
            for i in range(records):
                yield i, {"x": i}

    return Fixture


def main() -> None:
    """Prepare the datasets and the reference under the directory given."""
    root, records, seeds, shards = sys.argv[1:5]
    prepare_reference(Path(root), int(records), int(seeds), [int(s) for s in shards.split(",")])


if __name__ == "__main__":
    main()
