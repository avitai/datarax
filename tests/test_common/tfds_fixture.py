"""An offline TFDS dataset for the TFDS source tests, prepared on disk by a command of its own.

Preparing a TFDS dataset imports TensorFlow, which a process reading TFDS data must never hold,
so the fixture is prepared in a process of its own, before the tests that read it:

    python -m tests.test_common.tfds_fixture <directory>

It writes three TFDS data directories under ``<directory>``:

* ``array_record``: :data:`FIXTURE` prepared as ArrayRecord, the format the eager source reads;
* ``tfrecord``: :data:`FIXTURE` prepared as TFRecord, the format the streaming source reads and
  the eager source refuses;
* ``ragged``: :data:`RAGGED` prepared as ArrayRecord, whose images differ in shape.

:data:`FIXTURE` holds an 8x8 RGB image, a class label, a text name and a nested float score per
record, with ``("image", "label")`` as its supervised keys, in a ``train`` split of
:data:`TRAIN_RECORDS` records and a ``test`` split of :data:`TEST_RECORDS`. The builders are
defined only inside :func:`prepare`, so importing this module registers no dataset with TFDS and
a reading process opens the prepared copy from its files, as it would any prepared dataset.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FIXTURE = "datarax_tfds_fixture"
RAGGED = "datarax_tfds_ragged"
TRAIN_RECORDS = 20
TEST_RECORDS = 6
NUM_CLASSES = 3
IMAGE_SHAPE = (8, 8, 3)
COMMAND = "python -m tests.test_common.tfds_fixture"
DIRECTORY_VARIABLE = "DATARAX_TFDS_FIXTURE_DIR"


@dataclass(frozen=True, slots=True, kw_only=True)
class TFDSFixture:
    """Where the prepared fixture lives.

    Attributes:
        root: The directory :func:`prepare` wrote.
        array_record: The data directory holding :data:`FIXTURE` as ArrayRecord.
        tfrecord: The data directory holding :data:`FIXTURE` as TFRecord.
        ragged: The data directory holding :data:`RAGGED` as ArrayRecord.
    """

    root: Path
    array_record: Path
    tfrecord: Path
    ragged: Path

    @classmethod
    def at(cls, root: Path) -> TFDSFixture:
        """The fixture :func:`prepare` wrote under ``root``, checked to be there.

        Args:
            root: The directory given to :func:`prepare`.

        Returns:
            The fixture's data directories.

        Raises:
            FileNotFoundError: If a data directory does not hold its prepared dataset, naming the
                command that prepares it.
        """
        fixture = cls(
            root=root,
            array_record=root / "array_record",
            tfrecord=root / "tfrecord",
            ragged=root / "ragged",
        )
        for data_dir, name in (
            (fixture.array_record, FIXTURE),
            (fixture.tfrecord, FIXTURE),
            (fixture.ragged, RAGGED),
        ):
            if not any((data_dir / name).glob("*/dataset_info.json")):
                raise FileNotFoundError(
                    f"the TFDS test fixture {name} is not prepared in {data_dir}; prepare it with "
                    f"`{COMMAND} {root}` (it needs TensorFlow: the tfds extra)"
                )
        return fixture


def prepare(root: Path) -> TFDSFixture:
    """Prepare the fixture's three data directories under ``root``; this imports TensorFlow.

    Args:
        root: The directory to write.

    Returns:
        The prepared fixture.
    """
    import tensorflow_datasets as tfds  # noqa: PLC0415 - the builders exist only while preparing

    class DataraxTfdsFixture(tfds.core.GeneratorBasedBuilder):
        """Fixed-shape records with every kind of feature the eager source handles."""

        VERSION = tfds.core.Version("1.0.0")

        def _info(self) -> tfds.core.DatasetInfo:
            features = tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=IMAGE_SHAPE),
                    "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                    "name": tfds.features.Text(),
                    "meta": {"score": tfds.features.Tensor(shape=(2,), dtype=np.float32)},
                }
            )
            return tfds.core.DatasetInfo(
                builder=self, features=features, supervised_keys=("image", "label")
            )

        def _split_generators(self, dl_manager: object) -> dict[str, object]:
            del dl_manager
            return {
                "train": self._generate_examples(TRAIN_RECORDS, seed=0),
                "test": self._generate_examples(TEST_RECORDS, seed=1),
            }

        def _generate_examples(self, count: int, *, seed: int) -> object:
            rng = np.random.default_rng(seed)
            for i in range(count):
                yield (
                    i,
                    {
                        "image": rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
                        "label": i % NUM_CLASSES,
                        "name": f"record_{seed}_{i:02d}",
                        "meta": {"score": np.array([i, -i], np.float32)},
                    },
                )

    class DataraxTfdsRagged(tfds.core.GeneratorBasedBuilder):
        """Records whose images differ in shape, which no static batch shape holds."""

        VERSION = tfds.core.Version("1.0.0")

        def _info(self) -> tfds.core.DatasetInfo:
            features = tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
                }
            )
            return tfds.core.DatasetInfo(builder=self, features=features)

        def _split_generators(self, dl_manager: object) -> dict[str, object]:
            del dl_manager
            return {"train": self._generate_examples()}

        def _generate_examples(self) -> object:
            rng = np.random.default_rng(2)
            for i in range(4):
                side = 4 + 2 * i
                yield (
                    i,
                    {
                        "image": rng.integers(0, 256, (side, side, 3), dtype=np.uint8),
                        "label": i % NUM_CLASSES,
                    },
                )

    tfds.disable_progress_bar()
    DataraxTfdsFixture(
        data_dir=root / "array_record", file_format="array_record"
    ).download_and_prepare()
    DataraxTfdsFixture(data_dir=root / "tfrecord", file_format="tfrecord").download_and_prepare()
    DataraxTfdsRagged(data_dir=root / "ragged", file_format="array_record").download_and_prepare()
    return TFDSFixture.at(root)


def main() -> None:
    """Prepare the fixture under the directory given as the only argument."""
    if len(sys.argv) != 2:  # noqa: PLR2004 - the program name and one directory
        raise SystemExit(f"usage: {COMMAND} <directory>")
    prepare(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
