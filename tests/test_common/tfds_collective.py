"""A two-device all-reduce over a batch a TFDS source read: the program a test runs.

TensorFlow in a JAX process breaks JAX's NCCL collectives on GPUs ("corrupted comm object
detected"), so the process that reads TFDS data for training must not hold it. This program
reads the offline fixture (:mod:`tests.test_common.tfds_fixture`) with ``TFDSEagerSource``,
places a host-read batch on a two-device data mesh and takes its mean, an all-reduce across the
devices, then prints one line starting with :data:`REPORT_PREFIX`, then JSON: whether TensorFlow
is in the process, the devices, and the mean beside the host's. ``--import-tensorflow`` imports
TensorFlow first, hiding the GPUs from it as examples used to: the positive control, which on
GPUs reproduces the failure. ``--stream`` reads the batch with ``TFDSStreamingSource`` from the
fixture's TFRecord copy, a shuffled pull, instead of the eager source.

    python -m tests.test_common.tfds_collective <fixture directory> [--stream] [--import-tensorflow]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


DEVICES = 2
RECORDS = 8
# The report's line starts with this: NCCL writes to standard output too, through C stdio, whose
# buffer reaches the pipe after Python's at exit, so the report is not the last line there.
REPORT_PREFIX = "TFDS_COLLECTIVE "


def main(argv: list[str]) -> None:
    """Read, place, all-reduce and report; ``argv`` is the fixture directory and the flag."""
    root = Path(argv[0])
    if "--import-tensorflow" in argv[1:]:
        import tensorflow as tf  # noqa: PLC0415 - the positive control puts it in the process

        tf.config.set_visible_devices([], "GPU")

    import jax  # noqa: PLC0415 - after the control's TensorFlow, as a training script imports
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from substrax.spmd import (  # noqa: PLC0415
        create_data_parallel_sharding,
        place_batch_on_shards,
    )

    images = np.asarray(_read(root, stream="--stream" in argv[1:])["image"], np.float32)
    mesh = jax.make_mesh(
        (DEVICES,),
        ("data",),
        axis_types=(jax.sharding.AxisType.Auto,),
        devices=jax.devices()[:DEVICES],
    )
    with jax.set_mesh(mesh):
        placed = place_batch_on_shards({"image": images}, create_data_parallel_sharding(mesh))
        mean = float(jnp.mean(placed["image"]))
    report = {
        "tensorflow": "tensorflow" in sys.modules,
        "platform": jax.devices()[0].platform,
        "devices": len(placed["image"].sharding.device_set),
        "mean": mean,
        "host_mean": float(images.mean()),
    }
    sys.stdout.write(REPORT_PREFIX + json.dumps(report) + "\n")


def _read(root: Path, *, stream: bool) -> Any:
    """:data:`RECORDS` records of the fixture as a host ``Batch``, eager or streamed."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from datarax.core.index_words import to_words  # noqa: PLC0415
    from datarax.sources import (  # noqa: PLC0415
        TFDSEagerConfig,
        TFDSEagerSource,
        TFDSStreamingConfig,
        TFDSStreamingSource,
    )
    from tests.test_common.tfds_fixture import FIXTURE  # noqa: PLC0415

    if stream:
        source = TFDSStreamingSource(
            TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(root / "tfrecord"))
        )
        return source.get_batch(RECORDS, key=jax.random.key(0))
    eager = TFDSEagerSource(
        TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(root / "array_record"))
    )
    return eager.get_batch(to_words(np.arange(RECORDS, dtype=np.uint64)))


if __name__ == "__main__":
    main(sys.argv[1:])
