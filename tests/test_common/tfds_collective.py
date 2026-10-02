"""A two-device all-reduce over a batch the eager TFDS source read: the program a test runs.

TensorFlow in a JAX process breaks JAX's NCCL collectives on GPUs ("corrupted comm object
detected"), so the process that reads TFDS data for training must not hold it. This program
reads the offline fixture (:mod:`tests.test_common.tfds_fixture`) with ``TFDSEagerSource``,
places a host-read batch on a two-device data mesh and takes its mean, an all-reduce across the
devices, then prints one JSON line: whether TensorFlow is in the process, the devices, and the
mean beside the host's. ``--import-tensorflow`` imports TensorFlow first, hiding the GPUs from it
as examples used to: the positive control, which on GPUs reproduces the failure.

    python tests/test_common/tfds_collective.py <fixture directory> [--import-tensorflow]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


DEVICES = 2
RECORDS = 8


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

    from datarax.core.index_words import to_words  # noqa: PLC0415
    from datarax.sources import TFDSEagerConfig, TFDSEagerSource  # noqa: PLC0415
    from tests.test_common.tfds_fixture import FIXTURE  # noqa: PLC0415

    source = TFDSEagerSource(
        TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(root / "array_record"))
    )
    batch = source.get_batch(to_words(np.arange(RECORDS, dtype=np.uint64)))
    images = np.asarray(batch["image"], np.float32)
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
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
