#!/usr/bin/env python3
"""Download and prepare the example datasets that CI caches for the long-running example tier.

CIFAR-10 is served from https://www.cs.toronto.edu, which GitHub runners fetch at 13-16 s per MiB,
so an example that loads it cannot download it within its time budget. This script fills TFDS's
data directory and keras's dataset cache once; the long-running job restores them.

Usage:
    uv run python scripts/prepare_example_datasets.py
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from substrax.runtime import configure_entry_point_logging


LOGGER = logging.getLogger(__name__)

# Datasets the examples load from the slow host, by loader.
TFDS_DATASETS = ("cifar10",)
KERAS_DATASETS = ("cifar10",)


def prepare_tfds_dataset(name: str) -> None:
    """Download and prepare a TFDS dataset into TFDS's data directory.

    Args:
        name: The TFDS builder name.
    """
    import tensorflow_datasets as tfds

    tfds.builder(name).download_and_prepare()


def prepare_keras_dataset(name: str) -> None:
    """Download a keras dataset into keras's dataset cache.

    Args:
        name: The module name under ``keras.datasets``.
    """
    from keras import datasets

    getattr(datasets, name).load_data()


def _prepare_timed(loader: str, prepare: Callable[[str], None], name: str) -> None:
    """Run one preparation and log how long it took."""
    started = time.perf_counter()
    prepare(name)
    LOGGER.info("Prepared %s dataset %s in %.0f s", loader, name, time.perf_counter() - started)


def prepare_all() -> None:
    """Prepare every dataset, downloading them concurrently."""
    work = [("tfds", prepare_tfds_dataset, name) for name in TFDS_DATASETS]
    work += [("keras", prepare_keras_dataset, name) for name in KERAS_DATASETS]
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        for future in [pool.submit(_prepare_timed, *item) for item in work]:
            future.result()


def main() -> None:
    """Prepare the example datasets."""
    configure_entry_point_logging()
    prepare_all()


if __name__ == "__main__":
    main()
