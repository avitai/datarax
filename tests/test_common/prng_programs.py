"""What every path from a pipeline's key to its records names under one set of global settings.

:func:`names_under` sets a ``jax_default_prng_impl`` and a ``jax_threefry_partitionable``, in code
(``jax.config.update``) or by context manager, and digests the record names each path gives for a
pipeline over a threefry2x32 key: the host stage's served names on threads and in worker
processes, over an indexed source and, given the TFDS fixture, a TFDS stream; the compiled step's
naming (``Pipeline.step``); a stream's pass seed (``fold_on_host``); and the host index shuffle
(``index_shuffle``). Run as a script it prints the same for a fresh process.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from substrax.testing import restored_jax_config

from datarax.core.data_source import DataSourceModule
from datarax.core.index_shuffle import index_shuffle
from datarax.core.prng import fold_on_host, key_words
from datarax.pipeline import Pipeline
from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource
from datarax.sources.memory_source import MemorySourceConfig
from tests.test_common.tfds_fixture import FIXTURE
from tests.test_common.worker_reads import ProcessReadMemorySource, resources


IMPLS = ("rbg", "unsafe_rbg", "threefry2x32")
"""Two impls whose keys of one seed hold equal words, then the default. The others come first,
so a program or value a warm process caches is first made under an impl other than the
default: a key wrapped without its pinned impl would then show."""
PARTITIONABLE = (True, False)
HOWS = ("config.update", "context manager")
_RECORDS = 1000
_BATCH = 16


@contextlib.contextmanager
def under(impl: str, partitionable: bool, how: str) -> Iterator[None]:
    """The global settings ``impl`` and ``partitionable``, set in code or by context manager."""
    if how == "config.update":
        with restored_jax_config():
            jax.config.update("jax_default_prng_impl", impl)
            jax.config.update("jax_threefry_partitionable", partitionable)
            yield
        return
    with jax.default_prng_impl(impl), jax.threefry_partitionable(partitionable):
        yield


def _digest(values: list[bytes]) -> str:
    return hashlib.sha256(b"".join(values)).hexdigest()[:16]


def _served(source: DataSourceModule, workers: int, batch_size: int) -> str:
    """The digest of the record indices and epochs a shuffled run of ``source`` serves."""
    pipe = Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(jax.random.key(0, impl="threefry2x32")),
        shuffle=True,
        num_epochs=1,
        host_resources=resources(workers) if workers else None,
    )
    names = []
    for batch in pipe.raw_batches():
        host = jax.device_get(batch)
        names += [np.asarray(host.indices).tobytes(), np.asarray(host.epochs).tobytes()]
    pipe.close()
    return _digest(names)


def _indexed() -> ProcessReadMemorySource:
    return ProcessReadMemorySource(MemorySourceConfig(), {"x": np.arange(_RECORDS, dtype=np.int32)})


def _stream(tfrecord: Path) -> TFDSStreamingSource:
    return TFDSStreamingSource(
        TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tfrecord))
    )


def names_under(
    impl: str, partitionable: bool, how: str, *, workers: tuple[int, ...], tfrecord: Path | None
) -> dict[str, str]:
    """A digest of what each path names under these global settings.

    Args:
        impl: ``jax_default_prng_impl``.
        partitionable: ``jax_threefry_partitionable``.
        how: ``"config.update"`` or ``"context manager"``.
        workers: The worker counts the host stage serves with (0 reads on threads).
        tfrecord: The TFDS fixture's TFRecord directory, or ``None`` to leave the stream out.

    Returns:
        A digest per path.
    """
    with under(impl, partitionable, how):
        found = {}
        for count in workers:
            found[f"indexed workers={count}"] = _served(_indexed(), count, _BATCH)
            if tfrecord is not None:
                found[f"tfds stream workers={count}"] = _served(_stream(tfrecord), count, 3)
        stepped = Pipeline(
            source=_indexed(),
            stages=[],
            batch_size=_BATCH,
            rngs=nnx.Rngs(jax.random.key(0, impl="threefry2x32")),
            shuffle=True,
        )
        found["step"] = _digest(
            [np.asarray(jax.device_get(stepped.step().indices)).tobytes() for _ in range(3)]
        )
        words = key_words(jax.random.key_data(jax.random.key(7, impl="threefry2x32")))
        found["fold_on_host"] = _digest([fold_on_host(words, 3).tobytes()])
        found["index_shuffle"] = _digest(
            [np.asarray([index_shuffle(i, 11, _RECORDS, 2) for i in range(8)]).tobytes()]
        )
        return found


if __name__ == "__main__":
    fixture = None if sys.argv[2] == "-" else Path(sys.argv[2])
    names = names_under(
        "threefry2x32", sys.argv[1] == "True", "context manager", workers=(0,), tfrecord=fixture
    )
    sys.stdout.write(json.dumps(names) + "\n")
