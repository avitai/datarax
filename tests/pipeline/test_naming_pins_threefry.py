"""Record names never depend on the global ``jax_default_prng_impl``.

datarax wraps every key that orders or names records as ``threefry2x32``
(:data:`~datarax.core.prng.NAMING_PRNG_IMPL`). Under any global impl, set in code or by context
manager, in one warm process with nothing cleared between settings, every path from a pipeline's
key to its records names what a fresh process names under the default impl: the host stage on
threads and in worker processes, over an indexed source and a TFDS stream, the compiled step, a
stream's pass seed and the host index shuffle. ``jax_threefry_partitionable`` does change the
order (the Feistel order's round keys are threefry bits), so readers carry it and the names
follow it. A block of batches named in one vmapped call equals each batch named alone.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import nnx
from substrax.testing import run_python

from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.pipeline.epochs import EpochPlan, HostNaming
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.prng_programs import HOWS, IMPLS, names_under, PARTITIONABLE, under
from tests.test_common.tfds_fixture import TFDSFixture
from tests.test_common.worker_reads import ProcessReadMemorySource, resources


_ROOT = Path(__file__).resolve().parents[2]
_HELPER = _ROOT / "tests" / "test_common" / "prng_programs.py"


def _fresh_default(partitionable: bool, tfrecord: Path | None) -> dict[str, str]:
    """What a fresh process names under the default impl."""
    result = run_python(
        _HELPER,
        str(partitionable),
        "-" if tfrecord is None else str(tfrecord),
        timeout=600,
        cwd=_ROOT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _check_against_fresh(tfrecord: Path | None) -> None:
    fresh = {
        partitionable: _fresh_default(partitionable, tfrecord) for partitionable in PARTITIONABLE
    }
    # The control: the partitionable flag changes the indexed order and the host shuffle, so a
    # reader or a cache that dropped it would show.
    for path in ("indexed workers=0", "index_shuffle"):
        assert fresh[True][path] != fresh[False][path], path
    differing = {}
    for impl, partitionable, how in itertools.product(IMPLS, PARTITIONABLE, HOWS):
        warm = names_under(impl, partitionable, how, workers=(0, 2), tfrecord=tfrecord)
        expected = fresh[partitionable]
        for path, digest in warm.items():
            reference = expected[path.replace("workers=2", "workers=0")]
            if digest != reference:
                differing[(impl, partitionable, how, path)] = (digest, reference)
    assert not differing, differing


def test_a_warm_process_names_as_a_fresh_default_one_under_every_global_setting() -> None:
    _check_against_fresh(None)


@pytest.mark.tfds
def test_a_tfds_stream_names_as_a_fresh_default_one_under_every_global_setting(
    tfds_fixture: TFDSFixture,
) -> None:
    _check_against_fresh(tfds_fixture.tfrecord)


@pytest.mark.parametrize(
    ("impl", "partitionable"),
    list(itertools.product(IMPLS, PARTITIONABLE)),
    ids=lambda value: str(value),
)
def test_a_block_names_each_batch_as_alone(impl: str, partitionable: bool) -> None:
    """``HostNaming.block`` (vmapped) equals per-batch naming under every global setting.

    Each setting names over a plan of its own, so its programs are traced under it: a program a
    test traced earlier under another setting would otherwise be served again.
    """
    length = 997 - 7 * IMPLS.index(impl) - int(partitionable)
    plan = EpochPlan(length=length, batch_size=16, drop_last=False, num_epochs=1)
    source = MemorySource(MemorySourceConfig(), {"x": np.arange(length, dtype=np.int32)})
    words = np.asarray(jax.random.key_data(jax.random.key(0, impl="threefry2x32")), np.uint32)
    starts = list(range(0, 16 * 8, 16))
    with under(impl, partitionable, "context manager"):
        naming = HostNaming(source, plan, shuffled=True)
        alone = [np.asarray(naming(start, 0, 16, words)[0]) for start in starts]
        block, _ = naming.block(
            np.stack([to_words(start) for start in starts]).astype(np.uint32),
            np.zeros(len(starts), np.int32),
            words,
        )
    for slot, names in enumerate(alone):
        np.testing.assert_array_equal(np.asarray(block[slot]), names)


@pytest.mark.parametrize("how", HOWS)
def test_a_pipeline_over_a_key_of_another_impl_names_with_a_threefry_key(how: str) -> None:
    """``nnx.Rngs(0)`` under a global rbg draws rbg keys; the order key is threefry data drawn
    from it, the same on threads and in worker processes."""
    with under("rbg", True, how):
        pipes = [
            Pipeline(
                source=kind(MemorySourceConfig(), {"x": np.arange(64, dtype=np.int32)}),
                stages=[],
                batch_size=8,
                rngs=nnx.Rngs(0),
                shuffle=True,
                num_epochs=1,
                host_resources=budget,
            )
            for kind, budget in ((MemorySource, None), (ProcessReadMemorySource, resources(2)))
        ]
        names = [
            [np.asarray(jax.device_get(b.indices)).tobytes() for b in pipe.raw_batches()]
            for pipe in pipes
        ]
    assert pipes[0]._epoch_key_base.get_value().shape == (2,)  # noqa: SLF001 - the order key
    assert pipes[1].host_plan.workers == 2
    assert names[0] == names[1]
