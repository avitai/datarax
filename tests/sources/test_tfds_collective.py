"""R53: a two-device all-reduce over a TFDS read, with no TensorFlow in the process.

The reading process runs :mod:`tests.test_common.tfds_collective` in a fresh interpreter: it
reads the offline fixture with ``TFDSEagerSource``, places a host-read batch on a two-device data
mesh and takes its mean across the devices. On CPU the two devices are emulated, which checks the
program; on two GPUs the mean is an NCCL all-reduce, which TensorFlow in the process breaks. The
GPU case runs where two GPUs are visible (the ``tfds-nccl`` compute job in ``pyproject.toml``,
which also runs the program with ``--import-tensorflow`` as its positive control).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from substrax.testing import run_python

from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common.tfds_fixture import TFDSFixture


pytestmark = [pytest.mark.tfds, pytest.mark.devices(2)]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROGRAM = _REPO_ROOT / "tests" / "test_common" / "tfds_collective.py"
_SECONDS = 300.0


def _collective(fixture: TFDSFixture, env: dict[str, str]) -> dict[str, object]:
    result = run_python(_PROGRAM, str(fixture.root), timeout=_SECONDS, env=env, cwd=_REPO_ROOT)
    return json.loads(result.check().stdout.strip().splitlines()[-1])


def _assert_an_all_reduce_without_tensorflow(report: dict[str, object]) -> None:
    assert report["tensorflow"] is False
    assert report["devices"] == 2
    assert report["mean"] == pytest.approx(report["host_mean"], rel=1e-6)


def test_an_all_reduce_over_two_cpu_devices_after_a_tfds_read(tfds_fixture: TFDSFixture) -> None:
    env = {**forwarded_jax_environment(os.environ), "JAX_PLATFORMS": "cpu"}
    env.setdefault("JAX_NUM_CPU_DEVICES", "2")

    report = _collective(tfds_fixture, env)

    assert report["platform"] == "cpu"
    _assert_an_all_reduce_without_tensorflow(report)


@pytest.mark.accelerator(kind="gpu")
def test_an_nccl_all_reduce_over_two_gpus_after_a_tfds_read(tfds_fixture: TFDSFixture) -> None:
    report = _collective(
        tfds_fixture, {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    )

    assert report["platform"] == "gpu"
    _assert_an_all_reduce_without_tensorflow(report)
