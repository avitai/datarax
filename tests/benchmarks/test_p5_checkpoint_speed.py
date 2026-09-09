"""P5: Checkpoint speed target tests.

Target: a datarax checkpoint cycle within 1.5x of raw Orbax on PR-1.

``IteratorCheckpoint`` wraps substrax's store, which is an Orbax
``CheckpointManager`` writing a ``PyTreeSave`` payload and a ``JsonSave`` metadata
sidecar. The comparative benchmark measures what datarax's layer adds on top of
that same manager doing the same composite save and restore: the state
extraction, the identity validation and ``set_state``.
"""

from typing import Any

import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from datarax.checkpoint import IteratorCheckpoint
from tests.benchmarks.performance_targets import measure_latency


class _DictState:
    """A Checkpointable over a fixed state dictionary."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def get_state(self) -> dict[str, Any]:
        return self.state

    def set_state(self, state: dict[str, Any]) -> None:
        self.state = state


@pytest.fixture
def model_state():
    """Representative model state (~4MB) used across P5 tests."""
    return {
        "params": {
            "dense": {"kernel": jnp.ones((1024, 1024)), "bias": jnp.zeros(1024)},
        },
        "batch_norm": {"mean": jnp.zeros(1024), "var": jnp.ones(1024)},
    }


@pytest.mark.benchmark
class TestP5CheckpointSpeed:
    """P5: Checkpoint save+restore within 1.5x of raw Orbax."""

    def test_save_restore_cycle(self, tmp_path):
        """Verify basic save/restore cycle works correctly."""
        state = {"params": {"kernel": jnp.ones((64, 64)), "bias": jnp.zeros(64)}}
        target = _DictState({"params": {"kernel": jnp.zeros((64, 64)), "bias": jnp.ones(64)}})

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(_DictState(state), step=0)
            checkpoint.restore(target, step=0)

        assert jnp.array_equal(target.state["params"]["kernel"], state["params"]["kernel"])
        assert jnp.array_equal(target.state["params"]["bias"], state["params"]["bias"])

    def test_save_latency_is_reasonable(self, tmp_path, model_state):
        """Verify checkpoint save latency is under 5 seconds for a 4MB state."""
        with IteratorCheckpoint(tmp_path, max_to_keep=10) as checkpoint:
            counter = [0]

            def save_cycle():
                counter[0] += 1
                checkpoint.save(_DictState(model_state), step=counter[0])

            latency = measure_latency(save_cycle, repetitions=3)
        assert latency < 5.0, f"Save latency {latency:.2f}s exceeds 5s target"

    def test_checkpoint_cycle_within_1_5x_orbax(self, tmp_path, model_state):
        """Compare the datarax checkpoint cycle against the Orbax manager it wraps.

        The two cycles alternate, so disk-cache and thread-pool drift over the
        run lands on both sides alike; each side's best case is compared. This
        is the core P5 target-encoding test.
        """
        manager = ocp.CheckpointManager(
            tmp_path / "orbax", options=ocp.CheckpointManagerOptions(max_to_keep=20)
        )
        counter = [0]
        target = _DictState(model_state)

        with IteratorCheckpoint(tmp_path / "datarax", max_to_keep=20) as checkpoint:

            def datarax_cycle():
                checkpoint.save(_DictState(model_state), step=counter[0])
                checkpoint.restore(target, step=counter[0])

            def orbax_cycle():
                manager.save(
                    counter[0],
                    args=ocp.args.Composite(
                        model=ocp.args.PyTreeSave(model_state),
                        metadata=ocp.args.JsonSave({"step": counter[0]}),
                    ),
                )
                manager.wait_until_finished()
                manager.restore(
                    counter[0],
                    args=ocp.args.Composite(
                        model=ocp.args.PyTreeRestore(), metadata=ocp.args.JsonRestore
                    ),
                )

            datarax_latencies: list[float] = []
            orbax_latencies: list[float] = []
            for _ in range(9):
                counter[0] += 1
                datarax_latencies.append(measure_latency(datarax_cycle, repetitions=1))
                orbax_latencies.append(measure_latency(orbax_cycle, repetitions=1))
        manager.close()

        # The first two rounds absorb first-call filesystem and Orbax init costs. Each
        # remaining round yields one paired ratio, so a slow patch of disk hits both
        # sides of that pair; the median pair is the guard.
        ratios = sorted(
            datarax / orbax
            for datarax, orbax in zip(datarax_latencies[2:], orbax_latencies[2:], strict=True)
            if orbax > 0
        )
        ratio = ratios[len(ratios) // 2]
        assert ratio <= 1.5, (
            f"Datarax checkpoint cycle is {ratio:.2f}x the Orbax manager's over "
            f"{len(ratios)} paired rounds (datarax min {min(datarax_latencies) * 1000:.1f}ms, "
            f"orbax min {min(orbax_latencies) * 1000:.1f}ms), exceeds 1.5x target"
        )
