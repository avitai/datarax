"""P5: Checkpoint speed target tests.

Target: Datarax checkpoint cycle within 1.5x of raw Orbax on PR-1.
These tests MUST fail before optimization and pass after.

Note: The comparative benchmark compares OrbaxCheckpointHandler (with
datarax preprocessing) against raw ocp.StandardCheckpointer (baseline).
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from datarax.checkpoint.handlers import OrbaxCheckpointHandler
from tests.benchmarks.performance_targets import measure_latency


@pytest.fixture
def handler():
    return OrbaxCheckpointHandler()


@pytest.fixture
def async_handler():
    handler = OrbaxCheckpointHandler(async_checkpointing=True)
    yield handler
    handler.close()


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

    def test_save_restore_cycle(self, handler):
        """Verify basic save/restore cycle works correctly."""
        state = {
            "params": {"kernel": jnp.ones((64, 64)), "bias": jnp.zeros(64)},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            handler.save(tmpdir, state)
            restored = handler.restore(tmpdir)

            assert jnp.allclose(restored["params"]["kernel"], state["params"]["kernel"])
            assert jnp.allclose(restored["params"]["bias"], state["params"]["bias"])

    def test_save_latency_is_reasonable(self, handler, model_state):
        """Verify checkpoint save latency is under 5 seconds for a 4MB state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            counter = [0]

            def save_cycle():
                counter[0] += 1
                path = Path(tmpdir) / f"ckpt_{counter[0]}"
                handler.save(str(path), model_state)

            latency = measure_latency(save_cycle, repetitions=3)
            assert latency < 5.0, f"Save latency {latency:.2f}s exceeds 5s target"

    def test_checkpoint_cycle_within_1_5x_orbax(self, handler, model_state):
        """Compare datarax handler overhead against raw Orbax checkpointer.

        Measures the cost of datarax's preprocessing (_preprocess_prng_keys,
        _preprocess_strings) on top of raw orbax save/restore.
        This is the core P5 target-encoding test.
        """
        raw_checkpointer = ocp.StandardCheckpointer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Datarax handler cycle
            datarax_counter = [0]

            def datarax_cycle():
                datarax_counter[0] += 1
                ckpt_path = Path(tmpdir) / f"datarax_{datarax_counter[0]}"
                handler.save(str(ckpt_path), model_state)
                handler.restore(str(ckpt_path))

            datarax_latency = measure_latency(datarax_cycle, repetitions=3)

            # Raw Orbax cycle (baseline — no datarax preprocessing)
            orbax_counter = [0]

            def orbax_cycle():
                orbax_counter[0] += 1
                ckpt_path = Path(tmpdir) / f"orbax_{orbax_counter[0]}"
                raw_checkpointer.save(str(ckpt_path), model_state)
                raw_checkpointer.wait_until_finished()
                raw_checkpointer.restore(str(ckpt_path))

            orbax_latency = measure_latency(orbax_cycle, repetitions=3)

            # Assert: datarax overhead must be within 1.5x of raw orbax
            if orbax_latency > 0:
                ratio = datarax_latency / orbax_latency
                assert ratio <= 1.5, (
                    f"Datarax checkpoint ({datarax_latency * 1000:.1f}ms) is "
                    f"{ratio:.1f}x slower than raw Orbax ({orbax_latency * 1000:.1f}ms), "
                    f"exceeds 1.5x target"
                )


@pytest.mark.benchmark
class TestP5AsyncCheckpointing:
    """P5: Async checkpointing — save returns immediately."""

    def test_async_handler_uses_async_checkpointer(self, async_handler):
        """Verify async_checkpointing=True creates an AsyncCheckpointer."""
        assert isinstance(async_handler.checkpointer, ocp.AsyncCheckpointer)

    def test_async_save_restore_cycle(self, async_handler, model_state):
        """Verify async save + wait + restore produces correct state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async_handler.save(tmpdir, model_state)
            async_handler.wait_until_finished()
            restored = async_handler.restore(tmpdir)

            assert jnp.allclose(
                restored["params"]["dense"]["kernel"],
                model_state["params"]["dense"]["kernel"],
            )

    def test_sync_handler_uses_standard_checkpointer(self, handler):
        """Verify async_checkpointing=False uses StandardCheckpointer."""
        assert isinstance(handler.checkpointer, ocp.StandardCheckpointer)
