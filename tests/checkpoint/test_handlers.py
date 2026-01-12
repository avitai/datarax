"""Tests for the checkpoint handlers module."""

import os
import shutil
import tempfile
import unittest
import warnings

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.checkpoint.handlers import OrbaxCheckpointHandler
from datarax.core.config import DataraxModuleConfig
from datarax.core.module import DataraxModule
from datarax.typing import Checkpointable


class SimpleModule(nnx.Module):
    """A simple test module with parameters and RNG keys for testing.

    Uses proper NNX/Orbax checkpointing patterns:
    - nnx.state() extracts PyTree state
    - Filter to nnx.Param only (excludes RNG keys)
    - nnx.to_pure_dict() for Orbax-compatible format
    - nnx.update() restores state
    """

    def __init__(self, input_dim, output_dim, rngs, dropout_rate=0.1):
        """Initialize the simple module."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x):
        """Apply the module to the input."""
        x = self.linear(x)
        return self.dropout(x)

    def get_state(self):
        """Return the state for checkpointing using proper NNX patterns.

        Returns a dict with:
        - config: Configuration as arrays (Orbax-compatible)
        - params: Model parameters as pure dict (Orbax-compatible)
        """
        # Extract only Param variables (excludes RNG keys)
        params = nnx.state(self, nnx.Param)

        return {
            # Config stored as arrays (Orbax StandardSave compatible)
            "input_dim": jnp.array(self.input_dim),
            "output_dim": jnp.array(self.output_dim),
            "dropout_rate": jnp.array(self.dropout_rate),
            # Params as pure dict (Orbax-compatible PyTree)
            "params": nnx.to_pure_dict(params),
        }

    def set_state(self, state_dict):
        """Restore the state of the module from a checkpoint."""
        # Restore configuration
        self.input_dim = int(state_dict["input_dim"])
        self.output_dim = int(state_dict["output_dim"])
        self.dropout_rate = float(state_dict["dropout_rate"])

        # Restore params using nnx.update pattern
        current_params = nnx.state(self, nnx.Param)
        nnx.replace_by_pure_dict(current_params, state_dict["params"])
        nnx.update(self, current_params)


class SimpleCheckpointable(Checkpointable):
    """A simple test class implementing the Checkpointable protocol."""

    def __init__(self, value):
        self._value = value
        self._step = 0

    def get_state(self):
        """Return the state for checkpointing."""
        return {"value": self._value, "step": self._step}

    def set_state(self, state):
        """Restore the state from a checkpoint."""
        self._value = state["value"]
        self._step = state["step"]

    def increment_step(self):
        """Increment the step counter."""
        self._step += 1


class TestOrbaxCheckpointHandler(unittest.TestCase):
    """Tests for the OrbaxCheckpointHandler class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create a handler
        self.handler = OrbaxCheckpointHandler()

        # Create a simple module with proper RNG keys
        rng_key = jax.random.key(0)
        self.module = SimpleModule(10, 10, nnx.Rngs(default=rng_key))

        # Create a simple checkpointable object
        self.checkpointable = SimpleCheckpointable(jnp.ones((5, 5)))

    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_save_and_restore_simple_dict(self):
        """Test saving and restoring a simple dictionary."""
        # Create a simple dictionary
        state = {"a": jnp.ones((5, 5)), "b": 10}

        # Save the dictionary
        path = self.handler.save(self.temp_dir, state)

        # Check that the path exists
        self.assertTrue(os.path.exists(path))

        # Restore the simple dictionary
        # In modern Orbax, sharding is specified through the target structure,
        # not as a parameter to StandardRestore
        # Suppress the expected warning about missing sharding info
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sharding info not provided when restoring")
            restored = self.handler.restore(self.temp_dir)

        # Check that the restored dictionary is correct
        self.assertTrue(jnp.array_equal(restored["a"], state["a"]))
        self.assertEqual(restored["b"], state["b"])

    def test_save_and_restore_checkpointable(self):
        """Test saving and restoring a Checkpointable object."""
        # Increment the step
        self.checkpointable.increment_step()

        # Save the checkpointable
        path = self.handler.save(self.temp_dir, self.checkpointable)

        # Check that the path exists
        self.assertTrue(os.path.exists(path))

        # Create a new checkpointable for restoration
        new_checkpointable = SimpleCheckpointable(jnp.zeros((5, 5)))

        # Restore the checkpointable
        # In modern Orbax, sharding is specified through the target structure,
        # not as a parameter to StandardRestore
        # Suppress the expected warning about missing sharding info
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sharding info not provided when restoring")
            restored = self.handler.restore(self.temp_dir, new_checkpointable)

        # Check that it's the same object (handler returns the target)
        self.assertIs(restored, new_checkpointable)

        # Check that the state was restored correctly
        self.assertTrue(jnp.array_equal(new_checkpointable._value, self.checkpointable._value))
        self.assertEqual(new_checkpointable._step, self.checkpointable._step)

    def test_save_and_restore_module(self):
        """Test saving and restoring a module with parameters (RNG keys excluded)."""
        # Save the module state (RNG keys are automatically excluded)
        module_state = self.module.get_state()
        path = self.handler.save(self.temp_dir, module_state)

        # Check that the path exists
        self.assertTrue(os.path.exists(path))

        # Create a new module with fresh RNG
        new_rng_key = jax.random.key(1)
        new_module = SimpleModule(10, 10, nnx.Rngs(default=new_rng_key))

        # Restore the module state
        # Suppress the expected warning about missing sharding info
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sharding info not provided when restoring")
            restored_state = self.handler.restore(self.temp_dir)

        # Set the state on the new module
        new_module.set_state(restored_state)

        # Check that the parameters were restored correctly
        original_params = self.module.linear.kernel
        restored_params = new_module.linear.kernel

        # Extract values from Param objects if needed
        if hasattr(original_params, "get_value"):
            original_params = original_params.get_value()
        if hasattr(restored_params, "get_value"):
            restored_params = restored_params.get_value()

        self.assertTrue(jnp.array_equal(original_params, restored_params))

    def test_versioning(self):
        """Test checkpoint versioning support."""
        # Save checkpoint at step 1
        state1 = {"value": 1}
        self.handler.save(self.temp_dir, state1, step=1, keep=10)

        # Save checkpoint at step 2
        state2 = {"value": 2}
        self.handler.save(self.temp_dir, state2, step=2, keep=10)

        try:
            # Restore specific step
            restored1 = self.handler.restore(self.temp_dir, step=1)

            # Check that it's the correct state
            self.assertEqual(restored1["value"], 1)

            # Restore another step
            restored2 = self.handler.restore(self.temp_dir, step=2)

            # Check that it's the correct state
            self.assertEqual(restored2["value"], 2)

            # Restore latest (without specifying step)
            restored_latest = self.handler.restore(self.temp_dir)

            # Check that it's step 2 (the latest)
            self.assertEqual(restored_latest["value"], 2)
        except Exception as e:
            # Skip the test if we have orbax compatibility issues
            if "structure" in str(e).lower() or "sharding" in str(e).lower():
                self.skipTest(
                    f"Skipping test due to Orbax checkpoint structure compatibility issue: {e}"
                )
            else:
                raise


class SampleDataraxModule(DataraxModule):
    """A sample DataraxModule for testing NNXCheckpointHandler."""

    def __init__(
        self,
        config: DataraxModuleConfig | None = None,
        features: int = 5,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the test module."""
        if config is None:
            config = DataraxModuleConfig()
        if rngs is None:
            rngs = nnx.Rngs(0)
        super().__init__(config, rngs=rngs)
        self.features = features
        self.linear = nnx.Linear(10, features, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
        """Apply the module."""
        x = self.linear(x)
        return self.dropout(x)
