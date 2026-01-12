"""Complete integration tests for DAG implementation."""

import jax.numpy as jnp

from datarax import DAGExecutor
from datarax.dag import DAGConfig


class TestDAGIntegration:
    """Complete integration tests."""

    def test_mnist_pipeline(self):
        """Test realistic MNIST pipeline with transforms."""
        from datarax.sources.memory_source import MemorySource, MemorySourceConfig
        from datarax.operators import ElementOperator, ElementOperatorConfig

        # Create data
        data = [{"image": jnp.ones((28, 28, 1)), "label": i} for i in range(100)]

        # Simple normalize function as Element operator
        def normalize(element, key):
            new_data = dict(element.data)
            if "image" in new_data:
                new_data["image"] = (new_data["image"] - 0.5) / 0.5
            return element.replace(data=new_data)

        config = MemorySourceConfig()
        op_config = ElementOperatorConfig(stochastic=False)
        normalize_op = ElementOperator(op_config, fn=normalize)

        # Build pipeline
        executor = (
            DAGExecutor()
            .add(MemorySource(config, data))
            .batch(32)
            .operate(normalize_op)
            .shuffle(100)
        )

        # Process batches
        batch_count = 0
        for batch in executor:
            assert batch["image"].shape[0] <= 32
            assert batch["label"].shape[0] <= 32
            batch_count += 1

        assert batch_count == 4  # 100 samples / 32 batch size

    def test_config_loading(self):
        """Test loading from configuration."""
        from datarax.sources.memory_source import MemorySource, MemorySourceConfig

        # Config loading with the new config-first API requires a config object
        # For the simplest approach, we test the DAGConfig structure validation
        config = {
            "pipeline": {"name": "test_pipeline", "enforce_batch": True},
            "nodes": [
                {"id": "batch", "type": "BatchNode", "params": {"batch_size": 2}},
            ],
        }

        executor = DAGConfig.from_dict(config)

        assert executor.name == "test_pipeline"
        assert executor.enforce_batch is True

        # Separately test MemorySource with proper config-first API
        data = [1, 2, 3, 4, 5]
        source = MemorySource(MemorySourceConfig(), data)
        assert len(list(source)) == 5

    def test_checkpointing(self):
        """Test state saving and restoration."""
        from datarax.sources.memory_source import MemorySource, MemorySourceConfig

        data = list(range(10))
        config = MemorySourceConfig()

        # Create pipeline
        executor = DAGExecutor().add(MemorySource(config, data)).batch(3)

        # Process some batches
        iterator = iter(executor)
        _ = next(iterator)
        _ = next(iterator)

        # Save state
        state = executor.get_state()

        # Create new executor and restore
        new_executor = DAGExecutor().add(MemorySource(config, data)).batch(3)

        # Expect possible warning about state mismatch (this is ok for checkpointing)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            new_executor.set_state(state)

        # Continue from saved state - _iteration_count is now a plain int (not Variable)
        assert new_executor._iteration_count == executor._iteration_count

    def test_performance(self):
        """Test pipeline performance."""
        import time
        from datarax.sources.memory_source import MemorySource, MemorySourceConfig

        # Large dataset
        data = [{"value": i} for i in range(10000)]
        config = MemorySourceConfig()

        executor = (
            DAGExecutor(jit_compile=True).add(MemorySource(config, data)).batch(100).cache(50)
        )

        start = time.time()

        batch_count = 0
        for batch in executor:
            batch_count += 1

        elapsed = time.time() - start

        assert batch_count == 100
        assert elapsed < 10.0  # Should be reasonably fast (allowing for JIT compilation)

        print(f"Processed {len(data)} items in {elapsed:.2f}s")
        print(f"Throughput: {len(data) / elapsed:.0f} items/s")


if __name__ == "__main__":
    test = TestDAGIntegration()
    test.test_mnist_pipeline()
    test.test_config_loading()
    test.test_checkpointing()
    test.test_performance()
    print("✓ All integration tests passed!")
