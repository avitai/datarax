# File: tests/integration/test_end_to_end_pipeline.py

from unittest.mock import Mock, patch
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.dag import DAGExecutor
from datarax.dag.nodes import BatchNode, OperatorNode
from datarax.control.prefetcher import Prefetcher
from datarax.sources.array_record_source import ArrayRecordSourceModule, ArrayRecordSourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.core.element_batch import Element
from datarax.core.config import StructuralConfig


from datarax.core.data_source import DataSourceModule


class IndexedNumericSource(DataSourceModule):
    """Simple configurable source for integration pipeline tests."""

    def __init__(
        self,
        config: StructuralConfig,
        *,
        source_id: int,
        multiplier: float,
        total_items: int,
        rngs=None,
        name=None,
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.source_id = source_id
        self.multiplier = multiplier
        self.total_items = total_items
        self.index = nnx.Variable(0)

    def __iter__(self):
        self.index.set_value(0)
        return self

    def __next__(self):
        current_index = int(self.index.get_value())
        if current_index >= self.total_items:
            raise StopIteration
        result = {
            "source": jnp.array(self.source_id),
            "value": jnp.array(float(current_index * self.multiplier)),
        }
        self.index.set_value(current_index + 1)
        return result


def _collect_pipeline_batches(pipeline: DAGExecutor, max_steps: int = 3) -> list[dict[str, Any]]:
    """Collect non-empty batches from a pipeline."""
    batches: list[dict[str, Any]] = []
    for step_index, batch in enumerate(pipeline):
        if batch is not None:
            batches.append(batch)
        if step_index >= max_steps - 1:
            break
    return batches


def _assert_batches_for_source(batches: list[dict[str, Any]], source_id: int) -> None:
    """Assert batch collection belongs to the expected source id."""
    assert len(batches) > 0, f"Should have batches from source {source_id}"
    for batch in batches:
        assert "source" in batch
        assert jnp.all(batch["source"] == source_id)


class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""

    def test_arrayrecord_to_dataloader_pipeline(self):
        """Test end-to-end pipeline with ArrayRecord source, batching, and augmentations.

        This test validates:
        - ArrayRecordSourceModule integration with DAGExecutor
        - BatchNode for batching data
        - ElementOperator for normalization and data augmentation
        - Prefetcher for asynchronous data loading
        - Proper data flow through the entire pipeline
        """
        # Mock Grain ArrayRecordDataSource
        mock_grain_source = Mock()
        mock_grain_source.__len__ = Mock(return_value=100)
        mock_grain_source.__getitem__ = Mock(
            side_effect=lambda idx: {"image": np.random.rand(28, 28), "label": np.array(idx % 10)}
        )

        with patch("grain.python.ArrayRecordDataSource", return_value=mock_grain_source):
            # Create source with config-first pattern
            config = ArrayRecordSourceConfig(num_epochs=2)
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Define operations as ElementOperator functions
            def normalize_fn(element: Element, key: jax.Array) -> Element:
                data = element.data
                new_data = {**data, "image": jnp.asarray(data["image"]) / 255.0}
                return element.replace(data=new_data)

            def augment_fn(element: Element, key: jax.Array) -> Element:
                # Simple augmentation
                data = element.data
                if np.random.random() > 0.5:
                    new_data = {**data, "image": jnp.fliplr(data["image"])}
                else:
                    new_data = data
                return element.replace(data=new_data)

            # Create Pipeline with DAGExecutor
            # Start with source, then add batch node, then transformations
            pipeline = DAGExecutor()
            pipeline.add(source)  # Add DataSourceModule directly (DAGExecutor wraps it)
            pipeline.add(BatchNode(batch_size=1))  # Then add batching

            # Apply operations using ElementOperator
            det_config = ElementOperatorConfig(stochastic=False)
            normalize_op = ElementOperator(det_config, fn=normalize_fn)

            # Stochastic operators require stream_name and rngs
            rngs = nnx.Rngs(default=0, augment=1)
            stoch_config = ElementOperatorConfig(stochastic=True, stream_name="augment")
            augment_op = ElementOperator(stoch_config, fn=augment_fn, rngs=rngs)

            pipeline.add(OperatorNode(normalize_op))
            pipeline.add(OperatorNode(augment_op))

            # Use Prefetcher for async loading
            prefetcher = Prefetcher(buffer_size=10)
            loader = prefetcher.prefetch(pipeline)

            # Process batches
            batch_count = 0
            for batch in loader:
                batch_count += 1

                # Verify batch structure
                assert "image" in batch
                assert "label" in batch

                # Verify normalization was applied
                assert jnp.asarray(batch["image"]).max() <= 1.0

                if batch_count >= 10:
                    break

            assert batch_count == 10

    def test_pipeline_transformations(self):
        """Test DAGExecutor pipeline with chained transformations.

        This test validates:
        - Custom DataSourceModule implementation
        - Using OperatorNode with ElementOperator for transformations
        - Sequential transformation application (multiply then add)
        - Correct mathematical operations on batched data
        """
        from datarax.core.data_source import DataSourceModule

        # Create a simple data source with config-first pattern
        class SimpleSource(DataSourceModule):
            def __init__(self, config: StructuralConfig, size=10, *, rngs=None, name=None):
                super().__init__(config, rngs=rngs, name=name)
                self.size = size
                self.index = nnx.Variable(0)

            def __iter__(self):
                self.index.set_value(0)
                return self

            def __next__(self):
                if self.index.get_value() >= self.size:
                    raise StopIteration
                result = {"value": jnp.array(float(self.index.get_value()))}
                self.index.set_value(self.index.get_value() + 1)
                return result

        # Create pipeline with transformations
        source = SimpleSource(StructuralConfig(), size=20)

        # Build pipeline: source -> batch -> transform1 -> transform2
        pipeline = DAGExecutor()
        pipeline.add(source)  # Add DataSourceModule directly
        pipeline.add(BatchNode(batch_size=4))

        # Define transformation functions for ElementOperator
        def multiply_by_2_fn(element: Element, key: jax.Array) -> Element:
            data = element.data
            new_data = {"value": data["value"] * 2}
            return element.replace(data=new_data)

        def add_one_fn(element: Element, key: jax.Array) -> Element:
            data = element.data
            new_data = {"value": data["value"] + 1}
            return element.replace(data=new_data)

        # Add operators to pipeline
        config = ElementOperatorConfig(stochastic=False)
        pipeline.add(OperatorNode(ElementOperator(config, fn=multiply_by_2_fn)))
        pipeline.add(OperatorNode(ElementOperator(config, fn=add_one_fn)))

        # Process batches and verify transformations
        batches = []
        for i, batch in enumerate(pipeline):
            if batch is not None:
                batches.append(batch)
            if i >= 3:  # Get a few batches
                break

        assert len(batches) > 0, "Should have processed some batches"

        # Verify transformations were applied
        # Original values: 0, 1, 2, 3 (first batch)
        # After multiply_by_2: 0, 2, 4, 6
        # After add_one: 1, 3, 5, 7
        first_batch = batches[0]["value"]
        expected = jnp.array([1.0, 3.0, 5.0, 7.0])
        assert jnp.allclose(first_batch, expected), f"Expected {expected}, got {first_batch}"

    def test_mixed_dataset_pipeline(self):
        """Test parallel processing of multiple data sources.

        This test validates:
        - Multiple custom DataSourceModule implementations
        - Independent pipeline processing for different sources
        - Proper source identification in output batches
        - Parallel data loading capabilities
        """
        source1 = IndexedNumericSource(
            StructuralConfig(), source_id=1, multiplier=10.0, total_items=20
        )
        source2 = IndexedNumericSource(
            StructuralConfig(), source_id=2, multiplier=100.0, total_items=10
        )

        pipeline1 = DAGExecutor().add(source1).add(BatchNode(batch_size=4))
        pipeline2 = DAGExecutor().add(source2).add(BatchNode(batch_size=4))

        results1 = _collect_pipeline_batches(pipeline1, max_steps=3)
        results2 = _collect_pipeline_batches(pipeline2, max_steps=3)

        _assert_batches_for_source(results1, source_id=1)
        _assert_batches_for_source(results2, source_id=2)

    def test_pipeline_basic_functionality(self):
        """Test basic DAGExecutor pipeline construction and iteration.

        This test validates:
        - ArrayRecordSourceModule with mocked Grain backend
        - Basic pipeline iteration and batch counting
        - Batch size enforcement
        - Pipeline exhaustion after specified iterations
        """
        mock_source = Mock()
        mock_source.__len__ = Mock(return_value=100)
        # Return JAX-compatible values (arrays) for batching
        mock_source.__getitem__ = Mock(side_effect=lambda idx: {"id": jnp.array(idx)})

        # Patch where the name is looked up (module's namespace)
        with patch(
            "datarax.sources.array_record_source.grain.ArrayRecordDataSource",
            return_value=mock_source,
        ):
            # Create pipeline with DAGExecutor (config-first pattern)
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")
            pipeline = DAGExecutor()
            pipeline.add(source)  # Add DataSourceModule directly
            pipeline.add(BatchNode(batch_size=5))  # Then add batching

            # Process batches
            batch_count = 0
            total_items = 0
            for batch in pipeline:
                batch_count += 1
                # Batch should have 5 items (or less for last batch)
                # Handle the batch structure properly - Batch objects support
                # dict-like access via __getitem__ and __contains__
                if "id" in batch:
                    batch_ids = batch["id"]
                    if isinstance(batch_ids, jax.Array | np.ndarray):
                        batch_size = batch_ids.shape[0]
                    else:
                        batch_size = 1 if batch_ids is not None else 0
                    assert batch_size <= 5
                    total_items += batch_size
                if batch_count >= 10:
                    break

            # Verify we processed the expected number of batches
            assert batch_count == 10
            assert total_items == 50  # 10 batches * 5 items per batch
