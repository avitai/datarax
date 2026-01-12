"""Tests for Datarax's NNX module interfaces.

This module tests that nnx.Module implementations adhere to the
required interfaces and contracts.

NOTE: TransformerModule tests removed as part of Radical Unification.
TransformerModule was unified with AugmenterModule into OperatorModule.
See tests/operators/ for thorough OperatorModule tests (432+ tests).
"""

import flax.nnx as nnx
import jax.numpy as jnp

from datarax.core.batcher import BatcherModule
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule


def test_data_source_interface():
    """Test that DataSourceModule implementations adhere to the interface contract."""

    # Create a minimal implementation
    class MinimalDataSource(DataSourceModule):
        # REQUIRED: Annotate data with nnx.data()
        data: list = nnx.data()

        def __init__(self, config: StructuralConfig, data, *, rngs=None, name=None):
            super().__init__(config, rngs=rngs, name=name)
            self.data = data
            self.index = nnx.Variable(0)

        def __iter__(self):
            self.index.set_value(0)
            return self

        def __next__(self):
            if self.index.get_value() >= len(self.data):
                raise StopIteration
            item = self.data[self.index.get_value()]
            self.index.set_value(self.index.get_value() + 1)
            return item

        def __len__(self):
            return len(self.data)

    # Test the implementation
    source = MinimalDataSource(StructuralConfig(), [1, 2, 3, 4, 5])
    assert isinstance(source, DataSourceModule)
    assert hasattr(source, "__iter__")
    assert hasattr(source, "__next__")

    # Test iteration
    items = list(source)
    assert items == [1, 2, 3, 4, 5]

    # Test resetting iteration
    items = list(source)
    assert items == [1, 2, 3, 4, 5]

    # Verify NNX-specific functionality
    assert hasattr(source, "get_state")
    assert hasattr(source, "set_state")

    # The index should be managed by the NNX variable
    assert source.index.get_value() == 5


def test_operator_interface():
    """Test that OperatorModule implementations follow the interface contract.

    NOTE: This is a simplified test. See tests/operators/ for thorough
    OperatorModule tests (432+ tests covering all operator types).
    """

    # Create a minimal deterministic operator
    class MinimalOperator(OperatorModule):
        def __init__(self, *, rngs=None):
            config = OperatorConfig(stochastic=False)
            super().__init__(config, rngs=rngs)

        def apply(self, data, state, metadata, random_params=None, stats=None):
            # Double the values in the data
            new_data = {k: v * 2 for k, v in data.items()}
            return new_data, state, metadata

    # Test the implementation
    operator = MinimalOperator()
    assert isinstance(operator, OperatorModule)

    # Test apply method with sample element
    data = {"value": jnp.array([1, 2, 3])}
    new_data, _, _ = operator.apply(data, {}, None)
    assert jnp.array_equal(new_data["value"], jnp.array([2, 4, 6]))


def test_batcher_interface():
    """Test that BatcherModule implementations follow the interface contract.

    BatcherModule is now migrated to config-based StructuralModule architecture.
    """

    # Create a minimal implementation
    class MinimalBatcher(BatcherModule):
        def __init__(self, config: StructuralConfig, batch_size=2, *, rngs=None):
            super().__init__(config, rngs=rngs)
            self.batch_size = batch_size

        def create_batch(self, elements):
            return elements

    # Test the implementation
    config = StructuralConfig(stochastic=False)
    batcher = MinimalBatcher(config, batch_size=2)
    assert isinstance(batcher, BatcherModule)

    # Test creating batch
    batch = batcher.create_batch([1, 2])
    assert batch == [1, 2]


def test_data_source_extensibility():
    """Test that the DataSourceModule interface can be extended with custom methods."""

    # Define a custom source with additional methods
    class CustomDataSource(DataSourceModule):
        # REQUIRED: Annotate data with nnx.data()
        data: list = nnx.data()

        def __init__(self, config: StructuralConfig, data, *, rngs=None, name=None):
            super().__init__(config, rngs=rngs, name=name)
            self.data = data
            self.index = nnx.Variable(0)

        def __iter__(self):
            self.index.set_value(0)
            return self

        def __next__(self):
            if self.index.get_value() >= len(self.data):
                raise StopIteration
            item = self.data[self.index.get_value()]
            self.index.set_value(self.index.get_value() + 1)
            return item

        def __len__(self):
            return len(self.data)

        def get_length(self):
            """Custom method to get the length of data."""
            return len(self.data)

        def get_item_at(self, idx):
            """Custom method to get a specific item."""
            return self.data[idx]

    # Test the implementation
    source = CustomDataSource(StructuralConfig(), [1, 2, 3, 4, 5])
    assert isinstance(source, DataSourceModule)

    # Test standard iteration
    items = list(source)
    assert items == [1, 2, 3, 4, 5]

    # Test custom methods
    assert source.get_length() == 5
    assert source.get_item_at(2) == 3


def test_operator_extensibility():
    """Test that the OperatorModule interface can be extended with custom methods.

    NOTE: See tests/operators/ for thorough extensibility tests.
    """

    # Define a custom operator with additional methods
    class CustomOperator(OperatorModule):
        def __init__(self, factor=2, *, rngs=None):
            config = OperatorConfig(stochastic=False)
            super().__init__(config, rngs=rngs)
            self.factor = nnx.Param(factor)

        def apply(self, data, state, metadata, random_params=None, stats=None):
            new_data = {k: v * self.factor.get_value() for k, v in data.items()}
            return new_data, state, metadata

        def get_factor(self):
            """Custom method to get the multiplication factor."""
            return self.factor.get_value()

        def set_factor(self, factor):
            """Custom method to set the multiplication factor."""
            self.factor.set_value(factor)

    # Test the implementation
    operator = CustomOperator(factor=3)
    assert isinstance(operator, OperatorModule)

    # Test custom methods
    assert operator.get_factor() == 3

    # Test apply with updated factor
    operator.set_factor(4)
    assert operator.get_factor() == 4
    data = {"value": jnp.array(5.0)}
    new_data, _, _ = operator.apply(data, {}, None)
    assert new_data["value"] == 20.0


def test_batcher_extensibility():
    """Test that the BatcherModule interface can be extended with custom methods.

    BatcherModule is now migrated to config-based StructuralModule architecture.
    """

    # Define a custom batcher with additional methods
    class CustomBatcher(BatcherModule):
        def __init__(
            self, config: StructuralConfig, batch_size=2, drop_remainder=False, *, rngs=None
        ):
            super().__init__(config, rngs=rngs)
            self.batch_size = batch_size
            self.drop_remainder = nnx.Variable(drop_remainder)

        def create_batch(self, elements):
            return elements

        def get_batch_size(self):
            """Custom method to get the batch size."""
            return self.batch_size

        def set_drop_remainder(self, value):
            """Custom method to set drop_remainder flag."""
            self.drop_remainder.set_value(value)

    # Test the implementation
    config = StructuralConfig(stochastic=False)
    batcher = CustomBatcher(config, batch_size=3, drop_remainder=False)
    assert isinstance(batcher, BatcherModule)

    # Test custom methods
    assert batcher.get_batch_size() == 3
    batcher.set_drop_remainder(True)
    assert batcher.drop_remainder.get_value() is True


def test_nnx_module_integration():
    """Test integration between NNX-based modules."""

    # Define minimal source and operator
    class SimpleSource(DataSourceModule):
        # REQUIRED: Annotate data with nnx.data()
        data: list = nnx.data()

        def __init__(self, config: StructuralConfig, data, *, rngs=None, name=None):
            super().__init__(config, rngs=rngs, name=name)
            self.data = data
            self.index = nnx.Variable(0)

        def __iter__(self):
            self.index.set_value(0)
            return self

        def __next__(self):
            if self.index.get_value() >= len(self.data):
                raise StopIteration
            item = self.data[self.index.get_value()]
            self.index.set_value(self.index.get_value() + 1)
            return item

    class SimpleOperator(OperatorModule):
        def __init__(self, *, rngs=None):
            config = OperatorConfig(stochastic=False)
            super().__init__(config, rngs=rngs)

        def apply(self, data, state, metadata, random_params=None, stats=None):
            new_data = {k: v * 2 for k, v in data.items()}
            return new_data, state, metadata

    # Test pipeline flow
    source = SimpleSource(StructuralConfig(), [1, 2, 3])
    operator = SimpleOperator()

    # Process elements through the pipeline (simplified - just using apply)
    transformed_items = []
    for item in source:
        data = {"value": jnp.array(item)}
        new_data, _, _ = operator.apply(data, {}, None)
        transformed_items.append(int(new_data["value"]))

    assert transformed_items == [2, 4, 6]
