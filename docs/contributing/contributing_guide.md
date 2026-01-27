# Contributing to Datarax

Thank you for your interest in contributing to Datarax! This guide covers everything you need to know about contributing to the project, with a focus on the NNX-based architecture.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment-setup)
- [NNX Architecture Requirements](#nnx-architecture-requirements)
- [Testing Requirements](#testing-requirements)
- [Code Style and Quality](#code-style-and-quality)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended)
- Git
- Basic understanding of JAX and Flax NNX

### Development Environment Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/avitai/datarax.git
   cd datarax
   ```

2. **Set up the development environment** (recommended):

   ```bash
   # Run the automated setup script (handles GPU detection, dependencies, etc.)
   ./setup.sh

   # Activate the environment
   source activate.sh
   ```

   Or manually:

   ```bash
   # Create virtual environment with uv
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install all development dependencies
   uv pip install -e ".[all]"
   ```

3. **Verify installation**:

   ```bash
   # Run a quick test to ensure everything works
   JAX_PLATFORMS=cpu uv run pytest tests/sources/test_memory_source_module.py -v
   ```

   See the [Developer Guide](dev_guide.md) for detailed setup instructions and configuration options.

## NNX Architecture Requirements

Datarax follows a **Flax NNX-based architecture**. All contributions must adhere to these architectural principles:

### 1. Module Inheritance

All Datarax components **must** inherit from appropriate base classes:

```python
from datarax.core import DataraxModule
import flax.nnx as nnx

# ✅ Correct: Inherit from DataraxModule
class MyCustomModule(DataraxModule):
    def __init__(self, param=10, name="my_module"):
        super().__init__(name=name)  # Always call super().__init__
        self.param = nnx.Variable(param)  # Use NNX Variables for state

# ❌ Wrong: Direct inheritance from nnx.Module
class WrongModule(nnx.Module):  # Don't do this!
    pass
```

### 2. State Management with NNX Variables

All mutable state **must** be stored in NNX Variables:

```python
class StatefulModule(DataraxModule):
    def __init__(self, buffer_size=100):
        super().__init__()
        # ✅ Correct: Use nnx.Variable for mutable state
        self.buffer_size = nnx.Variable(buffer_size)
        self.position = nnx.Variable(0)
        self.internal_state = nnx.Variable({})

        # ❌ Wrong: Direct assignment
        # self.buffer_size = buffer_size  # Don't do this!

    def update_position(self, new_pos):
        # ✅ Correct: Access via slice notation (Flax 0.12.0+)
        self.position[...] = new_pos

        # ❌ Wrong: Direct assignment
        # self.position = new_pos  # Don't do this!
```

### 3. Checkpointing Support

All modules **must** support checkpointing:

```python
class CheckpointableModule(DataraxModule):
    def get_state(self):
        """Return current state for checkpointing."""
        state = super().get_state()
        # Add custom state if needed
        state.update({
            'custom_field': self.custom_value[...]
        })
        return state

    def set_state(self, state):
        """Restore state from checkpoint."""
        super().set_state(state)
        # Restore custom state if needed
        if 'custom_field' in state:
            self.custom_value[...] = state['custom_field']

    def get_serializable_state(self):
        """Override for complex serialization."""
        state = super().get_serializable_state()
        # Convert non-serializable objects
        return self._clean_state(state)
```

### 4. PRNG Handling

Random number generation **must** follow NNX patterns:

```python
class RandomModule(DataraxModule):
    def __init__(self, seed=0, name="random_module"):
        super().__init__(name=name)
        # ✅ Correct: Use nnx.Rngs
        self.rngs = nnx.Rngs(default=seed)

    def generate_random(self):
        # ✅ Correct: Use self.rngs
        key = self.rngs.default()
        return jax.random.normal(key, (10,))

    def get_state(self):
        """Include PRNG state in checkpoints."""
        state = super().get_state()
        state.update({
            'rng_key': self.rngs.default.key[...],
            'rng_count': self.rngs.default.count[...]
        })
        return state
```

## Testing Requirements

### 1. Test Coverage

All new code **must** have complete test coverage:

- **Unit Tests**: Test individual methods and functions
- **Integration Tests**: Test component interactions
- **NNX Tests**: Test state management and checkpointing
- **Regression Tests**: Prevent breaking existing functionality

### 2. NNX-Specific Testing

All NNX modules **must** include these test categories:

```python
import pytest
import flax.nnx as nnx
from your_module import YourModule

class TestYourModuleNNX:
    """NNX-specific tests for YourModule."""

    def test_state_management(self):
        """Test state can be retrieved and set."""
        module = YourModule(param=42)

        # Test state retrieval
        state = module.get_state()
        assert 'param' in state

        # Test state modification
        new_module = YourModule(param=0)
        new_module.set_state(state)
        assert new_module.param[...] == 42

    def test_checkpointing_compatibility(self):
        """Test module can be checkpointed with Orbax."""
        module = YourModule()

        # Test serializable state
        serializable_state = module.get_serializable_state()
        assert isinstance(serializable_state, dict)

        # Test state restoration
        new_module = YourModule()
        new_module.set_state(serializable_state)

    def test_variable_access_patterns(self):
        """Test correct Variable access patterns."""
        module = YourModule(param=10)

        # Test slice notation access (Flax 0.12.0+)
        assert module.param[...] == 10

        # Test slice notation modification
        module.param[...] = 20
        assert module.param[...] == 20

    def test_rngs_handling(self):
        """Test PRNG state management."""
        if hasattr(YourModule, 'rngs'):
            module = YourModule(seed=42)

            # Test deterministic behavior
            result1 = module.generate_random_value()
            module.rngs.default.key[...] = jax.random.key(42)
            module.rngs.default.count[...] = 0
            result2 = module.generate_random_value()

            assert jnp.allclose(result1, result2)
```

### 3. Test Organization

Tests **must** follow the project structure (mirroring `src/datarax`):

```text
tests/
├── augment/                 # Augmentation tests
├── batching/                # Batch processing tests
├── benchmarking/            # Benchmarking infrastructure tests
├── benchmarks/              # Performance benchmark tests
├── checkpoint/              # Checkpointing tests
├── cli/                     # CLI tool tests
├── config/                  # Configuration tests
├── control/                 # Control flow tests
├── core/                    # Core module tests
├── dag/                     # DAG execution tests
├── data/                    # Test data and fixtures
├── distributed/             # Distributed processing tests
├── examples/                # Example validation tests
├── integration/             # End-to-end integration tests
├── memory/                  # Memory management tests
├── monitoring/              # Monitoring functionality tests
├── operators/               # Operator tests
├── performance/             # Performance tests
├── samplers/                # Sampler tests
├── sharding/                # Sharding tests
├── sources/                 # Data source tests
├── test_common/             # Common testing utilities
├── transforms/              # Transform tests (neural network ops)
├── utils/                   # Utility function tests
└── conftest.py              # Pytest configuration and markers
```

See the [Developer Guide](dev_guide.md#testing) and [Test Structure Guide](test_structure.md) for the complete test directory structure.

### 4. Running Tests

Use the standardized test commands:

```bash
# Run all tests (CPU-only for stability)
JAX_PLATFORMS=cpu uv run pytest

# Run specific test categories
JAX_PLATFORMS=cpu uv run pytest tests/core/           # Core tests
JAX_PLATFORMS=cpu uv run pytest tests/sources/        # Data source tests
JAX_PLATFORMS=cpu uv run pytest -k "checkpoint"       # Checkpointing tests

# Run with coverage
JAX_PLATFORMS=cpu uv run pytest --cov=src/datarax --cov-report=html

# Run integration tests
JAX_PLATFORMS=cpu uv run pytest tests/integration/

# Run GPU tests (requires CUDA)
JAX_PLATFORMS=cuda uv run pytest --device=gpu -m gpu

# Or use the automated test runner
./run_tests.sh  # Runs on CPU, then GPU if available
```

See the [Developer Guide](dev_guide.md#testing) for comprehensive testing documentation.

## Code Style and Quality

### 1. Code Formatting

We use `ruff` for both formatting and linting:

```bash
# Format code
uv run ruff format src/ tests/

# Check formatting (without making changes)
uv run ruff format --check src/ tests/
```

### 2. Linting and Pre-commit

All code must pass linting checks. We use pre-commit hooks to automate quality checks:

```bash
# Install pre-commit hooks (run once after cloning)
uv run pre-commit install

# Run all checks before committing
uv run pre-commit run --all-files

# Individual tools:
uv run ruff check src/ tests/           # Linting
uv run ruff check --fix src/ tests/     # Auto-fix lint issues
uv run ruff format src/ tests/          # Format code
uv run pyright src/                     # Type checking
```

Pre-commit hooks automatically run: Ruff (linting + formatting), Pyright (type checking), Bandit (security), and more. See the [Developer Guide](dev_guide.md#pre-commit-hooks) for the full list.

### 3. Type Annotations

All public APIs **must** include type annotations using Python 3.11+ style:

```python
from typing import Any
import jax

def process_data(
    data: jax.Array,
    config: dict[str, Any],
    seed: int | None = None
) -> jax.Array:
    """Process data with given configuration.

    Args:
        data: Input data array
        config: Processing configuration
        seed: Optional random seed

    Returns:
        Processed data array
    """
    # Implementation
    return processed_data
```

**Note**: Use `jax.Array` for JAX arrays (not `jnp.ndarray`), and modern union syntax `X | None` instead of `Optional[X]`.

### 4. Documentation Strings

All classes and functions **must** have complete docstrings:

```python
from typing import Any
from flax import nnx
from datarax.core import DataraxModule

class DataProcessor(DataraxModule):
    """Process data using configurable transformations.

    This module provides a flexible framework for applying
    transformations to input data with state management.

    Attributes:
        config: Processing configuration stored as NNX Variable
        stats: Running statistics for processed data

    Example:
        >>> processor = DataProcessor({"normalize": True})
        >>> result = processor.process(data)
    """

    def __init__(self, config: dict[str, Any], name: str = "processor"):
        """Initialize the data processor.

        Args:
            config: Configuration dictionary for processing
            name: Module name for identification
        """
        super().__init__(name=name)
        self.config = nnx.Variable(config)
```

## Documentation

### 1. API Documentation

All public APIs must be documented:

- Include complete docstrings
- Provide usage examples
- Document state management patterns
- Explain checkpointing behavior

### 2. User Guides

For significant features, provide user guides:

- Step-by-step tutorials
- Best practices
- Common patterns
- Troubleshooting tips

### 3. Examples

Include practical examples:

```python
# examples/new_feature_example.py
"""Example demonstrating new feature usage."""

import jax
import flax.nnx as nnx
from datarax.core import DataraxModule

def main():
    """Run the example."""
    # Create components
    # Demonstrate usage
    # Show checkpointing
    pass

if __name__ == "__main__":
    main()
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] NNX architecture requirements met
- [ ] Performance impact considered

### 2. PR Checklist

Your PR should include:

- [ ] **Clear description** of changes and motivation
- [ ] **Test coverage** for new functionality
- [ ] **Documentation updates** for API changes
- [ ] **Benchmark results** for performance changes
- [ ] **Update guidelines** for breaking changes

### 3. Review Process

1. **Automated checks**: All CI checks must pass
2. **Code review**: Maintainer review for architecture compliance
3. **Testing**: Complete test validation
4. **Documentation**: Verify documentation completeness

### 4. Merge Requirements

- All CI checks passing
- At least one maintainer approval
- No merge conflicts
- Documentation updated
- Test coverage maintained

## Common Contribution Areas

### 1. New Data Sources

When adding data sources:

```python
from datarax.core import DataSourceModule

class NewDataSource(DataSourceModule):
    """Template for new data sources."""

    def __init__(self, source_config, name="new_source"):
        super().__init__(name=name)
        self.config = nnx.Variable(source_config)

    def __iter__(self):
        """Implement iteration protocol."""
        # Your implementation
        pass
```

### 2. New Operators

When adding data transformation operators:

```python
from flax import nnx
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.typing import Element

# Option 1: Use ElementOperator with a custom function
def my_transform(element: Element, key=None) -> Element:
    """Transform single element."""
    # Your transformation logic
    return element.update_data({"field": transformed_value})

config = ElementOperatorConfig(stochastic=False)
operator = ElementOperator(config, fn=my_transform, rngs=nnx.Rngs(0))

# Option 2: Create a custom operator class
from datarax.core.operator import OperatorModule

class CustomOperator(OperatorModule):
    """Template for custom operators."""

    def apply(self, element: Element, key=None) -> Element:
        """Apply transformation to single element."""
        # Your implementation
        return transformed_element
```

### 3. Performance Optimizations

See the [Performance Optimization Guide](performance_optimization_guide.md) for detailed guidelines.

## Getting Help

- **Developer Guide**: See the [Developer Guide](dev_guide.md) for detailed development setup and tooling
- **Documentation**: Check the [API Reference](https://datarax.readthedocs.io) for usage examples
- **Issues**: Search [GitHub Issues](https://github.com/avitai/datarax/issues) for similar problems
- **New Issues**: Open a new issue with a clear description and minimal reproduction steps

## License

By contributing to Datarax, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Datarax! Your contributions help make robust, stateful data pipelines accessible to the JAX community.
