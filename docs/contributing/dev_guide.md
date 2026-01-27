# Developer Guide

This guide covers everything you need to know to contribute to Datarax development.

## Development Environment Setup

Datarax uses `uv` as its package manager for all installation, development, and deployment tasks.

### Quick Start

```bash
# Install uv if not already installed
pip install uv

# Run the automatic setup script
./setup.sh

# Activate the environment
source activate.sh
```

### Setup Script Options

The `setup.sh` script provides several options:

| Option | Description |
|--------|-------------|
| `--deep-clean` | Perform complete cleaning (JAX cache, pip cache, etc.) |
| `--cpu-only` | Force CPU-only setup (skip GPU detection) |
| `--force` | Force reinstallation even if environment exists |
| `--verbose, -v` | Show detailed output during setup |
| `--help, -h` | Show help message |

Example usage:

```bash
./setup.sh                    # Standard setup with auto GPU detection
./setup.sh --deep-clean       # Clean setup with cache clearing
./setup.sh --cpu-only         # Force CPU-only development setup
./setup.sh --force --verbose  # Verbose forced reinstallation
```

### Files Created by Setup

| File | Purpose |
|------|---------|
| `.venv/` | Virtual environment directory |
| `.env` | Environment variables and CUDA configuration |
| `activate.sh` | Activation script |
| `uv.lock` | Dependency lock file |

## Package Management

### Installing Dependencies

Datarax defines dependencies in `pyproject.toml` using optional dependency groups:

```bash
# Install all dependencies
uv pip install -e ".[all]"

# Install specific groups
uv pip install -e ".[dev]"      # Development tools
uv pip install -e ".[test]"     # Testing dependencies
uv pip install -e ".[docs]"     # Documentation tools
uv pip install -e ".[data]"     # Data loading (HF, TFDS, etc.)
uv pip install -e ".[gpu]"      # GPU support (CUDA 12)
```

### Adding New Dependencies

```bash
# Add a runtime dependency (edit pyproject.toml manually)
# Then sync:
uv sync

# Or use uv add for development:
uv add package_name
```

### Installing Multiple Extras

> **Important:** `uv sync` and `uv pip install` have different syntax for extras.

```bash
# ✅ Correct: pip-style bracket syntax (commas inside brackets)
uv pip install -e ".[dev,test,data]"

# ✅ Correct: multiple --extra flags for uv sync
uv sync --extra dev --extra test --extra data

# ✅ Recommended: use compound extras defined in pyproject.toml
uv sync --extra all      # includes dev, test, data, docs, gpu
uv sync --extra all-cpu  # includes dev, test, data, docs (no gpu)

# ❌ Wrong: comma-separated values with --extra flag
# uv sync --extra dev,test,data  # This will ERROR!
```

### Dependency Groups

| Group | Contents |
|-------|----------|
| `dev` | Build tools, linters, type checkers, pytest plugins |
| `test` | Testing dependencies (pytest, coverage, etc.) |
| `docs` | Documentation tools (MkDocs, mkdocstrings) |
| `data` | Data loading libraries (datasets, tensorflow-datasets) |
| `gpu` | CUDA 12 support for JAX |
| `all` | All of the above |

## Type Checking

Datarax uses Pyright for static type checking. Configuration is in `pyproject.toml`:

```toml
[tool.pyright]
exclude = ["examples", "scripts", ".deprecated", "**/__pycache__", "**/.venv"]
include = ["src", "tests"]
```

All code in `src/` and `tests/` directories is type-checked. Certain rules are relaxed to accommodate JAX's dynamic typing patterns.

### Running Type Checks

```bash
# Run Pyright manually
uv run pyright

# Through pre-commit
uv run pre-commit run pyright --all-files
```

### Type Annotation Guidelines

When writing new code:

1. **Add type annotations** to all function signatures (parameters and return types)
2. **Use proper generics** for container types (e.g., `list[int]` instead of `list`)
3. **Avoid `Any`** whenever possible; use specific types or `TypeVar` for generic code
4. **Handle `None`** explicitly with `Optional[T]` or `T | None` syntax
5. **Use `jax.Array`** for JAX array types

### Common Type Checking Issues

- **Optional Types**: Always check if a value can be `None` before accessing attributes
- **JAX Arrays**: Use `jax.Array` for JAX array types
- **Type Narrowing**: Use appropriate guards (`isinstance()`, etc.) to narrow types properly
- **Union Types**: Ensure all operations are valid for all possible types in a union

## Code Style

Datarax follows standard Python code style practices enforced by Ruff:

| Setting | Value |
|---------|-------|
| Line length | 100 characters |
| Quote style | Double quotes |
| Docstring convention | Google style |
| Import sorting | isort-compatible |
| Target Python | 3.11+ |

### Running Linters

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Ruff Configuration

Key Ruff settings in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.ruff.lint.pydocstyle]
convention = "google"
```

## Pre-commit Hooks

Pre-commit hooks run automatically on every commit to ensure code quality.

### Setup

```bash
# Install pre-commit hooks (done automatically by setup.sh)
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files
```

### Configured Hooks

| Hook | Purpose |
|------|---------|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with newline |
| `check-yaml` | Validate YAML syntax |
| `check-toml` | Validate TOML syntax |
| `check-json` | Validate JSON syntax |
| `check-added-large-files` | Prevent large files (>2MB) |
| `ruff` | Linting with auto-fix |
| `ruff-format` | Code formatting |
| `pyright` | Type checking |
| `bandit` | Security scanning |
| `pydocstyle` | Docstring style checking |
| `shellcheck` | Shell script linting |
| `nbqa-ruff` | Notebook linting |

### Skipping Hooks

If you need to skip hooks temporarily (not recommended):

```bash
git commit --no-verify -m "message"
```

## Testing

### Running Tests

```bash
# Run all tests (CPU-only, most stable)
JAX_PLATFORMS=cpu uv run pytest

# Run specific test module
JAX_PLATFORMS=cpu uv run pytest tests/sources/test_memory_source_module.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/datarax --cov-report=html
```

### Test Categories

Tests use pytest markers for categorization:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit tests |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.gpu` | Tests requiring GPU |
| `@pytest.mark.gpu_required` | Tests that must have GPU |
| `@pytest.mark.slow` | Slow-running tests |
| `@pytest.mark.benchmark` | Performance benchmarks |
| `@pytest.mark.tfds` | TensorFlow Datasets tests |
| `@pytest.mark.hf` | HuggingFace Datasets tests |

### Running Specific Test Types

```bash
# Skip GPU tests
uv run pytest -m "not gpu"

# Run only integration tests
uv run pytest -m integration

# Run only unit tests (fast)
uv run pytest -m unit

# Run benchmarks
uv run pytest -m benchmark --benchmark-autosave
```

### Test Directory Structure

Tests mirror the source structure:

```text
tests/
├── augment/         # Augmentation tests
├── batching/        # Batch processing tests
├── benchmarking/    # Benchmarking utility tests
├── checkpoint/      # Checkpoint tests
├── cli/             # CLI tests
├── config/          # Configuration tests
├── control/         # Control flow tests
├── core/            # Core functionality tests
├── dag/             # DAG execution tests
├── distributed/     # Distributed training tests
├── integration/     # End-to-end tests
├── memory/          # Memory management tests
├── monitoring/      # Monitoring tests
├── operators/       # Pipeline operator tests
├── performance/     # Performance tests
├── samplers/        # Sampling tests
├── sharding/        # Sharding tests
├── sources/         # Data source tests
└── conftest.py      # Pytest configuration
```

### Writing New Tests

1. Place tests in the directory matching the module being tested
2. Name files `test_<component>.py`
3. Name test functions `test_<behavior>()`
4. Use appropriate markers for hardware requirements
5. Create standalone tests that don't depend on other test files

Example:

```python
import numpy as np
import pytest
from datarax.sources import MemorySource, MemorySourceConfig

@pytest.mark.unit
def test_memory_source_initialization():
    """Test that MemorySource initializes correctly."""
    config = MemorySourceConfig()
    data = {"x": np.array([1, 2, 3])}
    source = MemorySource(config, data=data)
    assert source is not None
    assert len(source) == 3
```

## Building and Packaging

### Building the Package

```bash
# Build source distribution and wheel
uv run python -m build

# Build outputs go to dist/
ls dist/
# datarax-0.1.0.tar.gz
# datarax-0.1.0-py3-none-any.whl
```

### Package Configuration

Build settings in `pyproject.toml`:

```toml
[build-system]
build-backend = "hatchling.build"
requires = ["hatchling>=1.18"]

[tool.hatch.build.targets.wheel]
packages = ["src/datarax"]
```

## GPU/CUDA Support

### Automatic Detection

The setup script automatically detects NVIDIA GPUs and configures CUDA support.

### Manual GPU Setup

```bash
# Force GPU setup
./setup.sh --force

# Or install GPU extras manually
uv pip install -e ".[gpu]"
```

### Environment Variables for GPU

The `.env` file configures JAX for GPU:

```bash
# GPU configuration
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"
```

### Testing GPU Support

```bash
# Check GPU availability
python -c "import jax; print(jax.devices())"

# Run GPU tests
uv run pytest -m gpu
```

## Utility Scripts

Located in `scripts/`:

| Script | Purpose |
|--------|---------|
| `run_tests.sh` | Run tests with standard configuration |
| `run_gpu_tests.sh` | Run GPU-specific tests |
| `run_benchmarks.sh` | Run performance benchmarks |
| `run_lint.sh` | Run linting tools |
| `run_typecheck.sh` | Run type checking |
| `check_gpu.py` | Check GPU availability |
| `validate_examples.py` | Validate example code |
| `generate_docs.py` | Generate documentation |
| `submit_vertex_job.py` | Submit jobs to Vertex AI |

### Running Scripts

```bash
# Run tests
./scripts/run_tests.sh

# Check GPU
uv run python scripts/check_gpu.py

# Validate examples
uv run python scripts/validate_examples.py
```

## Environment Variables

Key environment variables for development:

| Variable | Purpose | Default |
|----------|---------|---------|
| `JAX_PLATFORMS` | JAX device platforms | `cpu` or `cuda,cpu` |
| `JAX_ENABLE_X64` | Enable 64-bit floats | `0` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | GPU memory preallocation | `false` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | GPU memory fraction | `0.8` |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow logging level | `1` |

## Documentation

### Building Documentation

```bash
# Serve documentation locally
uv run mkdocs serve

# Build static documentation
uv run mkdocs build
```

### Documentation Structure

```text
docs/
├── index.md                 # Home page
├── getting_started/         # Installation and quick start
├── user_guide/              # User documentation
│   ├── data_sources.md
│   ├── dag_construction.md
│   ├── distributed_training.md
│   └── ...
├── examples/                # Example documentation
├── core/, operators/, ...   # API reference pages
├── api_reference/           # Consolidated API reference
└── contributing/            # Contribution guidelines
    ├── contributing_guide.md
    ├── dev_guide.md                      # This guide
    ├── testing_guide.md
    ├── test_structure.md
    ├── gpu_testing.md
    ├── type_issues_guide.md
    ├── example_documentation_design.md
    └── performance_optimization_guide.md
```

## Troubleshooting

### Common Issues

**Import errors after installation:**

```bash
# Reinstall in development mode
uv pip install -e ".[all]"
```

**GPU not detected:**

```bash
# Check NVIDIA drivers
nvidia-smi

# Force GPU reinstall
./setup.sh --force
```

**Pre-commit hook failures:**

```bash
# Update hooks
uv run pre-commit autoupdate

# Run specific hook
uv run pre-commit run <hook-id> --all-files
```

**Type checking errors:**

```bash
# Run with verbose output
uv run pyright --verbose

# Check specific file
uv run pyright src/datarax/module.py
```

### Getting Help

- Check existing [GitHub Issues](https://github.com/avitai/datarax/issues)
- Read the [API documentation](https://datarax.readthedocs.io)
- Review test files for usage examples
