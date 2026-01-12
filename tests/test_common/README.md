# Common Test Utilities

This directory contains common utility functions and classes used across Datarax tests.

## Importing

To import from this directory, ensure that the `tests` directory is in your Python path.
This can be done by setting the `PYTHONPATH` environment variable to include the project root directory
before running tests:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/project/root
```

In GitHub Actions workflows, this is handled by explicitly adding the project root to `PYTHONPATH`.

## Contents

- `device_detection.py`: Functions for detecting available hardware devices
- `data_generators.py`: Functions for generating test data
- `data_source_fixtures.py`: Test fixtures for data source components
- `fixtures.py`: PyTest fixtures for tests
- `utils.py`: General test utilities

### Data Source Fixtures

The `data_source_fixtures.py` module provides specialized test fixtures for testing data source components:

- `StatefulDataSourceModule`: A test data source that maintains state between iterations, designed for testing state management, checkpointing, and pipeline functionality.

## Note on Naming

This directory was previously named `test_utils` but was renamed to `test_common` to avoid confusion with tests for the `utils` module in `src/datarax`. The new name better reflects the purpose of this directory as containing common utilities used across all tests.
