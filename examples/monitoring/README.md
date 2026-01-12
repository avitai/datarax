# Datarax Monitoring Examples

This directory contains examples demonstrating how to use Datarax's monitoring system.

## Examples

### Simple Monitoring (`simple_monitoring.py`)

This basic example demonstrates:
- Creating a `MonitoredPipeline` from a data source
- Registering metrics reporters
- Creating a data processing pipeline with transformations
- Recording custom metrics during processing
- Viewing metrics in real-time via console output

### File Reporter (`file_reporter_example.py`)

This example shows:
- Using the `FileReporter` to save metrics to a CSV file
- Combining multiple reporters (Console + File)
- Recording various types of metrics from pipeline operations
- Organizing metrics by category
- Persisting monitoring data for later analysis

## Running the Examples

To run any example:

```bash
# From the root directory of the repository:
python examples/monitoring/simple_monitoring.py
# or
python examples/monitoring/file_reporter_example.py
```

## Key Components of Datarax Monitoring

Datarax's monitoring system consists of several key components:

1. **MonitoredPipeline**: A pipeline that includes metrics collection.
2. **Reporters**: Components that display or save metrics (Console, File, etc.)
3. **MetricsCollector**: The core system for collecting and storing metrics.
4. **Callbacks**: The event system that triggers reporting and other monitoring functions.

Refer to the Datarax documentation for more detailed information on the monitoring system.
