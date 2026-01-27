# Datarax Performance Optimization Guide

This document provides guidelines and best practices for optimizing Datarax components and contributing performance improvements.

## Performance Optimization Principles

When optimizing Datarax components, consider these key principles:

1. **Profile First**: Always identify bottlenecks through profiling before optimization
2. **Measure Impact**: Benchmark before and after changes to quantify improvements
3. **Maintain Correctness**: Ensure optimized code produces the same results
4. **Balance Readability**: Don't sacrifice code clarity without significant gains
5. **Hardware Awareness**: Consider different execution environments (CPU, GPU, TPU)

## Common Optimization Techniques

### 1. JIT Compilation

Use NNX JIT compilation for compute-intensive operations:

```python
from flax import nnx

# Before optimization
def compute_function(x):
    # Complex computation
    return result

# After optimization
@nnx.jit
def compute_function(x):
    # Same computation, but JIT-compiled
    return result
```

For operator components, Datarax operators automatically JIT-compile their batch processing:

```python
from flax import nnx
from datarax.operators import ElementOperator, ElementOperatorConfig

# Create a JIT-optimized operator
config = ElementOperatorConfig(stochastic=False)

def my_transform(element, key=None):
    # Your transformation - automatically JIT-compiled via apply_batch()
    return element.update_data({"image": element.data["image"] / 255.0})

# The operator's apply_batch() method uses vmap for automatic vectorization
operator = ElementOperator(config, fn=my_transform)
```

### 2. Vectorization

Datarax operators automatically vectorize operations via `vmap` in `apply_batch()`:

```python
from datarax.operators import ElementOperator, ElementOperatorConfig

# Operators automatically vectorize over the batch dimension
# The apply() method handles single elements, apply_batch() handles batches
config = ElementOperatorConfig(stochastic=False)
vectorized_op = ElementOperator(config, fn=your_transform_fn)
```

### 3. Prefetching

For data loading operations, use the device placement prefetching:

```python
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()

# Create a prefetching iterator for overlapping data transfer with compute
prefetched_iterator = placement.prefetch_to_device(
    data_iterator,
    buffer_size=2  # Prefetch 2 batches ahead
)
```

### 4. Composite Optimization

Build optimized pipelines by composing operators and using proper batch sizes:

```python
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.distributed.device_placement import get_batch_size_recommendation

# Get hardware-optimized batch size
rec = get_batch_size_recommendation()
optimal_batch = rec.optimal_batch_size

# Build an optimized pipeline with proper batch sizing
pipeline = (
    from_source(source, batch_size=optimal_batch)
    >> OperatorNode(normalize_op)
    >> OperatorNode(augment_op)
)
```

## Benchmarking Guidelines

When submitting performance improvements, include benchmark results:

1. **Test Environment**: Specify hardware (CPU, GPU, TPU), JAX version, etc.
2. **Benchmark Parameters**: Document batch sizes, data dimensions, etc.
3. **Before/After Metrics**: Report throughput (examples/sec) before and after
4. **Methodology**: Explain how the benchmark was conducted (warmup, iterations)
5. **Resource Usage**: Include memory usage if relevant

Example benchmark format:

```markdown
## Benchmark Results

- **Test Environment**: NVIDIA RTX 3090, JAX 0.6.1, Flax 0.12.0
- **Data Dimensions**: 1000 images, 224x224x3, batch size 32
- **Metrics**:

  - Before: 1,200 examples/sec, 90% GPU utilization
  - After: 4,800 examples/sec, 95% GPU utilization
  - Speedup: 4.0x
- **Methodology**: 10 warmup batches, 50 measurement batches, averaged over 3 runs
```

## Profiling Datarax

### Using JAX's Built-in Profiler

```python
from jax.profiler import start_trace, stop_trace

# Start profiling
start_trace("/tmp/profile_dir")

# Run your Datarax pipeline
for batch in pipeline:
    # Process batch
    pass

# Stop profiling
stop_trace()
```

### Using Datarax's Benchmark Utilities

```python
from datarax.benchmarking.pipeline_throughput import PipelineBenchmark

# Create a benchmark for your pipeline (pass DAGExecutor from from_source())
benchmark = PipelineBenchmark(
    data_stream=pipeline,
    num_batches=50,
    warmup_batches=5,
)

# Run the benchmark and get results
results = benchmark.run(pipeline_seed=42)
print(f"Throughput: {results['examples_per_second']:.2f} examples/sec")
print(f"Batches per second: {results['batches_per_second']:.2f}")
print(f"Duration: {results['duration_seconds']:.4f}s")
```

## Common Performance Pitfalls

1. **Excessive Python Overhead**: Prefer JAX-native operations over Python loops
2. **Small Operations**: Fuse small operations into larger ones when possible
3. **Unnecessary Data Transfers**: Minimize host-device transfers
4. **Recompilation**: Avoid unnecessary recompilation by controlling static args
5. **Large Batch Variance**: Handle variable batch sizes carefully

## Submitting Performance Improvements

When submitting a PR with performance improvements:

1. **Focus on Hot Spots**: Target components identified through profiling
2. **Include Benchmarks**: Document performance gains with benchmarks
3. **Maintain API Compatibility**: Ensure optimized components maintain the same API
4. **Add Tests**: Include tests that verify correctness of optimized code
5. **Document Tradeoffs**: Note any memory-speed tradeoffs or limitations

## Example PR Description Template

```markdown
# Performance Improvement: [Brief Description]

## Problem
[Describe the performance bottleneck identified through profiling]

## Solution
[Explain the optimization approach used]

## Benchmark Results
[Include before/after metrics as described above]

## Tradeoffs
[Document any tradeoffs (e.g., memory vs. speed)]

## Testing
[Describe how correctness was verified]
```

## Hardware-Specific Optimizations

### GPU Optimizations

- Prioritize vectorization and batched processing
- Use larger batch sizes (typically 32-64)
- Consider memory constraints when setting cache sizes

### CPU Optimizations

- Prioritize caching over repeated computation
- Use moderate batch sizes (typically 16-32)
- Consider CPU-specific compilation options

### TPU Optimizations

- Design for spatial/model parallelism
- Use large batch sizes (typically 64-128)
- Avoid operations not well-supported on TPUs

## Running Benchmarks

Datarax provides benchmark utilities for measuring performance:

```bash
# Run all benchmarks
./scripts/run_benchmarks.sh

# Run specific benchmark tests
JAX_PLATFORMS=cpu uv run pytest -m benchmark --benchmark-autosave

# View benchmark results
ls benchmark-results/
```

## Conclusion

By following these guidelines, you can contribute meaningful performance improvements to Datarax while maintaining code quality and correctness. Remember that the best optimizations are those that target actual bottlenecks identified through profiling, and that provide significant speedups with minimal impact on code readability and maintainability.

For general development setup and tools, see the [Developer Guide](dev_guide.md).
