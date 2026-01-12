"""Complete pipeline performance benchmarks for Datarax.

This module provides comprehensive performance benchmarks for the DAGExecutor,
testing various configurations including linear pipelines, parallel processing,
conditional branching, and complex DAG topologies.
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from tests.test_common.data_generators import generate_image_data, generate_text_data

from datarax import DAGExecutor
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.core.element_batch import Element
from flax import nnx
from datarax.dag.nodes import (
    ShuffleNode,
    PrefetchNode,
)


# Mark all tests in this file as benchmarks
pytestmark = pytest.mark.benchmark


@pytest.fixture
def benchmark_image_data() -> list[dict[str, Any]]:
    """Generate image data for benchmarking."""
    data = generate_image_data(num_samples=10000, image_height=32, image_width=32)
    # Convert from batch format to list of individual samples with JAX arrays
    return [
        {"image": jnp.array(data["image"][i]), "label": jnp.array(data["label"][i])}
        for i in range(len(data["image"]))
    ]


@pytest.fixture
def benchmark_text_data() -> list[dict[str, Any]]:
    """Generate text data for benchmarking."""
    data = generate_text_data(num_samples=10000, max_seq_length=100)
    # Convert from batch format to list of individual samples with JAX arrays
    return [
        {
            "tokens": jnp.array(data["tokens"][i]),
            "length": jnp.array(data["length"][i]),
            "label": jnp.array(data["label"][i]),
        }
        for i in range(len(data["tokens"]))
    ]


@pytest.fixture
def rngs():
    """Create Rngs for testing."""
    return nnx.Rngs(default=42, augment=43, shuffle=44)


def benchmark_pipeline_configuration(
    data: list[dict[str, Any]],
    batch_size: int,
    num_transforms: int = 2,
    num_augmenters: int = 1,
    prefetch_size: int | None = None,
    buffer_size: int | None = None,
    benchmark_runner=None,
    warmup_steps: int = 3,
):
    """Benchmark a pipeline with various configurations."""
    # Create the data source with RNGs
    rngs = nnx.Rngs(default=42)
    source = MemorySource(MemorySourceConfig(), data, rngs=rngs)

    # Start building the pipeline with RNGs
    rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
    pipeline = DAGExecutor(rngs=rngs).add(source)

    # Add shuffle if buffer size is provided (BEFORE batching)
    if buffer_size is not None:
        pipeline = pipeline.add(ShuffleNode(buffer_size=buffer_size, seed=42))

    # Set batch size (BEFORE transforms and augmenters)
    pipeline = pipeline.batch(batch_size=batch_size)

    # Add transforms (deterministic operators) - Element API: fn(element, key) -> element
    det_config = ElementOperatorConfig(stochastic=False)
    for i in range(num_transforms):
        identity_op = ElementOperator(det_config, fn=lambda e, k: e)
        pipeline = pipeline.operate(identity_op)

    # Add augmenters if requested (stochastic operators)
    if num_augmenters > 0:
        stoch_config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        for i in range(num_augmenters):
            identity_aug = ElementOperator(stoch_config, fn=lambda e, k: e, rngs=rngs)
            pipeline = pipeline.operate(identity_aug)

    # Add prefetch if requested
    if prefetch_size is not None:
        pipeline = pipeline.add(PrefetchNode(buffer_size=prefetch_size))

    # Create iterator
    iterator = iter(pipeline)

    # Warmup (process a few batches to trigger JIT compilation)
    if warmup_steps > 0:
        warmup_batches = []
        for i, batch in enumerate(iterator):
            if hasattr(batch, "block_until_ready"):
                batch.block_until_ready()
            warmup_batches.append(batch)
            if i >= warmup_steps - 1:
                break

        # Reset and create new iterator for actual measurement
        pipeline.reset()
        iterator = iter(pipeline)

    # Measure performance
    start_time = time.time()
    num_batches = 0
    num_examples = 0

    for batch in iterator:
        # Ensure computation completes
        if hasattr(batch, "block_until_ready"):
            batch.block_until_ready()

        num_batches += 1
        # Count actual examples (handle last incomplete batch)
        if isinstance(batch, dict) and "image" in batch:
            num_examples += batch["image"].shape[0]
        elif isinstance(batch, dict) and "text" in batch:
            num_examples += batch["text"].shape[0]
        else:
            # Fallback - assume first dimension is batch size
            leaves = jax.tree_util.tree_leaves(batch)
            if leaves:
                first_value = leaves[0]
                if hasattr(first_value, "shape"):
                    num_examples += first_value.shape[0]
                else:
                    # If no shape, assume batch size of 1
                    num_examples += 1
            else:
                # Empty batch, skip
                continue

        # Stop after processing 50 batches or all data
        if num_batches >= 50:
            break

    end_time = time.time()
    duration = end_time - start_time

    # Calculate metrics
    examples_per_second = num_examples / duration if duration > 0 else 0
    batches_per_second = num_batches / duration if duration > 0 else 0

    result = {
        "duration_seconds": duration,
        "batches_processed": num_batches,
        "examples_processed": num_examples,
        "examples_per_second": examples_per_second,
        "batches_per_second": batches_per_second,
    }

    return result


@pytest.mark.parametrize("batch_size", [8, 32, 128])
def test_batch_size_impact(benchmark, benchmark_image_data, batch_size):
    """Benchmark the impact of batch size on pipeline performance."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]  # Use only 1000 samples

    def run_benchmark():
        # Run single configuration without internal timing
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs).add(source).batch(batch_size=batch_size)

        # Process limited batches
        iterator = iter(pipeline)
        num_batches = 0
        num_examples = 0

        for batch in iterator:
            num_batches += 1
            if isinstance(batch, dict) and "image" in batch:
                num_examples += batch["image"].shape[0]
            else:
                leaves = jax.tree_util.tree_leaves(batch)
                if leaves and hasattr(leaves[0], "shape"):
                    num_examples += leaves[0].shape[0]

            # Process only 10 batches for benchmarking
            if num_batches >= 10:
                break

        return {"batches": num_batches, "examples": num_examples}

    # Warmup - run once to trigger JAX JIT compilation of internal functions
    _ = run_benchmark()

    # Now benchmark
    result = benchmark(run_benchmark)
    assert result["batches"] > 0
    assert result["examples"] > 0


@pytest.mark.parametrize("transform_count", [0, 2, 5])
def test_transform_count_impact(benchmark, benchmark_image_data, transform_count):
    """Benchmark the impact of transform count on pipeline performance."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs).add(source)

        # Add batching first (before transforms)
        pipeline = pipeline.batch(batch_size=32)

        # Add transforms (deterministic operators) - Element API: fn(element, key) -> element
        det_config = ElementOperatorConfig(stochastic=False)
        for i in range(transform_count):
            identity_op = ElementOperator(det_config, fn=lambda e, k: e)
            pipeline = pipeline.operate(identity_op)

        # Process limited batches
        iterator = iter(pipeline)
        num_batches = 0

        for batch in iterator:
            num_batches += 1
            if num_batches >= 10:
                break

        return num_batches

    result = benchmark(run_benchmark)
    assert result > 0


@pytest.mark.parametrize("prefetch_size", [None, 2, 4])
def test_prefetch_impact(benchmark, benchmark_image_data, prefetch_size):
    """Benchmark the impact of prefetch on pipeline performance."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs).add(source).batch(batch_size=32)

        # Add prefetch if requested
        if prefetch_size is not None:
            pipeline = pipeline.add(PrefetchNode(buffer_size=prefetch_size))

        # Process limited batches
        iterator = iter(pipeline)
        num_batches = 0

        for batch in iterator:
            num_batches += 1
            if num_batches >= 10:
                break

        return num_batches

    result = benchmark(run_benchmark)
    assert result > 0


@pytest.mark.parametrize("buffer_size", [None, 100, 500])
def test_shuffle_impact(benchmark, benchmark_image_data, buffer_size):
    """Benchmark the impact of shuffle buffer size on pipeline performance."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs).add(source)

        # Add shuffle if buffer size is provided
        if buffer_size is not None:
            pipeline = pipeline.add(ShuffleNode(buffer_size=buffer_size, seed=42))

        pipeline = pipeline.batch(batch_size=32)

        # Process limited batches
        try:
            iterator = iter(pipeline)
            num_batches = 0

            for batch in iterator:
                num_batches += 1
                if num_batches >= 10:
                    break
        except Exception as e:
            print(f"Error during iteration: {e}")
            return 0

        return num_batches

    result = benchmark(run_benchmark)
    assert result > 0


@pytest.mark.parametrize("augmenter_count", [0, 1, 2])
def test_augmentation_impact(benchmark, benchmark_image_data, augmenter_count):
    """Benchmark the impact of augmentation on pipeline performance."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs).add(source)

        # Add batching first (before augmenters)
        pipeline = pipeline.batch(batch_size=32)

        # Add augmenters (stochastic operators) - Element API: fn(element, key) -> element
        stoch_config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        for i in range(augmenter_count):
            augmenter = ElementOperator(stoch_config, fn=lambda e, k: e, rngs=rngs)
            pipeline = pipeline.operate(augmenter)

        # Process limited batches
        iterator = iter(pipeline)
        num_batches = 0

        for batch in iterator:
            num_batches += 1
            if num_batches >= 10:
                break

        return num_batches

    result = benchmark(run_benchmark)
    assert result > 0


def test_end_to_end_optimal_configuration(benchmark, benchmark_image_data):
    """Benchmark an end-to-end pipeline with optimal configuration."""

    # Define a realistic pipeline with transforms and augmentations
    def build_optimal_pipeline():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), benchmark_image_data, rngs=rngs)

        # Define meaningful transforms (deterministic) - Element API: fn(element, key) -> element
        det_config = ElementOperatorConfig(stochastic=False)
        normalizer = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={"image": e.data["image"] / 255.0, "label": e.data["label"]}
            ),
        )
        scaler = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={"image": e.data["image"] * 2 - 1, "label": e.data["label"]}
            ),
        )

        # Random horizontal flip augmenter function (stochastic) - Element API
        def flip_fn(element: Element, rng: jax.Array) -> Element:
            # Use JAX's where instead of Python if/else for traceability
            should_flip = jax.random.uniform(rng) > 0.5
            flipped = jnp.flip(element.data["image"], axis=1)
            new_data = {
                "image": jnp.where(
                    should_flip[..., None, None, None], flipped, element.data["image"]
                ),
                "label": element.data["label"],
            }
            return element.replace(data=new_data)

        # Build pipeline with optimal settings
        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)

        # Create stochastic operator for flip
        stoch_config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        augmenter = ElementOperator(stoch_config, fn=flip_fn, rngs=rngs)

        pipeline = (
            DAGExecutor(rngs=rngs)
            .add(source)
            .add(ShuffleNode(buffer_size=1000, seed=42))
            .batch(batch_size=64)
            .operate(normalizer)
            .operate(scaler)
            .operate(augmenter)
            .add(PrefetchNode(buffer_size=2))
        )

        # Process data
        iterator = iter(pipeline)
        num_batches = 0
        num_examples = 0

        for batch in iterator:
            num_batches += 1
            num_examples += batch["image"].shape[0]

            # Stop after 50 batches
            if num_batches >= 50:
                break

        return {"batches_processed": num_batches, "examples_processed": num_examples}

    # Run the benchmark
    result = benchmark(build_optimal_pipeline)

    # Verify the pipeline processed data
    assert result["batches_processed"] > 0
    assert result["examples_processed"] > 0


@pytest.mark.parametrize("num_parallel", [2, 4, 8])
def test_parallel_processing_performance(benchmark, benchmark_image_data, num_parallel):
    """Benchmark parallel processing performance with different branch counts."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs).add(source).batch(batch_size=32)

        # Create parallel transforms (deterministic operators)
        # Element API: fn(element, key) -> element
        det_config = ElementOperatorConfig(stochastic=False)
        transforms = [
            ElementOperator(
                det_config,
                fn=lambda e, k, i=i: e.replace(
                    data={"image": e.data["image"] * (1.0 + i * 0.1), "label": e.data["label"]}
                ),
            )
            for i in range(num_parallel)
        ]

        # Add parallel processing
        pipeline = pipeline.parallel(transforms)

        # Add merge to combine results
        pipeline = pipeline.merge(strategy="stack", axis=0)

        # Process limited batches
        iterator = iter(pipeline)
        num_batches = 0
        results = []

        for batch in iterator:
            num_batches += 1
            results.append(batch)
            if num_batches >= 10:
                break

        return {"batches": num_batches, "parallel_branches": num_parallel}

    # Warmup
    _ = run_benchmark()

    # Benchmark
    result = benchmark(run_benchmark)
    assert result["batches"] > 0
    assert result["parallel_branches"] == num_parallel


@pytest.mark.parametrize("enable_caching", [False, True])
def test_caching_performance_impact(benchmark, benchmark_image_data, enable_caching):
    """Benchmark the impact of caching on repeated pipeline execution."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:500]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        # Create pipeline with or without caching
        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs, enable_caching=enable_caching)
        pipeline = pipeline.add(source).batch(batch_size=32)

        # Add transforms with potential cache benefit - Element API: fn(element, key) -> element
        det_config = ElementOperatorConfig(stochastic=False)
        expensive_transform = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={
                    "image": jnp.fft.fft2(e.data["image"], axes=(1, 2)).real,
                    "label": e.data["label"],
                }
            ),
        )

        pipeline = pipeline.operate(expensive_transform)

        # Run pipeline twice to test cache benefit
        total_batches = 0

        for run in range(2):
            pipeline.reset()
            iterator = iter(pipeline)
            for i, batch in enumerate(iterator):
                total_batches += 1
                if i >= 5:  # Process only 5 batches per run
                    break

        return {"total_batches": total_batches, "caching_enabled": enable_caching}

    result = benchmark(run_benchmark)
    assert result["total_batches"] > 0


def test_branching_performance(benchmark, benchmark_text_data):
    """Benchmark conditional branching performance."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_text_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        # Define branch condition based on sequence length
        def length_condition(x):
            avg_length = jnp.mean(x["length"])
            return avg_length > 50

        # Define different processing paths (deterministic operators) - Element API
        det_config = ElementOperatorConfig(stochastic=False)
        short_path = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={**e.data, "tokens": e.data["tokens"] * 2}
            ),  # Simple doubling
        )

        long_path = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={**e.data, "tokens": jnp.clip(e.data["tokens"], 0, 100)}
            ),  # Clipping
        )

        # Build pipeline with branching
        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs)
        pipeline = pipeline.add(source).batch(batch_size=32)
        pipeline = pipeline.branch(length_condition, long_path, short_path)

        # Process batches
        iterator = iter(pipeline)
        num_batches = 0
        branch_counts = {"short": 0, "long": 0}

        for batch in iterator:
            num_batches += 1
            # Check which branch was taken based on output
            if jnp.all(batch["tokens"] <= 100):
                branch_counts["long"] += 1
            else:
                branch_counts["short"] += 1

            if num_batches >= 10:
                break

        return {"batches": num_batches, "branch_counts": branch_counts}

    result = benchmark(run_benchmark)
    assert result["batches"] > 0
    assert sum(result["branch_counts"].values()) == result["batches"]


@pytest.mark.parametrize("jit_compile", [False, True])
def test_jit_compilation_impact(benchmark, benchmark_image_data, jit_compile):
    """Benchmark JIT compilation performance impact."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:1000]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        # Create pipeline with or without JIT
        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs, jit_compile=jit_compile)
        pipeline = pipeline.add(source).batch(batch_size=32)

        # Add compute-intensive transforms that benefit from JIT - Element API
        det_config = ElementOperatorConfig(stochastic=False)
        jit_transform = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={
                    "image": jnp.tanh(jnp.sin(e.data["image"]) + jnp.cos(e.data["image"])),
                    "label": e.data["label"],
                }
            ),
        )
        pipeline = pipeline.operate(jit_transform)

        # Process batches
        iterator = iter(pipeline)
        num_batches = 0

        for batch in iterator:
            num_batches += 1
            if num_batches >= 10:
                break

        return {"batches": num_batches, "jit_enabled": jit_compile}

    # Warmup for JIT
    if jit_compile:
        _ = run_benchmark()

    result = benchmark(run_benchmark)
    assert result["batches"] > 0


def test_complex_dag_topology_performance(benchmark, benchmark_image_data):
    """Benchmark complex DAG topology with multiple parallel and sequential stages."""

    # Reduce dataset size for faster testing
    small_dataset = benchmark_image_data[:500]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs)
        pipeline = pipeline.add(source).batch(batch_size=16)

        # Stage 1: Parallel preprocessing - Element API
        det_config = ElementOperatorConfig(stochastic=False)
        preprocess_transforms = [
            ElementOperator(
                det_config,
                fn=lambda e, k: e.replace(
                    data={"image": e.data["image"] / 255.0, "label": e.data["label"]}
                ),
            ),
            ElementOperator(
                det_config,
                fn=lambda e, k: e.replace(
                    data={"image": (e.data["image"] - 0.5) * 2, "label": e.data["label"]}
                ),
            ),
        ]
        pipeline = pipeline.parallel(preprocess_transforms)
        pipeline = pipeline.merge(strategy="mean", axis=0)

        # Stage 2: Sequential processing - Element API
        clip_transform = ElementOperator(
            det_config,
            fn=lambda e, k: e.replace(
                data={"image": jnp.clip(e.data["image"], -1, 1), "label": e.data["label"]}
            ),
        )
        pipeline = pipeline.operate(clip_transform)

        # Stage 3: Another parallel stage - Element API
        feature_transforms = [
            ElementOperator(
                det_config,
                fn=lambda e, k: e.replace(
                    data={"image": e.data["image"] ** 2, "label": e.data["label"]}
                ),
            ),
            ElementOperator(
                det_config,
                fn=lambda e, k: e.replace(
                    data={"image": jnp.abs(e.data["image"]), "label": e.data["label"]}
                ),
            ),
            ElementOperator(
                det_config,
                fn=lambda e, k: e.replace(
                    data={"image": jnp.sqrt(jnp.abs(e.data["image"])), "label": e.data["label"]}
                ),
            ),
        ]
        pipeline = pipeline.parallel(feature_transforms)
        pipeline = pipeline.merge(strategy="concat", axis=-1)

        # Process batches
        iterator = iter(pipeline)
        num_batches = 0
        output_shapes = []

        for batch in iterator:
            num_batches += 1
            output_shapes.append(batch["image"].shape)
            if num_batches >= 5:
                break

        return {"batches": num_batches, "output_shapes": output_shapes}

    result = benchmark(run_benchmark)
    assert result["batches"] > 0
    assert all(len(shape) == 4 for shape in result["output_shapes"])  # All outputs should be 4D


def test_memory_efficiency(benchmark_image_data):
    """Test memory efficiency with large batches and multiple transforms."""

    # Test that pipeline doesn't create unnecessary copies
    rngs = nnx.Rngs(default=42)
    source = MemorySource(MemorySourceConfig(), benchmark_image_data[:100], rngs=rngs)

    rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
    pipeline = DAGExecutor(rngs=rngs, enable_caching=False)
    pipeline = pipeline.add(source).batch(batch_size=10)

    # Add multiple identity transforms (should not increase memory) - Element API
    det_config = ElementOperatorConfig(stochastic=False)
    for i in range(10):
        identity_op = ElementOperator(det_config, fn=lambda e, k: e)
        pipeline = pipeline.operate(identity_op)

    # Process one batch
    iterator = iter(pipeline)
    batch = next(iterator)

    # Verify batch is processed correctly
    assert batch is not None
    assert "image" in batch
    assert batch["image"].shape[0] == 10


def test_reset_performance(benchmark, benchmark_image_data):
    """Benchmark pipeline reset performance."""

    small_dataset = benchmark_image_data[:500]

    def run_benchmark():
        rngs = nnx.Rngs(default=42)
        source = MemorySource(MemorySourceConfig(), small_dataset, rngs=rngs)

        rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
        pipeline = DAGExecutor(rngs=rngs)
        pipeline = pipeline.add(source).batch(batch_size=32)
        det_config = ElementOperatorConfig(stochastic=False)
        identity_op = ElementOperator(det_config, fn=lambda e, k: e)
        pipeline = pipeline.operate(identity_op)

        reset_count = 0
        total_batches = 0

        for epoch in range(3):
            pipeline.reset()
            reset_count += 1

            iterator = iter(pipeline)
            for i, batch in enumerate(iterator):
                total_batches += 1
                if i >= 5:  # Process only 5 batches per epoch
                    break

        return {"resets": reset_count, "total_batches": total_batches}

    result = benchmark(run_benchmark)
    assert result["resets"] == 3
    assert result["total_batches"] == 18  # 3 epochs * 6 batches


def test_dag_executor_state_tracking(benchmark_image_data):
    """Test DAGExecutor state tracking during iteration."""

    rngs = nnx.Rngs(default=42)
    source = MemorySource(MemorySourceConfig(), benchmark_image_data[:100], rngs=rngs)

    rngs = nnx.Rngs(default=42, augment=43, shuffle=44)
    pipeline = DAGExecutor(rngs=rngs, name="test_pipeline")
    pipeline = pipeline.add(source).batch(batch_size=10)

    # Verify initial state - now plain integers, not Variables
    assert pipeline._iteration_count == 0
    assert pipeline._epoch_count == 0

    # Process some batches
    iterator = iter(pipeline)
    for i, batch in enumerate(iterator):
        if i >= 3:
            break

    # Check iteration count increased
    assert pipeline._iteration_count > 0

    # Reset and check state
    pipeline.reset()
    assert pipeline._iteration_count == 0
    # After reset, epoch count should be back to 0
    assert pipeline._epoch_count == 0


if __name__ == "__main__":
    # Allow running directly for manual testing
    pytest.main(["-xvs", __file__])
