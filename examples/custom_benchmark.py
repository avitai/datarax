"""Custom benchmark script for Datarax.

Demonstrates how to use Datarax's benchmark utilities directly
in Python code, using TimingCollector and rank_table.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.analysis import rank_table
from calibrax.core import Metric, MetricDef, MetricDirection, Point, Run
from calibrax.profiling import TimingCollector, TimingSample
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


def _sync_fn(_: object = None) -> None:
    """JAX device sync for accurate GPU timing."""
    jnp.array(0.0).block_until_ready()


def generate_sample_image_data(num_samples: int = 1000, image_size: int = 32) -> dict:
    """Generate sample image data for benchmarking."""
    rng = np.random.RandomState(42)
    return {
        "image": rng.rand(num_samples, image_size, image_size, 3).astype(np.float32),
        "label": rng.randint(0, 10, (num_samples,)).astype(np.int32),
    }


def normalize_transform(element, key=None):
    """Normalize image values to [0, 1] range."""
    return element.update_data({"image": element.data["image"] / 255.0})


def random_flip_transform(element, key):
    """Apply random horizontal flip to image."""
    image = element.data["image"]
    flip = jax.random.bernoulli(key, 0.5)
    flipped_image = jnp.where(flip, jnp.fliplr(image), image)
    return element.update_data({"image": flipped_image})


def simulated_heavy_transform(element, key):
    """Simulate a compute-intensive operation."""
    image = element.data["image"]
    for _ in range(10):
        image = image * 0.99
    return element.update_data({"image": image})


def create_basic_pipeline(batch_size: int = 32) -> Pipeline:
    """Create a basic image pipeline with minimal processing."""
    data = generate_sample_image_data()
    source_config = MemorySourceConfig()
    source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))
    normalizer_config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(normalizer_config, fn=normalize_transform, rngs=nnx.Rngs(0))
    pipeline = Pipeline(source=source, stages=[normalizer], batch_size=batch_size, rngs=nnx.Rngs(0))
    return pipeline


def create_advanced_pipeline(batch_size: int = 32) -> Pipeline:
    """Create a more complex image pipeline with augmentation."""
    data = generate_sample_image_data()
    source_config = MemorySourceConfig()
    source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))
    normalizer_config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(normalizer_config, fn=normalize_transform, rngs=nnx.Rngs(0))
    flip_config = ElementOperatorConfig(stochastic=True, stream_name="flip")
    flip_augmenter = ElementOperator(flip_config, fn=random_flip_transform, rngs=nnx.Rngs(flip=42))
    heavy_config = ElementOperatorConfig(stochastic=True, stream_name="heavy")
    heavy_transform = ElementOperator(
        heavy_config, fn=simulated_heavy_transform, rngs=nnx.Rngs(heavy=43)
    )
    pipeline = Pipeline(
        source=source,
        stages=[normalizer, flip_augmenter, heavy_transform],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )
    return pipeline


def measure_pipeline(pipeline, num_batches: int = 50, warmup: int = 5) -> TimingSample:
    """Measure pipeline throughput with warmup."""
    for i, _ in enumerate(pipeline):
        if i >= warmup - 1:
            break

    collector = TimingCollector(sync_fn=_sync_fn)
    return collector.measure_iteration(iter(pipeline), num_batches=num_batches)


def run_pipeline_benchmark():
    """Run a basic pipeline benchmark."""
    print("\n=== Running Pipeline Benchmark ===")
    pipeline = create_basic_pipeline(batch_size=32)
    sample = measure_pipeline(pipeline, num_batches=50)

    bps = sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
    eps = sample.num_elements / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
    print(f"  Wall clock:   {sample.wall_clock_sec:.4f} s")
    print(f"  Batches/sec:  {bps:.2f}")
    print(f"  Elements/sec: {eps:.2f}")
    return sample


def run_comparison_benchmark():
    """Run a comparison benchmark between different pipelines."""
    print("\n=== Running Comparison Benchmark ===")

    configs = {
        "basic": create_basic_pipeline(batch_size=32),
        "advanced": create_advanced_pipeline(batch_size=32),
    }

    points: list[Point] = []
    for name, pipeline in configs.items():
        sample = measure_pipeline(pipeline, num_batches=30)
        throughput = (
            sample.num_elements / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0.0
        )
        points.append(
            Point(
                name=f"custom/{name}",
                scenario="custom",
                tags={"framework": "Datarax", "variant": name},
                metrics={"throughput": Metric(value=throughput)},
            )
        )

    run = Run(
        points=tuple(points),
        metric_defs={
            "throughput": MetricDef(
                name="throughput",
                unit="elem/s",
                direction=MetricDirection.HIGHER,
            )
        },
    )

    rankings = rank_table(run, "throughput", group_by_tag="variant")
    print("\n  Ranking by throughput:")
    for row in rankings:
        print(f"  {row.rank}. {row.label}: {row.value:.2f} elem/s")

    return run


def run_batch_size_benchmark():
    """Run a batch size benchmark."""
    print("\n=== Running Batch Size Benchmark ===")

    batch_sizes = [8, 16, 32, 64, 128]
    print(f"{'Batch Size':>10} | {'Elements/sec':>15} | {'Wall Clock (s)':>15}")
    print("-" * 50)

    for batch_size in batch_sizes:
        pipeline = create_basic_pipeline(batch_size=batch_size)
        sample = measure_pipeline(pipeline, num_batches=30)
        eps = sample.num_elements / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        print(f"{batch_size:>10} | {eps:>15.2f} | {sample.wall_clock_sec:>15.4f}")


def main():
    """Run all benchmarks."""
    start_time = time.time()

    run_pipeline_benchmark()
    run_comparison_benchmark()
    run_batch_size_benchmark()

    print(f"\nAll benchmarks completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
