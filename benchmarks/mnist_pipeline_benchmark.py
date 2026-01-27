#!/usr/bin/env python3
"""MNIST Pipeline Benchmark.

Benchmarks Datarax pipeline performance for MNIST dataset with various
configurations. Generates throughput and latency metrics for optimization.

Usage:
    python mnist_pipeline_benchmark.py [--batch-sizes 32,64,128] [--num-batches 100]
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# GPU Memory Configuration
os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
)
from datarax.sources import TFDSEagerConfig, TFDSEagerSource


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_mnist(element, key=None):  # noqa: ARG001
    """Standard MNIST preprocessing."""
    del key
    image = element.data["image"]
    image = image.astype(jnp.float32) / 255.0
    if image.ndim == 2:
        image = image[..., None]
    image = (image - MNIST_MEAN) / MNIST_STD
    return element.update_data({"image": image})


def create_pipeline(batch_size: int, num_samples: int, with_augmentation: bool, seed: int = 42):
    """Create MNIST pipeline with optional augmentation."""
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="mnist",
            split=f"train[:{num_samples}]",
            shuffle=True,
            seed=seed,
        ),
        rngs=nnx.Rngs(seed),
    )

    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_mnist,
        rngs=nnx.Rngs(0),
    )

    pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(prep))

    if with_augmentation:
        brightness = BrightnessOperator(
            BrightnessOperatorConfig(
                field_key="image",
                brightness_range=(-0.1, 0.1),
                stochastic=True,
                stream_name="brightness",
            ),
            rngs=nnx.Rngs(brightness=seed + 100),
        )

        noise = NoiseOperator(
            NoiseOperatorConfig(
                field_key="image",
                mode="gaussian",
                noise_std=0.05,
                stochastic=True,
                stream_name="noise",
            ),
            rngs=nnx.Rngs(noise=seed + 200),
        )

        pipeline = pipeline.add(OperatorNode(brightness)).add(OperatorNode(noise))

    return pipeline


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def benchmark_throughput(
    batch_size: int,
    num_batches: int,
    with_augmentation: bool,
    warmup_batches: int = 5,
) -> dict:
    """Benchmark pipeline throughput."""
    num_samples = batch_size * (num_batches + warmup_batches + 10)
    pipeline = create_pipeline(batch_size, num_samples, with_augmentation)

    # Warmup
    for i, batch in enumerate(pipeline):
        if i >= warmup_batches:
            break
        _ = batch["image"].block_until_ready()

    # Measurement
    latencies = []
    samples = 0
    batches = 0

    start_total = time.time()
    for batch in pipeline:
        start_batch = time.time()
        _ = batch["image"].block_until_ready()
        latencies.append(time.time() - start_batch)

        samples += batch["image"].shape[0]
        batches += 1

        if batches >= num_batches:
            break

    total_time = time.time() - start_total

    return {
        "batch_size": batch_size,
        "num_batches": batches,
        "total_samples": samples,
        "total_time_s": total_time,
        "throughput_samples_per_s": samples / total_time if total_time > 0 else 0,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p50_latency_ms": np.percentile(latencies, 50) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        "with_augmentation": with_augmentation,
    }


def run_benchmark_suite(
    batch_sizes: list[int],
    num_batches: int = 100,
    trials: int = 3,
) -> list[dict]:
    """Run complete benchmark suite."""
    results = []

    for batch_size in batch_sizes:
        for with_aug in [False, True]:
            aug_label = "augmented" if with_aug else "baseline"
            print(f"  Benchmarking batch_size={batch_size}, {aug_label}...")

            trial_results = []
            for _ in range(trials):
                result = benchmark_throughput(batch_size, num_batches, with_aug)
                trial_results.append(result)

            # Aggregate across trials
            avg_result = {
                "batch_size": batch_size,
                "with_augmentation": with_aug,
                "num_batches": num_batches,
                "trials": trials,
                "throughput_mean": np.mean([r["throughput_samples_per_s"] for r in trial_results]),
                "throughput_std": np.std([r["throughput_samples_per_s"] for r in trial_results]),
                "latency_mean_ms": np.mean([r["avg_latency_ms"] for r in trial_results]),
                "latency_p95_ms": np.mean([r["p95_latency_ms"] for r in trial_results]),
            }
            results.append(avg_result)

            print(
                f"    Throughput: {avg_result['throughput_mean']:.0f} ± "
                f"{avg_result['throughput_std']:.0f} samples/s"
            )

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MNIST Pipeline Benchmark")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="32,64,128,256",
        help="Comma-separated batch sizes",
    )
    parser.add_argument("--num-batches", type=int, default=100, help="Batches to measure")
    parser.add_argument("--trials", type=int, default=3, help="Trials per configuration")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("MNIST Pipeline Benchmark")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Batches per config: {args.num_batches}")
    print(f"Trials: {args.trials}")
    print()

    results = run_benchmark_suite(batch_sizes, args.num_batches, args.trials)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Batch Size':>12} {'Augmented':>10} {'Throughput':>15} {'Latency (ms)':>15}")
    print("-" * 60)

    for r in results:
        aug = "Yes" if r["with_augmentation"] else "No"
        print(
            f"{r['batch_size']:>12} {aug:>10} "
            f"{r['throughput_mean']:>12.0f} ± {r['throughput_std']:>4.0f} "
            f"{r['latency_mean_ms']:>12.2f}"
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"mnist_benchmark_{timestamp}.json"

    output_data = {
        "benchmark": "mnist_pipeline",
        "timestamp": datetime.now().isoformat(),
        "jax_backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "config": {
            "batch_sizes": batch_sizes,
            "num_batches": args.num_batches,
            "trials": args.trials,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
