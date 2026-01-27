#!/usr/bin/env python3
"""Performance Sweep Benchmark.

Comprehensive benchmark sweeping multiple parameters to identify optimal
pipeline configurations. Tests batch sizes, operator combinations, and
memory usage patterns.

Usage:
    python performance_sweep_benchmark.py [--quick] [--output results.json]
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
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
    RotationOperator,
    RotationOperatorConfig,
)
from datarax.sources import MemorySource, MemorySourceConfig


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def preprocess(element, key=None):  # noqa: ARG001
    """Simple normalization."""
    del key
    image = element.data["image"] / 255.0
    return element.update_data({"image": image})


def create_operators(config_name: str, seed: int = 42) -> list:
    """Create operator list based on configuration."""
    operators = []

    if config_name == "baseline":
        pass  # No augmentation
    elif config_name == "brightness":
        operators.append(
            BrightnessOperator(
                BrightnessOperatorConfig(
                    field_key="image",
                    brightness_range=(-0.2, 0.2),
                    stochastic=True,
                    stream_name="brightness",
                ),
                rngs=nnx.Rngs(brightness=seed),
            )
        )
    elif config_name == "contrast":
        operators.append(
            ContrastOperator(
                ContrastOperatorConfig(
                    field_key="image",
                    contrast_range=(0.8, 1.2),
                    stochastic=True,
                    stream_name="contrast",
                ),
                rngs=nnx.Rngs(contrast=seed),
            )
        )
    elif config_name == "noise":
        operators.append(
            NoiseOperator(
                NoiseOperatorConfig(
                    field_key="image",
                    mode="gaussian",
                    noise_std=0.1,
                    stochastic=True,
                    stream_name="noise",
                ),
                rngs=nnx.Rngs(noise=seed),
            )
        )
    elif config_name == "rotation":
        operators.append(
            RotationOperator(
                RotationOperatorConfig(
                    field_key="image",
                    angle_range=(-15, 15),
                ),
                rngs=nnx.Rngs(seed),
            )
        )
    elif config_name == "light_augment":
        operators.extend(
            [
                BrightnessOperator(
                    BrightnessOperatorConfig(
                        field_key="image",
                        brightness_range=(-0.1, 0.1),
                        stochastic=True,
                        stream_name="brightness",
                    ),
                    rngs=nnx.Rngs(brightness=seed),
                ),
                ContrastOperator(
                    ContrastOperatorConfig(
                        field_key="image",
                        contrast_range=(0.9, 1.1),
                        stochastic=True,
                        stream_name="contrast",
                    ),
                    rngs=nnx.Rngs(contrast=seed + 1),
                ),
            ]
        )
    elif config_name == "full_augment":
        operators.extend(
            [
                BrightnessOperator(
                    BrightnessOperatorConfig(
                        field_key="image",
                        brightness_range=(-0.15, 0.15),
                        stochastic=True,
                        stream_name="brightness",
                    ),
                    rngs=nnx.Rngs(brightness=seed),
                ),
                ContrastOperator(
                    ContrastOperatorConfig(
                        field_key="image",
                        contrast_range=(0.85, 1.15),
                        stochastic=True,
                        stream_name="contrast",
                    ),
                    rngs=nnx.Rngs(contrast=seed + 1),
                ),
                RotationOperator(
                    RotationOperatorConfig(
                        field_key="image",
                        angle_range=(-10, 10),
                    ),
                    rngs=nnx.Rngs(seed + 2),
                ),
                NoiseOperator(
                    NoiseOperatorConfig(
                        field_key="image",
                        mode="gaussian",
                        noise_std=0.05,
                        stochastic=True,
                        stream_name="noise",
                    ),
                    rngs=nnx.Rngs(noise=seed + 3),
                ),
            ]
        )

    return operators


def create_pipeline(data: dict, batch_size: int, operator_config: str, seed: int = 42):
    """Create pipeline with specified operator configuration."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess,
        rngs=nnx.Rngs(0),
    )

    pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(prep))

    for op in create_operators(operator_config, seed):
        pipeline = pipeline.add(OperatorNode(op))

    return pipeline


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def benchmark_configuration(
    data: dict,
    batch_size: int,
    operator_config: str,
    num_batches: int,
    warmup_batches: int = 5,
) -> dict:
    """Benchmark a specific configuration."""
    pipeline = create_pipeline(data, batch_size, operator_config)

    # Warmup
    for i, batch in enumerate(pipeline):
        if i >= warmup_batches:
            break
        _ = batch["image"].block_until_ready()

    # Measurement
    pipeline = create_pipeline(data, batch_size, operator_config)
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

        if batches >= num_batches + warmup_batches:
            break

    total_time = time.time() - start_total
    measured_latencies = latencies[warmup_batches:]

    # Memory estimate
    image_mem = batch_size * np.prod(data["image"].shape[1:]) * 4
    total_mem = image_mem * 1.2  # Overhead factor

    return {
        "batch_size": batch_size,
        "operator_config": operator_config,
        "num_batches": batches - warmup_batches,
        "total_samples": samples,
        "total_time_s": total_time,
        "throughput_samples_per_s": samples / total_time if total_time > 0 else 0,
        "avg_latency_ms": np.mean(measured_latencies) * 1000 if measured_latencies else 0,
        "p50_latency_ms": np.percentile(measured_latencies, 50) * 1000 if measured_latencies else 0,
        "p95_latency_ms": np.percentile(measured_latencies, 95) * 1000 if measured_latencies else 0,
        "p99_latency_ms": np.percentile(measured_latencies, 99) * 1000 if measured_latencies else 0,
        "estimated_memory_mb": total_mem / 1e6,
    }


def run_sweep(
    batch_sizes: list[int],
    operator_configs: list[str],
    num_batches: int = 50,
    trials: int = 3,
    image_shape: tuple = (32, 32, 3),
) -> list[dict]:
    """Run full parameter sweep."""
    # Create test data
    num_samples = max(batch_sizes) * (num_batches + 20)
    data = {
        "image": np.random.randint(0, 256, (num_samples, *image_shape)).astype(np.float32),
        "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
    }

    results = []
    total_configs = len(batch_sizes) * len(operator_configs)
    config_num = 0

    for batch_size in batch_sizes:
        for op_config in operator_configs:
            config_num += 1
            print(f"  [{config_num}/{total_configs}] batch_size={batch_size}, ops={op_config}")

            trial_results = []
            for _ in range(trials):
                result = benchmark_configuration(data, batch_size, op_config, num_batches)
                trial_results.append(result)

            avg_result = {
                "batch_size": batch_size,
                "operator_config": op_config,
                "trials": trials,
                "throughput_mean": np.mean([r["throughput_samples_per_s"] for r in trial_results]),
                "throughput_std": np.std([r["throughput_samples_per_s"] for r in trial_results]),
                "latency_mean_ms": np.mean([r["avg_latency_ms"] for r in trial_results]),
                "latency_p95_ms": np.mean([r["p95_latency_ms"] for r in trial_results]),
                "memory_mb": np.mean([r["estimated_memory_mb"] for r in trial_results]),
            }
            results.append(avg_result)

            print(f"    -> {avg_result['throughput_mean']:.0f} samples/s")

    return results


# ============================================================================
# ANALYSIS
# ============================================================================


def analyze_results(results: list[dict]) -> dict:
    """Analyze benchmark results and provide recommendations."""
    # Find optimal configurations
    best_throughput = max(results, key=lambda x: x["throughput_mean"])
    best_latency = min(results, key=lambda x: x["latency_mean_ms"])

    # Compute overhead for each operator
    baseline_results = [r for r in results if r["operator_config"] == "baseline"]
    overhead_analysis = {}

    for op_config in set(r["operator_config"] for r in results):
        if op_config == "baseline":
            continue

        op_results = [r for r in results if r["operator_config"] == op_config]
        for op_result in op_results:
            bs = op_result["batch_size"]
            baseline = next((r for r in baseline_results if r["batch_size"] == bs), None)
            if baseline:
                overhead_pct = (
                    (baseline["throughput_mean"] - op_result["throughput_mean"])
                    / baseline["throughput_mean"]
                    * 100
                )
                if op_config not in overhead_analysis:
                    overhead_analysis[op_config] = []
                overhead_analysis[op_config].append(overhead_pct)

    avg_overhead = {k: np.mean(v) for k, v in overhead_analysis.items()}

    return {
        "best_throughput": {
            "batch_size": best_throughput["batch_size"],
            "operator_config": best_throughput["operator_config"],
            "throughput": best_throughput["throughput_mean"],
        },
        "best_latency": {
            "batch_size": best_latency["batch_size"],
            "operator_config": best_latency["operator_config"],
            "latency_ms": best_latency["latency_mean_ms"],
        },
        "operator_overhead_pct": avg_overhead,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Performance Sweep Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick sweep with fewer configs")
    parser.add_argument("--num-batches", type=int, default=50, help="Batches to measure")
    parser.add_argument("--trials", type=int, default=3, help="Trials per configuration")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    if args.quick:
        batch_sizes = [32, 64, 128]
        operator_configs = ["baseline", "light_augment", "full_augment"]
    else:
        batch_sizes = [16, 32, 64, 128, 256]
        operator_configs = [
            "baseline",
            "brightness",
            "contrast",
            "noise",
            "rotation",
            "light_augment",
            "full_augment",
        ]

    print("=" * 60)
    print("Performance Sweep Benchmark")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Operator configs: {operator_configs}")
    print(f"Batches per config: {args.num_batches}")
    print()

    results = run_sweep(batch_sizes, operator_configs, args.num_batches, args.trials)
    analysis = analyze_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    print(f"{'Batch':>8} {'Config':>15} {'Throughput':>15} {'Latency (ms)':>15}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['batch_size']:>8} {r['operator_config']:>15} "
            f"{r['throughput_mean']:>12.0f} ± {r['throughput_std']:>4.0f} "
            f"{r['latency_mean_ms']:>12.2f}"
        )

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"Best Throughput: {analysis['best_throughput']['throughput']:.0f} samples/s")
    print(
        f"  Config: batch_size={analysis['best_throughput']['batch_size']}, "
        f"ops={analysis['best_throughput']['operator_config']}"
    )
    print(f"\nBest Latency: {analysis['best_latency']['latency_ms']:.2f} ms")
    print(
        f"  Config: batch_size={analysis['best_latency']['batch_size']}, "
        f"ops={analysis['best_latency']['operator_config']}"
    )

    if analysis["operator_overhead_pct"]:
        print("\nOperator Overhead (vs baseline):")
        for op, overhead in sorted(analysis["operator_overhead_pct"].items(), key=lambda x: x[1]):
            print(f"  {op}: {overhead:.1f}%")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"performance_sweep_{timestamp}.json"

    output_data = {
        "benchmark": "performance_sweep",
        "timestamp": datetime.now().isoformat(),
        "jax_backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "config": {
            "batch_sizes": batch_sizes,
            "operator_configs": operator_configs,
            "num_batches": args.num_batches,
            "trials": args.trials,
        },
        "results": results,
        "analysis": analysis,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
