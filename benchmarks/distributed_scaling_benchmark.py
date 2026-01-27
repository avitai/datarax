#!/usr/bin/env python3
"""Distributed Scaling Benchmark.

Benchmarks Datarax pipeline performance across different device counts
to measure scaling efficiency for distributed data loading.

Usage:
    python distributed_scaling_benchmark.py [--batch-size 128] [--num-batches 100]
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================


def preprocess(element, key=None):  # noqa: ARG001
    """Simple normalization."""
    del key
    image = element.data["image"] / 255.0
    return element.update_data({"image": image})


def create_pipeline(data: dict, batch_size: int):
    """Create pipeline from memory data."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess,
        rngs=nnx.Rngs(0),
    )
    return from_source(source, batch_size=batch_size).add(OperatorNode(prep))


def create_sharding_spec(shape, mesh):
    """Create NamedSharding for batch-first data."""
    if mesh is None:
        return None
    ndim = len(shape)
    spec = ("data",) + (None,) * (ndim - 1)
    return NamedSharding(mesh, PartitionSpec(*spec))


def distribute_batch(batch, mesh):
    """Distribute batch across devices."""
    if mesh is None:
        return batch

    distributed = {}
    for key, array in batch.items():
        if hasattr(array, "shape"):
            sharding = create_sharding_spec(array.shape, mesh)
            distributed[key] = jax.device_put(array, sharding)
        else:
            distributed[key] = array
    return distributed


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def benchmark_device_count(
    data: dict,
    batch_size: int,
    num_devices: int,
    num_batches: int,
    warmup_batches: int = 5,
) -> dict:
    """Benchmark with specific device count."""
    devices = jax.devices()[:num_devices]

    if len(devices) < num_devices:
        return {
            "num_devices": num_devices,
            "available_devices": len(devices),
            "error": f"Only {len(devices)} devices available",
        }

    # Create mesh
    if num_devices > 1:
        mesh = Mesh(np.array(devices), axis_names=("data",))
    else:
        mesh = None

    # Benchmark
    pipeline = create_pipeline(data, batch_size)

    # Warmup
    warmup_count = 0
    if mesh is not None:
        with mesh:
            for batch in pipeline:
                sharded = distribute_batch(batch, mesh)
                _ = sharded["image"].block_until_ready()
                warmup_count += 1
                if warmup_count >= warmup_batches:
                    break
    else:
        for batch in pipeline:
            _ = batch["image"].block_until_ready()
            warmup_count += 1
            if warmup_count >= warmup_batches:
                break

    # Measurement
    pipeline = create_pipeline(data, batch_size)
    latencies = []
    samples = 0
    batches = 0

    start_total = time.time()

    if mesh is not None:
        with mesh:
            for batch in pipeline:
                start_batch = time.time()
                sharded = distribute_batch(batch, mesh)
                _ = sharded["image"].block_until_ready()
                latencies.append(time.time() - start_batch)

                samples += batch["image"].shape[0]
                batches += 1

                if batches >= num_batches:
                    break
    else:
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
        "num_devices": num_devices,
        "batch_size": batch_size,
        "samples_per_device": batch_size // max(num_devices, 1),
        "num_batches": batches,
        "total_samples": samples,
        "total_time_s": total_time,
        "throughput_samples_per_s": samples / total_time if total_time > 0 else 0,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
    }


def run_scaling_benchmark(
    batch_size: int = 128,
    num_batches: int = 100,
    trials: int = 3,
) -> list[dict]:
    """Run scaling benchmark across available devices."""
    # Create test data
    num_samples = batch_size * (num_batches + 20)
    data = {
        "image": np.random.randint(0, 256, (num_samples, 32, 32, 3)).astype(np.float32),
        "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
    }

    max_devices = len(jax.devices())
    device_counts = [1]
    # Add powers of 2 up to max devices
    d = 2
    while d <= max_devices:
        device_counts.append(d)
        d *= 2

    results = []

    for num_devices in device_counts:
        print(f"  Benchmarking {num_devices} device(s)...")

        trial_results = []
        for _ in range(trials):
            result = benchmark_device_count(data, batch_size, num_devices, num_batches)
            if "error" not in result:
                trial_results.append(result)
            else:
                print(f"    {result['error']}")
                break

        if trial_results:
            avg_result = {
                "num_devices": num_devices,
                "batch_size": batch_size,
                "samples_per_device": batch_size // max(num_devices, 1),
                "trials": trials,
                "throughput_mean": np.mean([r["throughput_samples_per_s"] for r in trial_results]),
                "throughput_std": np.std([r["throughput_samples_per_s"] for r in trial_results]),
                "latency_mean_ms": np.mean([r["avg_latency_ms"] for r in trial_results]),
            }

            # Compute scaling efficiency (relative to single device)
            if len(results) > 0 and results[0]["throughput_mean"] > 0:
                base_tp = results[0]["throughput_mean"]
                avg_result["scaling_efficiency"] = avg_result["throughput_mean"] / (
                    base_tp * num_devices
                )
            else:
                avg_result["scaling_efficiency"] = 1.0

            results.append(avg_result)

            print(
                f"    Throughput: {avg_result['throughput_mean']:.0f} samples/s "
                f"(efficiency: {avg_result['scaling_efficiency']:.1%})"
            )

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Distributed Scaling Benchmark")
    parser.add_argument("--batch-size", type=int, default=128, help="Total batch size")
    parser.add_argument("--num-batches", type=int, default=100, help="Batches to measure")
    parser.add_argument("--trials", type=int, default=3, help="Trials per configuration")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("Distributed Scaling Benchmark")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Total devices: {len(jax.devices())}")
    print(f"Device types: {[str(d.device_kind) for d in jax.devices()]}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per config: {args.num_batches}")
    print()

    results = run_scaling_benchmark(args.batch_size, args.num_batches, args.trials)

    # Summary
    print("\n" + "=" * 60)
    print("SCALING RESULTS")
    print("=" * 60)
    print(f"{'Devices':>8} {'Throughput':>15} {'Scaling Eff.':>15} {'Latency (ms)':>15}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['num_devices']:>8} "
            f"{r['throughput_mean']:>12.0f} ± {r['throughput_std']:>4.0f} "
            f"{r['scaling_efficiency']:>14.1%} "
            f"{r['latency_mean_ms']:>12.2f}"
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"distributed_scaling_{timestamp}.json"

    output_data = {
        "benchmark": "distributed_scaling",
        "timestamp": datetime.now().isoformat(),
        "jax_backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "config": {
            "batch_size": args.batch_size,
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
