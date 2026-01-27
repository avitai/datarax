"""MNIST Pipeline Benchmark for Datarax.

Measures throughput, latency, and memory characteristics of an MNIST
classification pipeline with varying configurations. Results are saved
to JSON for reproducible comparison across hardware setups.

Usage:
    python benchmarks/mnist_pipeline_benchmark.py
    python benchmarks/mnist_pipeline_benchmark.py --epochs 3 --output results.json
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

matplotlib.use("Agg")

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig


# ── Data generation ──────────────────────────────────────────────────────────

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def generate_mnist_like_data(
    num_samples: int = 10000, seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate synthetic MNIST-shaped data for benchmarking."""
    rng = np.random.RandomState(seed)
    return {
        "image": rng.randint(0, 256, (num_samples, 28, 28, 1)).astype(np.float32),
        "label": rng.randint(0, 10, (num_samples,)).astype(np.int32),
    }


# ── Pipeline factories ──────────────────────────────────────────────────────


def normalize_mnist(element, key=None):
    """Standard MNIST normalization."""
    del key
    image = element.data["image"] / 255.0
    image = (image - MNIST_MEAN) / MNIST_STD
    return element.update_data({"image": image})


def create_pipeline(
    data: dict, batch_size: int = 32, with_augmentation: bool = False, seed: int = 0
):
    """Create an MNIST pipeline with optional augmentation."""
    source = MemorySource(
        MemorySourceConfig(), data=data, rngs=nnx.Rngs(seed)
    )

    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=normalize_mnist,
        rngs=nnx.Rngs(0),
    )

    pipeline = from_source(source, batch_size=batch_size).add(
        OperatorNode(normalizer)
    )

    if with_augmentation:

        def add_noise(element, key):
            image = element.data["image"]
            noise = jax.random.normal(key, image.shape) * 0.1
            return element.update_data({"image": image + noise})

        noise_op = ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="noise"),
            fn=add_noise,
            rngs=nnx.Rngs(noise=seed + 100),
        )
        pipeline = pipeline.add(OperatorNode(noise_op))

    return pipeline


# ── Benchmark routines ───────────────────────────────────────────────────────


def benchmark_throughput(
    data: dict,
    batch_size: int,
    num_epochs: int = 1,
    with_augmentation: bool = False,
    warmup_batches: int = 5,
) -> dict:
    """Measure pipeline throughput for a given configuration."""
    pipeline = create_pipeline(data, batch_size, with_augmentation)

    # Warmup
    for i, batch in enumerate(pipeline):
        _ = batch["image"].block_until_ready()
        if i >= warmup_batches:
            break

    # Timed run
    total_samples = 0
    batch_times = []

    for epoch in range(num_epochs):
        pipeline = create_pipeline(data, batch_size, with_augmentation, seed=epoch)
        for batch in pipeline:
            t0 = time.perf_counter()
            _ = batch["image"].block_until_ready()
            batch_times.append(time.perf_counter() - t0)
            total_samples += batch["image"].shape[0]

    elapsed = sum(batch_times)
    return {
        "batch_size": batch_size,
        "augmented": with_augmentation,
        "num_epochs": num_epochs,
        "total_samples": total_samples,
        "total_batches": len(batch_times),
        "elapsed_seconds": round(elapsed, 4),
        "samples_per_second": round(total_samples / elapsed, 2),
        "batches_per_second": round(len(batch_times) / elapsed, 2),
        "mean_batch_ms": round(np.mean(batch_times) * 1000, 3),
        "p50_batch_ms": round(np.percentile(batch_times, 50) * 1000, 3),
        "p95_batch_ms": round(np.percentile(batch_times, 95) * 1000, 3),
        "p99_batch_ms": round(np.percentile(batch_times, 99) * 1000, 3),
    }


def run_batch_size_sweep(
    data: dict, batch_sizes: list[int], num_epochs: int = 1
) -> list[dict]:
    """Sweep over batch sizes and measure throughput for each."""
    results = []
    for bs in batch_sizes:
        print(f"  Batch size {bs:>4d}...", end=" ", flush=True)
        result = benchmark_throughput(data, bs, num_epochs)
        print(f"{result['samples_per_second']:>10.0f} samples/s")
        results.append(result)
    return results


def run_augmentation_comparison(
    data: dict, batch_size: int = 64, num_epochs: int = 1
) -> dict:
    """Compare throughput with and without augmentation."""
    print("  Without augmentation...", end=" ", flush=True)
    base = benchmark_throughput(data, batch_size, num_epochs, with_augmentation=False)
    print(f"{base['samples_per_second']:>10.0f} samples/s")

    print("  With augmentation...   ", end=" ", flush=True)
    aug = benchmark_throughput(data, batch_size, num_epochs, with_augmentation=True)
    print(f"{aug['samples_per_second']:>10.0f} samples/s")

    overhead_pct = (
        (base["samples_per_second"] - aug["samples_per_second"])
        / base["samples_per_second"]
        * 100
    )

    return {
        "baseline": base,
        "augmented": aug,
        "augmentation_overhead_pct": round(overhead_pct, 2),
    }


# ── Visualization ────────────────────────────────────────────────────────────


def plot_results(results: dict, output_dir: Path) -> None:
    """Generate and save MNIST benchmark visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Batch size sweep: throughput
    sweep = results["batch_size_sweep"]
    bs = [r["batch_size"] for r in sweep]
    tp = [r["samples_per_second"] for r in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(bs, tp, "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Throughput (samples/s)")
    axes[0].set_title("MNIST Throughput vs Batch Size")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)
    for x, y in zip(bs, tp):
        axes[0].annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8)

    # 2. Batch size sweep: latency percentiles
    p50 = [r["p50_batch_ms"] for r in sweep]
    p95 = [r["p95_batch_ms"] for r in sweep]
    p99 = [r["p99_batch_ms"] for r in sweep]
    x_pos = np.arange(len(bs))
    width = 0.25

    axes[1].bar(x_pos - width, p50, width, label="p50", color="#4fc3f7")
    axes[1].bar(x_pos, p95, width, label="p95", color="#ff8a65")
    axes[1].bar(x_pos + width, p99, width, label="p99", color="#e57373")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([str(b) for b in bs])
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("MNIST Batch Latency Percentiles")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "mnist-throughput-and-latency.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")

    # 3. Augmentation comparison: grouped bar
    aug = results["augmentation_comparison"]
    labels = ["Baseline", "With Augmentation"]
    throughputs = [aug["baseline"]["samples_per_second"],
                   aug["augmented"]["samples_per_second"]]
    latencies = [aug["baseline"]["mean_batch_ms"],
                 aug["augmented"]["mean_batch_ms"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars = axes[0].bar(labels, throughputs, color=["steelblue", "#ff8a65"])
    axes[0].set_ylabel("Throughput (samples/s)")
    axes[0].set_title("Augmentation Impact on Throughput")
    for bar, val in zip(bars, throughputs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + max(throughputs) * 0.02,
                     f"{val:.0f}", ha="center", fontsize=10)
    overhead = aug["augmentation_overhead_pct"]
    axes[0].annotate(f"{overhead:.1f}% overhead", xy=(0.5, 0.95),
                     xycoords="axes fraction", ha="center", fontsize=10,
                     color="red", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    bars = axes[1].bar(labels, latencies, color=["steelblue", "#ff8a65"])
    axes[1].set_ylabel("Mean Batch Latency (ms)")
    axes[1].set_title("Augmentation Impact on Latency")
    for bar, val in zip(bars, latencies):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(latencies) * 0.02,
                     f"{val:.2f}ms", ha="center", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "mnist-augmentation-comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MNIST Pipeline Benchmark")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    print("MNIST Pipeline Benchmark")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Samples: {args.samples}")
    print()

    data = generate_mnist_like_data(num_samples=args.samples)

    # Batch size sweep
    print("Batch Size Sweep")
    print("-" * 60)
    batch_sizes = [8, 16, 32, 64, 128, 256]
    sweep_results = run_batch_size_sweep(data, batch_sizes, args.epochs)

    # Augmentation comparison
    print()
    print("Augmentation Overhead")
    print("-" * 60)
    aug_results = run_augmentation_comparison(data, batch_size=64, num_epochs=args.epochs)
    print(f"  Overhead: {aug_results['augmentation_overhead_pct']:.1f}%")

    # Collect results
    all_results = {
        "hardware": {
            "backend": str(jax.default_backend()),
            "devices": [str(d) for d in jax.devices()],
        },
        "config": {
            "num_samples": args.samples,
            "num_epochs": args.epochs,
        },
        "batch_size_sweep": sweep_results,
        "augmentation_comparison": aug_results,
    }

    # Save results
    output_path = Path(args.output or "temp/mnist_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate plots in same directory as JSON output
    print()
    print("Generating plots...")
    plot_results(all_results, output_path.parent)


if __name__ == "__main__":
    main()
