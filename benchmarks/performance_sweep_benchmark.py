"""Performance Sweep Benchmark for Datarax.

Systematically varies pipeline parameters (batch size, operator count,
data size, image resolution) to build a complete performance profile.
Generates summary tables and visualization plots.

Usage:
    python benchmarks/performance_sweep_benchmark.py
    python benchmarks/performance_sweep_benchmark.py --quick
    python benchmarks/performance_sweep_benchmark.py --output results.json
"""

import argparse
import json
import time
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

matplotlib.use("Agg")

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
)
from datarax.sources import MemorySource, MemorySourceConfig


# ── Data generation ──────────────────────────────────────────────────────────


def generate_data(
    num_samples: int = 5000, image_size: int = 32, seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate synthetic image data at a given resolution."""
    rng = np.random.RandomState(seed)
    return {
        "image": rng.rand(num_samples, image_size, image_size, 3).astype(np.float32),
        "label": rng.randint(0, 10, (num_samples,)).astype(np.int32),
    }


# ── Pipeline factories ──────────────────────────────────────────────────────


def normalize(element, key=None):
    """Simple normalization."""
    del key
    return element.update_data({"image": element.data["image"] / 255.0})


def create_pipeline(
    data: dict,
    batch_size: int = 32,
    num_operators: int = 1,
    seed: int = 0,
):
    """Create pipeline with N chained operators.

    num_operators controls complexity:
        1 = normalize only
        2 = normalize + brightness
        3 = normalize + brightness + contrast
        4 = normalize + brightness + contrast + noise
    """
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(seed))
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
    )

    pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(normalizer))

    if num_operators >= 2:
        brightness = BrightnessOperator(
            BrightnessOperatorConfig(
                field_key="image",
                brightness_range=(-0.1, 0.1),
                stochastic=True,
                stream_name="brightness",
            ),
            rngs=nnx.Rngs(brightness=seed + 100),
        )
        pipeline = pipeline.add(OperatorNode(brightness))

    if num_operators >= 3:
        contrast = ContrastOperator(
            ContrastOperatorConfig(
                field_key="image",
                contrast_range=(0.9, 1.1),
                stochastic=True,
                stream_name="contrast",
            ),
            rngs=nnx.Rngs(contrast=seed + 200),
        )
        pipeline = pipeline.add(OperatorNode(contrast))

    if num_operators >= 4:
        noise = NoiseOperator(
            NoiseOperatorConfig(
                field_key="image",
                mode="gaussian",
                noise_std=0.05,
                stochastic=True,
                stream_name="noise",
            ),
            rngs=nnx.Rngs(noise=seed + 300),
        )
        pipeline = pipeline.add(OperatorNode(noise))

    return pipeline


# ── Core measurement ─────────────────────────────────────────────────────────


def measure(
    data: dict,
    batch_size: int = 32,
    num_operators: int = 1,
    warmup_batches: int = 5,
) -> dict:
    """Measure throughput and latency for a single configuration."""
    pipeline = create_pipeline(data, batch_size, num_operators)

    # Warmup
    for i, batch in enumerate(pipeline):
        _ = batch["image"].block_until_ready()
        if i >= warmup_batches:
            break

    # Timed run
    pipeline = create_pipeline(data, batch_size, num_operators)
    total_samples = 0
    batch_times = []

    for batch in pipeline:
        t0 = time.perf_counter()
        _ = batch["image"].block_until_ready()
        batch_times.append(time.perf_counter() - t0)
        total_samples += batch["image"].shape[0]

    elapsed = sum(batch_times)
    return {
        "batch_size": batch_size,
        "num_operators": num_operators,
        "num_samples": len(data["image"]),
        "image_size": data["image"].shape[1],
        "total_batches": len(batch_times),
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 4),
        "samples_per_second": round(total_samples / elapsed, 2),
        "mean_batch_ms": round(np.mean(batch_times) * 1000, 3),
        "p50_batch_ms": round(np.percentile(batch_times, 50) * 1000, 3),
        "p95_batch_ms": round(np.percentile(batch_times, 95) * 1000, 3),
    }


# ── Sweep functions ──────────────────────────────────────────────────────────


def sweep_batch_sizes(data: dict, batch_sizes: list[int]) -> list[dict]:
    """Sweep batch sizes with a fixed operator count."""
    results = []
    for bs in batch_sizes:
        print(f"  batch_size={bs:>4d}", end="  ", flush=True)
        r = measure(data, batch_size=bs)
        print(f"{r['samples_per_second']:>10.0f} samples/s  p95={r['p95_batch_ms']:.1f}ms")
        results.append(r)
    return results


def sweep_operators(data: dict, batch_size: int = 64) -> list[dict]:
    """Sweep operator count from 1 to 4."""
    labels = ["normalize", "+brightness", "+contrast", "+noise"]
    results = []
    for n_ops in range(1, 5):
        print(f"  {n_ops} ops ({labels[n_ops - 1]:>12s})", end="  ", flush=True)
        r = measure(data, batch_size=batch_size, num_operators=n_ops)
        print(f"{r['samples_per_second']:>10.0f} samples/s  p95={r['p95_batch_ms']:.1f}ms")
        results.append(r)
    return results


def sweep_data_sizes(image_size: int = 32, sizes: list[int] | None = None) -> list[dict]:
    """Sweep dataset sizes at a fixed image resolution."""
    if sizes is None:
        sizes = [1000, 5000, 10000, 25000]
    results = []
    for n in sizes:
        print(f"  {n:>6d} samples", end="  ", flush=True)
        data = generate_data(num_samples=n, image_size=image_size)
        r = measure(data, batch_size=64)
        print(f"{r['samples_per_second']:>10.0f} samples/s")
        results.append(r)
    return results


def sweep_image_resolutions(
    num_samples: int = 5000, resolutions: list[int] | None = None
) -> list[dict]:
    """Sweep image resolutions at a fixed dataset size."""
    if resolutions is None:
        resolutions = [16, 32, 64, 128]
    results = []
    for res in resolutions:
        print(f"  {res}x{res}", end="  ", flush=True)
        data = generate_data(num_samples=num_samples, image_size=res)
        r = measure(data, batch_size=64)
        print(f"{r['samples_per_second']:>10.0f} samples/s")
        results.append(r)
    return results


# ── Visualization ────────────────────────────────────────────────────────────


def plot_results(results: dict, output_dir: Path) -> None:
    """Generate and save performance sweep visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Batch size sweep: throughput + latency percentiles
    if "batch_size_sweep" in results:
        sweep = results["batch_size_sweep"]
        bs = [r["batch_size"] for r in sweep]
        tp = [r["samples_per_second"] for r in sweep]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(bs, tp, "o-", color="steelblue", linewidth=2, markersize=8)
        axes[0].set_xlabel("Batch Size")
        axes[0].set_ylabel("Throughput (samples/s)")
        axes[0].set_title("Throughput vs Batch Size")
        axes[0].set_xscale("log", base=2)
        axes[0].grid(True, alpha=0.3)
        for x, y in zip(bs, tp):
            axes[0].annotate(
                f"{y:.0f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        p50 = [r["p50_batch_ms"] for r in sweep]
        p95 = [r["p95_batch_ms"] for r in sweep]
        x_pos = np.arange(len(bs))
        width = 0.35

        axes[1].bar(x_pos - width / 2, p50, width, label="p50", color="#4fc3f7")
        axes[1].bar(x_pos + width / 2, p95, width, label="p95", color="#ff8a65")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([str(b) for b in bs])
        axes[1].set_xlabel("Batch Size")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Batch Latency Percentiles")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = output_dir / "sweep-batch-size.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    # 2. Operator count sweep: throughput + latency
    if "operator_sweep" in results:
        sweep = results["operator_sweep"]
        labels = ["Normalize", "+Brightness", "+Contrast", "+Noise"]
        tp = [r["samples_per_second"] for r in sweep]
        latency = [r["mean_batch_ms"] for r in sweep]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        bars = axes[0].bar(labels[: len(tp)], tp, color="steelblue", edgecolor="white")
        axes[0].set_ylabel("Throughput (samples/s)")
        axes[0].set_title("Throughput vs Operator Count")
        axes[0].grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, tp):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + max(tp) * 0.02,
                f"{val:.0f}",
                ha="center",
                fontsize=9,
            )

        bars = axes[1].bar(labels[: len(latency)], latency, color="#ff8a65", edgecolor="white")
        axes[1].set_ylabel("Mean Batch Latency (ms)")
        axes[1].set_title("Latency vs Operator Count")
        axes[1].grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, latency):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                val + max(latency) * 0.02,
                f"{val:.2f}ms",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        path = output_dir / "sweep-operator-count.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    # 3. Data size sweep: throughput scaling
    if "data_size_sweep" in results:
        sweep = results["data_size_sweep"]
        sizes = [r["num_samples"] for r in sweep]
        tp = [r["samples_per_second"] for r in sweep]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sizes, tp, "o-", color="steelblue", linewidth=2, markersize=8)
        ax.set_xlabel("Dataset Size (samples)")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title("Throughput vs Dataset Size")
        ax.grid(True, alpha=0.3)
        for x, y in zip(sizes, tp):
            ax.annotate(
                f"{y:.0f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )
        plt.tight_layout()
        path = output_dir / "sweep-data-size.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    # 4. Image resolution sweep: throughput
    if "resolution_sweep" in results:
        sweep = results["resolution_sweep"]
        res_labels = [f"{r['image_size']}x{r['image_size']}" for r in sweep]
        tp = [r["samples_per_second"] for r in sweep]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(res_labels, tp, color="steelblue", edgecolor="white")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title("Throughput vs Image Resolution")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, tp):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(tp) * 0.02,
                f"{val:.0f}",
                ha="center",
                fontsize=9,
            )
        plt.tight_layout()
        path = output_dir / "sweep-resolution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Performance Sweep Benchmark")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Run quick subset")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Performance Sweep Benchmark")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Samples: {args.samples}")
    print()

    data = generate_data(num_samples=args.samples)

    # Batch size sweep
    print("1. Batch Size Sweep")
    print("-" * 60)
    batch_sizes = [16, 32, 64, 128] if args.quick else [8, 16, 32, 64, 128, 256, 512]
    batch_results = sweep_batch_sizes(data, batch_sizes)

    # Operator count sweep
    print()
    print("2. Operator Count Sweep")
    print("-" * 60)
    op_results = sweep_operators(data)

    # Data size sweep
    print()
    print("3. Data Size Sweep")
    print("-" * 60)
    size_list = [1000, 5000] if args.quick else [1000, 5000, 10000, 25000]
    size_results = sweep_data_sizes(sizes=size_list)

    # Resolution sweep
    print()
    print("4. Image Resolution Sweep")
    print("-" * 60)
    res_list = [16, 32, 64] if args.quick else [16, 32, 64, 128]
    res_results = sweep_image_resolutions(num_samples=args.samples, resolutions=res_list)

    # Collect
    all_results = {
        "hardware": {
            "backend": str(jax.default_backend()),
            "devices": [str(d) for d in jax.devices()],
        },
        "config": {"base_samples": args.samples, "quick": args.quick},
        "batch_size_sweep": batch_results,
        "operator_sweep": op_results,
        "data_size_sweep": size_results,
        "resolution_sweep": res_results,
    }

    # Save results
    output_path = Path(args.output or "temp/performance_sweep_results.json")
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
