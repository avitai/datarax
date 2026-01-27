"""Distributed Scaling Benchmark for Datarax.

Measures how pipeline throughput scales with the number of JAX devices.
On CPU, uses XLA_FLAGS to simulate multiple devices (one per core) so the
sharding code path is fully exercised. On CUDA/TPU, measures real scaling.

Usage:
    python benchmarks/distributed_scaling_benchmark.py
    python benchmarks/distributed_scaling_benchmark.py --platform cpu --num-devices 4
    python benchmarks/distributed_scaling_benchmark.py --platform cuda --samples 50000

Scaling Analysis & Design Notes
================================

This benchmark was designed after extensive investigation of JAX's distributed
execution model.  The notes below document findings, verify claims against
empirical data, and explain why certain design choices were made.

1. CPU Device Simulation — How It Works
   -------------------------------------
   `--xla_force_host_platform_device_count=N` creates N *logical* CPU devices
   backed by a **shared OS threadpool** (confirmed in XLA source: xla.proto).
   These devices DO execute in parallel via threads — they are NOT sequential.
   Evidence:
     - JAX GitHub Discussion #14762: user measured 1.8× speedup on 10 cores.
     - BlackJAX benchmarks: 1.47× wall-time speedup with pmap vs vmap.
     - XLA source (xla.proto): "All devices are backed by the same threadpool."

   However, scaling is sublinear because:
     a) All devices SHARE the threadpool → context-switching overhead.
     b) Intra-op parallelism (Eigen matmul using all cores on 1 device) is
        replaced by inter-op parallelism (N devices competing for cores).
        Ref: JAX Issue #6790, B. Nikolic blog on JAX multithreading.
     c) XLA docs explicitly warn: "Setting this to anything other than 1 can
        increase overhead from context switching."

2. Why the Workload Must Be Heavy
   --------------------------------
   The benchmark loop has this structure:

       for batch in pipeline:                    # (S) sequential
           images = device_put(batch, sharding)  # (S) shard overhead
           result = _workload(images)            # (P) parallel across N devices
           result.block_until_ready()            # (S) sync barrier

   Amdahl's law: Speedup = 1 / (s + (1-s)/N), where s = sequential fraction.

   Empirical verification (from 8-device CPU run, batch_size=128):
     - 1 device: 141 ms/batch  →  8 devices: 54 ms/batch  →  2.6× speedup
     - Solving Amdahl: s ≈ 0.27  (27% sequential fraction)
     - Predicted speedups: N=2 → 1.6× (actual: 1.6×), N=4 → 2.2× (actual: 2.2×)
     - The model fits the data at all device counts.

   The sequential fraction comes from:
     - Pipeline iteration (Python host loop, Datarax DAG executor)
     - `jax.device_put(..., sharding)` — splits + copies data
     - `block_until_ready()` — synchronization barrier after each batch

   With 32×32 images and 3 light iterations (original design), per-batch
   compute was ~2.5 ms while shard overhead was ~5-10 ms.  This made s > 0.7,
   causing throughput to DECREASE with more devices.  Increasing to 128×128
   images with 10 heavy iterations pushed per-batch compute to ~140 ms,
   making s ≈ 0.27 and enabling real speedups.

3. Batch Size and Cache Effects
   ------------------------------
   Empirical observation: per-sample throughput degrades at larger batch sizes.
     - bs=64 @ 1 device:  1261 samples/s (0.79 ms/sample)
     - bs=128 @ 1 device:  950 samples/s (1.05 ms/sample)
     - bs=256 @ 1 device:  841 samples/s (1.19 ms/sample)

   This is consistent with L3 cache pressure:
     - bs=64 × 128×128×3 × 4B = 12.6 MB  (fits in most L3 caches)
     - bs=128 → 25.2 MB  (exceeds typical L3, falls to DRAM bandwidth)
     - bs=256 → 50.3 MB  (well beyond cache)
   The sweet spot for per-device batch size is ~8-16 samples, where per-sample
   cache locality is preserved while dispatch overhead is amortized.

4. Maximizing Scaling Efficiency
   --------------------------------
   Near-linear scaling on simulated CPU is fundamentally limited by the shared
   threadpool and the intra-op / inter-op trade-off (see point 1b above).  The
   ~2.6× speedup at 8 devices (32.5% efficiency) is consistent with community
   benchmarks.

   For near-linear scaling, use REAL separate devices (multi-GPU/TPU), where:
     a) Each device has independent compute units and memory bandwidth.
     b) XLA's latency-hiding scheduler overlaps compute and communication.
        Flags: --xla_gpu_enable_latency_hiding_scheduler=true
     c) Async dispatch naturally pipelines: `train_step()` returns immediately,
        host loop continues loading the next batch while accelerator computes.
        Ref: JAX docs/async_dispatch.md
     d) `donate_argnums` enables buffer reuse, eliminating allocation overhead.
        Ref: JAX SPMD MNIST example, Flax NNX performance guide.
     e) Collective combine thresholds merge small AllReduce/AllGather ops:
        --xla_gpu_all_reduce_combine_threshold_bytes=256

   The data-parallel pattern from Flax NNX (examples/04_data_parallel_with_jit.py):
     - Replicate model weights: `jax.device_put(state, NamedSharding(mesh, P()))`
     - Shard batch dimension: `jax.device_put(data, NamedSharding(mesh, P('data')))`
     - `@nnx.jit` with `donate_argnums` for zero-copy parameter updates
     - No explicit device_put per batch — jit handles sharding automatically

   The SPMD MNIST example from JAX (examples/spmd_mnist_classifier_fromscratch.py):
     - Uses `@jax.jit(donate_argnums=0)` to reuse param buffers
     - Shards data in the data_stream generator, not in the training loop
     - Wraps train_step with `jax.set_mesh(mesh)` for automatic SPMD

References
----------
- XLA xla.proto: "xla_force_host_platform_device_count" documentation
- JAX docs/sharded-computation.md: automatic + manual parallelism
- JAX docs/benchmarking.md: JIT warmup, async dispatch, block_until_ready
- JAX docs/gpu_performance_tips.md: XLA flags for multi-device training
- JAX examples/spmd_mnist_classifier_fromscratch.py: data-parallel pattern
- Flax examples/nnx_toy_examples/04_data_parallel_with_jit.py: NNX sharding
- Flax docs_nnx/guides/flax_gspmd.md: SPMD sharding annotations
- JAX GitHub Discussion #14762: CPU pmap speedup measurements
- JAX GitHub Issue #6790: intra-op vs inter-op parallelism control
- B. Nikolic blog: "Is JAX multi-threaded when run on CPUs?" (2023)
"""

# ── Platform & device configuration ─────────────────────────────────────────
# JAX_PLATFORMS and XLA_FLAGS must be set *before* JAX is imported, so we do
# a lightweight pre-parse of --platform and --num-devices here.
# See docstring point 1 for why this matters.

import argparse
import os

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--platform", type=str, default="cpu")
_pre_parser.add_argument("--num-devices", type=int, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

os.environ["JAX_PLATFORMS"] = _pre_args.platform

if _pre_args.num_devices is not None and _pre_args.platform == "cpu":
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={_pre_args.num_devices}"
    )

# ── Regular imports (JAX now sees the configured platform & device count) ───

import json
import time
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

matplotlib.use("Agg")

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig


# ── Data generation ──────────────────────────────────────────────────────────


def generate_image_data(
    num_samples: int = 10000, image_size: int = 128, seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate synthetic image data for distributed benchmarking."""
    rng = np.random.RandomState(seed)
    return {
        "image": rng.rand(num_samples, image_size, image_size, 3).astype(np.float32),
        "label": rng.randint(0, 10, (num_samples,)).astype(np.int32),
    }


# ── Pipeline factory ────────────────────────────────────────────────────────


def normalize(element, key=None):
    """Normalize image to [0, 1]."""
    del key
    return element.update_data({"image": element.data["image"] / 255.0})


def create_pipeline(data: dict, batch_size: int = 64, seed: int = 0):
    """Create a standard image processing pipeline."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(seed))
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
    )
    return from_source(source, batch_size=batch_size).add(OperatorNode(normalizer))


# ── Simulated workload ──────────────────────────────────────────────────────
# The workload must be heavy enough that T_compute >> T_shard for scaling
# to appear (see docstring point 2).  With 128×128×3 images at bs=128,
# each batch is ~25 MB.  Ten iterations of softmax + gelu + normalization
# produce ~140 ms/batch on a single device, well above the ~5-10 ms
# sharding overhead.
#
# All operations are per-sample (reductions on axes 1,2,3 only), so the
# batch dimension (axis 0) shards cleanly — XLA compiles independent
# per-shard programs with zero cross-device communication.  This is
# "embarrassingly parallel" / SPMD.


@jax.jit
def _workload(images: jax.Array) -> jax.Array:
    """Simulate a forward-pass-like workload on each shard.

    The computation is SPMD-compatible: all reductions operate on spatial
    dimensions (1,2,3), so sharding along the batch dimension (0) requires
    no cross-device communication.  XLA compiles a per-shard program that
    each device executes independently on its local slice.
    """
    x = images
    for _ in range(10):
        # softmax along spatial axis → element-wise multiply → per-sample norm
        # This pattern is intentionally heavy to ensure T_compute >> T_shard.
        weights = jax.nn.softmax(x.mean(axis=-1, keepdims=True), axis=1)
        x = x * weights
        x = jax.nn.gelu(x - x.mean(axis=(1, 2, 3), keepdims=True))
        x = x / (x.std(axis=(1, 2, 3), keepdims=True) + 1e-5)
    return x.sum(axis=(1, 2, 3))


# ── Benchmark routines ───────────────────────────────────────────────────────


def benchmark_single_device(data: dict, batch_size: int = 64) -> dict:
    """Baseline: iterate pipeline on a single device with workload."""
    pipeline = create_pipeline(data, batch_size)

    # Warmup
    for i, batch in enumerate(pipeline):
        _ = _workload(batch["image"]).block_until_ready()
        if i >= 5:
            break

    # Timed
    pipeline = create_pipeline(data, batch_size)
    total_samples = 0
    t0 = time.perf_counter()
    for batch in pipeline:
        _ = _workload(batch["image"]).block_until_ready()
        total_samples += batch["image"].shape[0]
    elapsed = time.perf_counter() - t0

    return {
        "num_devices": 1,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 4),
        "samples_per_second": round(total_samples / elapsed, 2),
    }


def benchmark_sharded(
    data: dict, num_devices: int, batch_size: int = 64
) -> dict:
    """Benchmark pipeline + sharded computation across N devices.

    Uses JAX Mesh and NamedSharding to distribute batches, then runs
    a jitted workload that executes in parallel across shards.

    The loop structure — device_put → workload → block_until_ready — is
    intentionally synchronous so we measure true per-batch wall time.
    In production training loops (see docstring point 4), jax.jit's async
    dispatch naturally overlaps compute with host-side data loading,
    which would further improve effective throughput.
    """
    devices = jax.devices()[:num_devices]
    mesh = Mesh(np.array(devices), axis_names=("data",))
    # Data-parallel sharding: batch dimension split across devices.
    # Follows the pattern from Flax examples/04_data_parallel_with_jit.py:
    #   data_sharding = NamedSharding(mesh, PartitionSpec('data'))
    sharding = NamedSharding(mesh, PartitionSpec("data"))

    # Batch size must be divisible by device count for even sharding.
    effective_bs = (batch_size // num_devices) * num_devices
    pipeline = create_pipeline(data, effective_bs)

    # Warmup: JIT compilation happens on first call with each new input shape.
    # We run several batches to ensure compilation is complete and caches warm.
    # Ref: JAX docs/benchmarking.md — "first run includes compilation overhead"
    with mesh:
        for i, batch in enumerate(pipeline):
            images = jax.device_put(batch["image"], sharding)
            _ = _workload(images).block_until_ready()
            if i >= 5:
                break

    # Timed run.  block_until_ready() is mandatory for accurate timing because
    # JAX uses async dispatch — without it we'd only measure dispatch overhead
    # (~269 µs) not actual compute time.  Ref: JAX docs/benchmarking.md
    pipeline = create_pipeline(data, effective_bs)
    total_samples = 0
    t0 = time.perf_counter()
    with mesh:
        for batch in pipeline:
            images = jax.device_put(batch["image"], sharding)
            _ = _workload(images).block_until_ready()
            total_samples += batch["image"].shape[0]
    elapsed = time.perf_counter() - t0

    return {
        "num_devices": num_devices,
        "effective_batch_size": effective_bs,
        "total_samples": total_samples,
        "elapsed_seconds": round(elapsed, 4),
        "samples_per_second": round(total_samples / elapsed, 2),
    }


def run_scaling_benchmark(data: dict, batch_size: int = 64) -> list[dict]:
    """Run scaling benchmark across available device counts."""
    all_devices = jax.devices()
    max_devices = len(all_devices)

    # Test with powers of 2 up to available devices
    device_counts = []
    n = 1
    while n <= max_devices:
        device_counts.append(n)
        n *= 2

    results = []

    # Single device baseline
    print("  1 device (baseline)...", end=" ", flush=True)
    base = benchmark_single_device(data, batch_size)
    print(f"{base['samples_per_second']:>10.0f} samples/s")
    results.append(base)

    # Sharded configurations
    for nd in device_counts:
        print(f"  {nd} device(s) (sharded)...", end=" ", flush=True)
        result = benchmark_sharded(data, nd, batch_size)
        scaling_efficiency = (
            result["samples_per_second"] / (base["samples_per_second"] * nd)
        ) * 100
        result["scaling_efficiency_pct"] = round(scaling_efficiency, 2)
        print(
            f"{result['samples_per_second']:>10.0f} samples/s  "
            f"({scaling_efficiency:.1f}% efficiency)"
        )
        results.append(result)

    return results


def run_batch_size_vs_devices(
    data: dict, batch_sizes: list[int]
) -> dict[str, list[dict]]:
    """Benchmark batch size impact at each device count."""
    all_devices = jax.devices()
    device_counts = [1]
    n = 2
    while n <= len(all_devices):
        device_counts.append(n)
        n *= 2

    results = {}
    for nd in device_counts:
        key = f"{nd}_devices"
        results[key] = []
        print(f"\n  {nd} device(s):")
        for bs in batch_sizes:
            effective_bs = (bs // nd) * nd
            if effective_bs < nd:
                continue
            print(f"    batch_size={effective_bs:>4d}...", end=" ", flush=True)
            r = benchmark_sharded(data, nd, bs)
            print(f"{r['samples_per_second']:>10.0f} samples/s")
            results[key].append(r)

    return results


# ── Visualization ────────────────────────────────────────────────────────────


def plot_results(results: dict, output_dir: Path) -> None:
    """Generate and save distributed scaling benchmark visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device scaling: throughput + efficiency
    if "device_scaling" in results:
        data = results["device_scaling"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        devices = [r["num_devices"] for r in data]
        throughput = [r["samples_per_second"] for r in data]

        axes[0].bar(
            [str(d) for d in devices], throughput,
            color="steelblue", edgecolor="white",
        )
        axes[0].set_xlabel("Number of Devices")
        axes[0].set_ylabel("Throughput (samples/s)")
        axes[0].set_title("Throughput vs Device Count")
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, tp in enumerate(throughput):
            axes[0].text(i, tp + max(throughput) * 0.02, f"{tp:.0f}",
                         ha="center", fontsize=9)

        # Efficiency (skip baseline entry that lacks efficiency)
        eff_data = [r for r in data if "scaling_efficiency_pct" in r]
        if eff_data:
            eff_devices = [r["num_devices"] for r in eff_data]
            efficiency = [r["scaling_efficiency_pct"] for r in eff_data]
            axes[1].bar(
                [str(d) for d in eff_devices], efficiency,
                color="#ff8a65", edgecolor="white",
            )
            axes[1].axhline(y=100, color="gray", linestyle="--", alpha=0.5,
                            label="Ideal (100%)")
            axes[1].set_xlabel("Number of Devices")
            axes[1].set_ylabel("Scaling Efficiency (%)")
            axes[1].set_title("Scaling Efficiency vs Device Count")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis="y")
            for i, eff in enumerate(efficiency):
                axes[1].text(i, eff + 1, f"{eff:.1f}%", ha="center", fontsize=9)
        else:
            axes[1].text(0.5, 0.5, "Single device\n(no scaling data)",
                         ha="center", va="center", transform=axes[1].transAxes,
                         fontsize=12, color="gray")
            axes[1].set_title("Scaling Efficiency vs Device Count")

        plt.tight_layout()
        path = output_dir / "distributed-scaling-throughput.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")

    # 2. Batch size vs device count matrix
    if "batch_size_vs_devices" in results:
        matrix = results["batch_size_vs_devices"]
        fig, ax = plt.subplots(figsize=(12, 5))

        colors = ["steelblue", "#ff8a65", "#4fc3f7", "#e57373", "#81c784"]
        group_width = 0.8
        device_keys = sorted(matrix.keys())

        # Collect all batch sizes across device configs
        all_bs = sorted({
            r.get("effective_batch_size", r.get("batch_size"))
            for key in device_keys for r in matrix[key]
        })
        x_pos = np.arange(len(all_bs))
        bar_width = group_width / max(len(device_keys), 1)

        for idx, key in enumerate(device_keys):
            entries = matrix[key]
            bs_to_tp = {
                r.get("effective_batch_size", r.get("batch_size")): r["samples_per_second"]
                for r in entries
            }
            tp_values = [bs_to_tp.get(bs, 0) for bs in all_bs]
            offset = (idx - len(device_keys) / 2 + 0.5) * bar_width
            ax.bar(
                x_pos + offset, tp_values, bar_width,
                label=key.replace("_", " ").title(),
                color=colors[idx % len(colors)],
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(bs) for bs in all_bs])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title("Batch Size vs Device Count")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = output_dir / "distributed-batch-vs-devices.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Distributed Scaling Benchmark")
    parser.add_argument(
        "--platform", type=str, default="cpu", choices=["cpu", "cuda", "tpu"],
        help="JAX platform to run on (default: cpu)",
    )
    parser.add_argument(
        "--num-devices", type=int, default=None,
        help="Simulate N CPU devices via XLA_FLAGS (cpu platform only)",
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Distributed Scaling Benchmark")
    print("=" * 60)
    print(f"Platform:  {args.platform}")
    print(f"Backend:   {jax.default_backend()}")
    print(f"Devices:   {len(jax.devices())} ({jax.devices()})")
    print(f"Samples:   {args.samples}")
    print()

    data = generate_image_data(num_samples=args.samples)

    # Device scaling
    print("Device Scaling")
    print("-" * 60)
    scaling_results = run_scaling_benchmark(data, args.batch_size)

    # Batch size x device count matrix
    print()
    print("Batch Size vs Device Count")
    print("-" * 60)
    matrix_results = run_batch_size_vs_devices(data, [32, 64, 128, 256])

    # Collect results
    all_results = {
        "hardware": {
            "backend": str(jax.default_backend()),
            "devices": [str(d) for d in jax.devices()],
            "num_devices": len(jax.devices()),
        },
        "config": {
            "num_samples": args.samples,
            "base_batch_size": args.batch_size,
        },
        "device_scaling": scaling_results,
        "batch_size_vs_devices": matrix_results,
    }

    # Save results
    output_path = Path(args.output or "temp/distributed_scaling_results.json")
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
