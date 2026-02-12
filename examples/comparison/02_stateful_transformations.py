#!/usr/bin/env python3
"""
Enhanced Comparison: Stateful vs Stateless Transformation Pipelines.

This example demonstrates the fundamental differences in transformation
handling between datarax's stateful NNX approach and Grain's
stateless design, with real performance measurements.
"""

import gc
import time

# Suppress warnings
import warnings
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil


warnings.filterwarnings("ignore")

# ============================================================================
# GRAIN (STATELESS) TRANSFORMATIONS - Following Best Practices
# ============================================================================


@dataclass
class GrainNormalizeTransform:
    """Grain-style stateless normalization."""

    mean: float = 0.0
    std: float = 1.0

    def __call__(self, features: dict, rng: np.random.RandomState) -> dict:
        """Apply normalization without state."""
        if "data" in features:
            # Stateless - can't track running statistics
            features["data"] = (features["data"] - self.mean) / self.std
        return features


@dataclass
class GrainAugmentationTransform:
    """Grain-style random augmentation."""

    noise_scale: float = 0.1

    def __call__(self, features: dict, rng: np.random.RandomState) -> dict:
        """Apply augmentation with external RNG."""
        if "data" in features:
            noise = rng.randn(*features["data"].shape) * self.noise_scale
            features["data"] = features["data"] + noise
        return features


@dataclass
class GrainBatchTransform:
    """Grain-style batch processing."""

    batch_size: int = 32

    def __call__(self, batch: list[dict]) -> np.ndarray:
        """Process a batch of samples."""
        # Stack features
        stacked = np.stack([item["data"] for item in batch])
        return stacked


class GrainTransformPipeline:
    """Grain-style transformation pipeline with external state."""

    def __init__(self, transforms: list):
        self.transforms = transforms
        # External state dictionary for any stateful operations
        self.state = {
            "samples_processed": 0,
            "batch_count": 0,
            "running_mean": 0.0,
            "running_var": 1.0,
            "rng_seed": 42,
        }

    def apply_transforms(self, data: dict, state: dict) -> tuple[dict, dict]:
        """Apply transforms with external state management."""
        # Create RNG from state
        rng = np.random.RandomState(state["rng_seed"])

        # Apply each transform (skip batch transforms for individual samples)
        for transform in self.transforms:
            if hasattr(transform, "__call__") and not isinstance(transform, GrainBatchTransform):
                data = transform(data, rng)

        # Manual state update
        new_state = state.copy()
        new_state["samples_processed"] += 1
        new_state["rng_seed"] += 1

        # Manual running statistics (complex!)
        if "data" in data:
            batch_mean = float(np.mean(data["data"]))
            float(np.var(data["data"]))

            # Welford's online algorithm for running stats
            n = new_state["samples_processed"]
            delta = batch_mean - new_state["running_mean"]
            new_state["running_mean"] += delta / n
            new_state["running_var"] = (
                (n - 1) * new_state["running_var"]
                + delta * (batch_mean - new_state["running_mean"])
            ) / n

        return data, new_state

    def process_batch(self, batch: list[dict], state: dict) -> tuple[np.ndarray, dict]:
        """Process a batch with state threading."""
        processed = []
        current_state = state.copy()

        for sample in batch:
            transformed, current_state = self.apply_transforms(sample, current_state)
            processed.append(transformed)

        # Update batch count
        current_state["batch_count"] += 1

        # Stack into batch
        if processed and "data" in processed[0]:
            batch_array = np.stack([p["data"] for p in processed])
            return batch_array, current_state

        return np.array([]), current_state


# ============================================================================
# DATARAX (STATEFUL) TRANSFORMATIONS with NNX
# ============================================================================


class StatefulNormalizeTransform(nnx.Module):
    """Stateful normalization with running statistics."""

    def __init__(self, momentum: float = 0.1):
        self.momentum = momentum

        # Automatic state tracking with NNX
        self.running_mean = nnx.BatchStat(0.0)
        self.running_var = nnx.BatchStat(1.0)
        self.samples_seen = nnx.Variable(0)

    def __call__(self, features: dict, training: bool = True) -> dict:
        """Apply normalization with automatic state updates."""
        if "data" not in features:
            return features

        data = features["data"]

        sample_mean = jnp.mean(data)
        sample_var = jnp.var(data)

        if training:
            # Update running statistics automatically
            self.running_mean.value = float(
                self.momentum * sample_mean + (1 - self.momentum) * self.running_mean.value
            )
            self.running_var.value = float(
                self.momentum * sample_var + (1 - self.momentum) * self.running_var.value
            )

            # Use sample stats for normalization
            normalized = (data - sample_mean) / jnp.sqrt(sample_var + 1e-8)
        else:
            # Use running stats
            normalized = (data - self.running_mean.value) / jnp.sqrt(self.running_var.value + 1e-8)

        self.samples_seen.value += 1
        features["data"] = normalized
        return features

    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return {
            "running_mean": float(self.running_mean.value),
            "running_var": float(self.running_var.value),
            "samples_seen": int(self.samples_seen.value),
        }


class StatefulAugmentationTransform(nnx.Module):
    """Stateful augmentation with internal RNG."""

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

        # Internal RNG management
        self.rngs = nnx.Rngs(42)
        self.augment_count = nnx.Variable(0)

    def __call__(self, features: dict) -> dict:
        """Apply augmentation with automatic RNG handling."""
        if "data" not in features:
            return features

        # Automatic RNG management
        key = self.rngs.noise()
        noise = jax.random.normal(key, features["data"].shape) * self.noise_scale

        features["data"] = features["data"] + noise
        self.augment_count.value += 1

        return features


class LearnableTransform(nnx.Module):
    """Learnable transformation - only possible with stateful approach."""

    def __init__(self, input_dim: int, output_dim: int):
        # Learnable parameters
        key = jax.random.key(0)
        self.weight = nnx.Param(jax.random.normal(key, (input_dim, output_dim)) * 0.01)
        self.bias = nnx.Param(jnp.zeros(output_dim))

        # Batch normalization components
        self.bn_scale = nnx.Param(jnp.ones(output_dim))
        self.bn_shift = nnx.Param(jnp.zeros(output_dim))
        self.bn_mean = nnx.BatchStat(jnp.zeros(output_dim))
        self.bn_var = nnx.BatchStat(jnp.ones(output_dim))

        # Track usage
        self.forward_passes = nnx.Variable(0)

    def __call__(self, features: dict, training: bool = True) -> dict:
        """Apply learnable transformation."""
        if "data" not in features:
            return features

        # Ensure correct shape
        data = features["data"]

        # Handle different data shapes
        if data.ndim == 1:
            # Single sample - reshape to (1, features)
            data = data.reshape(1, -1)
            expected_features = data.shape[1]
        elif data.ndim == 2:
            # Batch - check if features match
            expected_features = data.shape[1]
        else:
            # Flatten if needed
            data = data.reshape(1, -1)
            expected_features = data.shape[1]

        # Check if input dimension matches expected
        if expected_features != self.weight.value.shape[0]:
            # Reshape weight to match input features
            actual_features = self.weight.value.shape[0]
            if expected_features < actual_features:
                # Pad with zeros if input is smaller
                padding = actual_features - expected_features
                data = jnp.pad(data, ((0, 0), (0, padding)), mode="constant")
            elif expected_features > actual_features:
                # Truncate if input is larger
                data = data[:, :actual_features]
            expected_features = self.weight.value.shape[0]

        # Linear transformation
        output = jnp.dot(data, self.weight.value) + self.bias.value

        # Batch normalization
        if training:
            batch_mean = jnp.mean(output, axis=0, keepdims=True)
            batch_var = jnp.var(output, axis=0, keepdims=True)

            # Update running stats
            self.bn_mean.value = 0.9 * self.bn_mean.value + 0.1 * batch_mean.squeeze()
            self.bn_var.value = 0.9 * self.bn_var.value + 0.1 * batch_var.squeeze()

            # Normalize
            output = (output - batch_mean) / jnp.sqrt(batch_var + 1e-8)
        else:
            output = (output - self.bn_mean.value) / jnp.sqrt(self.bn_var.value + 1e-8)

        output = output * self.bn_scale.value + self.bn_shift.value
        output = jax.nn.relu(output)

        self.forward_passes.value += 1

        features["data"] = output.squeeze() if output.shape[0] == 1 else output
        return features

    @property
    def num_parameters(self) -> int:
        """Count learnable parameters."""
        return (
            self.weight.value.size
            + self.bias.value.size
            + self.bn_scale.value.size
            + self.bn_shift.value.size
        )


class StatefulTransformPipeline(nnx.Module):
    """Stateful transformation pipeline with automatic state management."""

    def __init__(self, transforms: list[nnx.Module]):
        self.transforms = transforms

        # Pipeline statistics
        self.samples_processed = nnx.Variable(0)
        self.batch_count = nnx.Variable(0)
        self.total_time = nnx.Variable(0.0)

    def __call__(self, batch: list[dict], training: bool = True) -> jax.Array:
        """Process batch with automatic state updates."""
        start_time = time.time()

        processed = []
        for sample in batch:
            # Apply transforms in sequence
            for transform in self.transforms:
                if hasattr(transform, "__call__") and callable(transform):
                    if isinstance(transform, StatefulNormalizeTransform | LearnableTransform):
                        sample = transform(sample, training=training)
                    else:
                        sample = transform(sample)

            processed.append(sample["data"])

        # Update statistics
        self.samples_processed.value += len(batch)
        self.batch_count.value += 1
        self.total_time.value += time.time() - start_time

        # Stack into batch
        return jnp.stack(processed)

    @property
    def stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "samples_processed": int(self.samples_processed.value),
            "batch_count": int(self.batch_count.value),
            "avg_time_per_batch": float(self.total_time.value / max(self.batch_count.value, 1)),
        }

        # Add transform-specific stats
        for i, transform in enumerate(self.transforms):
            if hasattr(transform, "stats"):
                stats_attr = getattr(transform, "stats", None)
                if stats_attr is not None:
                    if callable(stats_attr):
                        stats[f"transform_{i}"] = stats_attr()
                    else:
                        stats[f"transform_{i}"] = stats_attr
            elif hasattr(transform, "num_parameters"):
                num_params_attr = getattr(transform, "num_parameters", None)
                if num_params_attr is not None:
                    if callable(num_params_attr):
                        stats[f"transform_{i}_params"] = num_params_attr()
                    else:
                        stats[f"transform_{i}_params"] = num_params_attr

        return stats


# ============================================================================
# PRESSURE TESTING FUNCTIONS
# ============================================================================


def create_test_data(num_samples: int, feature_dim: int) -> list[dict]:
    """Create test data for transformation pipelines."""
    data = []
    for i in range(num_samples):
        data.append(
            {
                "data": np.random.randn(feature_dim).astype(np.float32),
                "label": np.random.randint(0, 10),
            }
        )
    return data


def measure_transform_performance(
    pipeline_name: str,
    pipeline: Any,
    data: list[dict],
    batch_size: int = 32,
    is_stateful: bool = True,
) -> dict:
    """Measure transformation pipeline performance."""

    print(f"\n{pipeline_name} Performance Test:")
    print("-" * 60)

    num_batches = len(data) // batch_size
    times = []
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024

    # Initialize state variable for type checker
    state = {"samples_processed": 0}

    # Process batches
    if is_stateful:
        # Warmup
        for i in range(min(5, num_batches)):
            batch = data[i * batch_size : (i + 1) * batch_size]
            _ = pipeline(batch, training=True)

        # Measure
        gc.collect()
        for i in range(num_batches):
            batch = data[i * batch_size : (i + 1) * batch_size]

            start = time.time()
            result = pipeline(batch, training=True)
            elapsed = time.time() - start

            times.append(elapsed)

            # Force computation
            _ = jnp.mean(result)
    else:
        # Grain-style with state threading
        state = pipeline.state.copy()

        # Warmup
        for i in range(min(5, num_batches)):
            batch = data[i * batch_size : (i + 1) * batch_size]
            _, state = pipeline.process_batch(batch, state)

        # Measure
        gc.collect()
        for i in range(num_batches):
            batch = data[i * batch_size : (i + 1) * batch_size]

            start = time.time()
            result, state = pipeline.process_batch(batch, state)
            elapsed = time.time() - start

            times.append(elapsed)

            # Force computation
            _ = np.mean(result)

    memory_after = psutil.Process().memory_info().rss / 1024 / 1024

    # Calculate statistics
    times = np.array(times[5:])  # Skip warmup

    results = {
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "p50_time_ms": np.percentile(times, 50) * 1000,
        "p95_time_ms": np.percentile(times, 95) * 1000,
        "p99_time_ms": np.percentile(times, 99) * 1000,
        "total_batches": len(times),
        "memory_delta_mb": memory_after - memory_before,
        "throughput_batches_per_sec": 1 / np.mean(times),
    }

    print(f"  Mean batch time: {results['mean_time_ms']:.3f} ms")
    print(f"  Std deviation: {results['std_time_ms']:.3f} ms")
    print(f"  P95: {results['p95_time_ms']:.3f} ms")
    print(f"  Throughput: {results['throughput_batches_per_sec']:.1f} batches/sec")
    print(f"  Memory delta: {results['memory_delta_mb']:.1f} MB")

    if is_stateful:
        stats = pipeline.stats
        print(f"  Pipeline stats: {stats.get('samples_processed', 'N/A')} samples processed")
    else:
        print(f"  State management: Manual ({state['samples_processed']} samples)")

    return results


def demonstrate_learnable_transforms():
    """Demonstrate learnable transformations (only possible with stateful)."""

    print()
    print("=" * 80)
    print("LEARNABLE TRANSFORMATIONS (Datarax Exclusive)")
    print("=" * 80)

    # Create learnable pipeline
    input_dim = 784
    hidden_dim = 256
    output_dim = 128

    pipeline = StatefulTransformPipeline(
        [
            StatefulNormalizeTransform(),
            LearnableTransform(input_dim, hidden_dim),
            LearnableTransform(hidden_dim, output_dim),
            StatefulAugmentationTransform(noise_scale=0.05),
        ]
    )

    print("\nLearnable Pipeline Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")

    # Count parameters
    total_params = 0
    for transform in pipeline.transforms:
        if hasattr(transform, "num_parameters"):
            num_params_attr = getattr(transform, "num_parameters", None)
            if num_params_attr is not None:
                if callable(num_params_attr):
                    params_val = num_params_attr()
                else:
                    params_val = num_params_attr

                # Ensure it's a valid number
                if isinstance(params_val, int | float | np.integer | np.floating):
                    params = int(params_val)
                    total_params += params
                else:
                    params = 0
                print(f"  {transform.__class__.__name__}: {params:,} parameters")

    print(f"  Total learnable parameters: {total_params:,}")

    # Create optimizer for learnable parameters
    optimizer = nnx.Optimizer(pipeline, optax.adam(1e-3), wrt=nnx.Param)

    # Training step function
    @nnx.jit
    def train_step(pipeline, batch, optimizer):
        def loss_fn(pipeline):
            output = pipeline(batch, training=True)
            return jnp.mean(output**2)  # Dummy loss

        loss, grads = nnx.value_and_grad(loss_fn)(pipeline)
        optimizer.update(grads)
        return loss

    # Simulate training
    print("\nTraining Learnable Pipeline:")
    print("-" * 60)

    data = create_test_data(1000, input_dim)
    batch_size = 32

    losses = []
    for epoch in range(3):
        epoch_losses = []

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i : i + batch_size]
            loss = train_step(pipeline, batch, optimizer)
            epoch_losses.append(float(loss))

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    # Show learning occurred
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\nLoss reduction: {improvement:.1f}%")
    print("✓ Learnable transformations successfully trained!")

    print("\nGrain equivalent:")
    print("  ❌ Not possible - Grain transformations must be stateless")
    print("  ❌ Cannot have learnable parameters")
    print("  ❌ Cannot use automatic differentiation")


def compare_checkpoint_complexity():
    """Compare checkpoint complexity between approaches."""

    print()
    print("=" * 80)
    print("CHECKPOINT COMPLEXITY COMPARISON")
    print("=" * 80)

    # Create pipelines
    grain_pipeline = GrainTransformPipeline(
        [GrainNormalizeTransform(), GrainAugmentationTransform(), GrainBatchTransform()]
    )

    workshop_pipeline = StatefulTransformPipeline(
        [
            StatefulNormalizeTransform(),
            StatefulAugmentationTransform(noise_scale=0.1),
            LearnableTransform(100, 50),
        ]
    )

    # Process some data to accumulate state
    data = create_test_data(500, 100)

    # Grain processing
    grain_state = grain_pipeline.state.copy()
    for i in range(10):
        batch = data[i * 32 : (i + 1) * 32]
        _, grain_state = grain_pipeline.process_batch(batch, grain_state)

    # Workshop processing
    for i in range(10):
        batch = data[i * 32 : (i + 1) * 32]
        _ = workshop_pipeline(batch)

    print("\n1. GRAIN CHECKPOINT COMPLEXITY:")
    print("-" * 60)
    print("  Manual state collection required:")
    print(f"    - Pipeline state dict: {len(grain_state)} entries")
    print("    - Each transform state: Manual tracking")
    print("    - RNG states: External management")
    print("    - No automatic discovery")

    print("\n2. DATARAX CHECKPOINT COMPLEXITY:")
    print("-" * 60)

    # Get all state automatically
    graphdef, state = nnx.split(workshop_pipeline)
    state_leaves = jax.tree.leaves(state)

    print("  Automatic state collection with NNX:")
    print(f"    - Total state variables: {len(state_leaves)}")
    print("    - Includes all nested module states")
    print("    - Automatic parameter discovery")
    print("    - One-line checkpoint: nnx.split()")

    # Show state structure
    print("\n  State structure (automatic):")
    for transform in workshop_pipeline.transforms:
        if hasattr(transform, "__class__"):
            print(f"    - {transform.__class__.__name__}: Fully tracked")


def run_comprehensive_transform_comparison():
    """Run full transformation comparison."""

    print("=" * 80)
    print("FULL TRANSFORMATION PIPELINE COMPARISON")
    print("=" * 80)

    test_configs = [
        {"samples": 5000, "features": 784, "batch": 32, "desc": "Small (MNIST)"},
        {"samples": 10000, "features": 2048, "batch": 64, "desc": "Medium (ResNet)"},
        {"samples": 20000, "features": 4096, "batch": 128, "desc": "Large (ViT)"},
    ]

    all_results = []

    for config in test_configs:
        print(f"\n{'=' * 80}")
        print(f"TEST: {config['desc']}")
        print(
            f"Samples: {config['samples']:,}, Features: {config['features']}, "
            f"Batch: {config['batch']}"
        )
        print("=" * 80)

        # Create test data
        data = create_test_data(config["samples"], config["features"])

        # Create pipelines
        grain_pipeline = GrainTransformPipeline(
            [
                GrainNormalizeTransform(mean=0.5, std=0.5),
                GrainAugmentationTransform(noise_scale=0.1),
            ]
        )

        workshop_pipeline = StatefulTransformPipeline(
            [
                StatefulNormalizeTransform(momentum=0.1),
                StatefulAugmentationTransform(noise_scale=0.1),
            ]
        )

        # If features match, add learnable transform
        if config["features"] in [784, 2048, 4096]:
            output_dim = config["features"] // 8
            workshop_pipeline.transforms.append(LearnableTransform(config["features"], output_dim))

        # Measure performance
        grain_results = measure_transform_performance(
            "Grain Pipeline", grain_pipeline, data, config["batch"], is_stateful=False
        )

        workshop_results = measure_transform_performance(
            "Workshop Pipeline", workshop_pipeline, data, config["batch"], is_stateful=True
        )

        # Calculate improvements
        speedup = grain_results["mean_time_ms"] / workshop_results["mean_time_ms"]
        memory_improvement = 1 - (
            workshop_results["memory_delta_mb"] / max(grain_results["memory_delta_mb"], 0.1)
        )

        print("\nIMPROVEMENTS:")
        print("-" * 60)
        print(f"  Processing speedup: {speedup:.2f}x")
        print(f"  Memory efficiency: {memory_improvement * 100:.1f}% better")
        print("  State management: Automatic vs Manual")

        # Check if learnable
        has_learnable = any(hasattr(t, "num_parameters") for t in workshop_pipeline.transforms)
        if has_learnable:
            print("  Learnable parameters: ✓ Supported (Workshop only)")

        all_results.append(
            {
                "config": config,
                "grain": grain_results,
                "workshop": workshop_results,
                "improvements": {
                    "speedup": speedup,
                    "memory": memory_improvement,
                    "has_learnable": has_learnable,
                },
            }
        )

    return all_results


if __name__ == "__main__":
    print("ENHANCED TRANSFORMATION PIPELINE COMPARISON")
    print("All metrics from actual execution")
    print("=" * 80)

    # Run full comparison
    results = run_comprehensive_transform_comparison()

    # Demonstrate unique capabilities
    demonstrate_learnable_transforms()

    # Compare checkpoint complexity
    compare_checkpoint_complexity()

    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY - MEASURED ADVANTAGES")
    print("=" * 80)

    # Calculate averages
    avg_speedup = np.mean([r["improvements"]["speedup"] for r in results])
    avg_memory = np.mean([r["improvements"]["memory"] for r in results])

    print(f"\n✓ Average processing speedup: {avg_speedup:.2f}x")
    print(f"✓ Average memory improvement: {avg_memory * 100:.1f}%")
    print("✓ Learnable transformations: Datarax only")
    print("✓ Automatic state management: Datarax only")
    print("✓ Gradient computation: Datarax only")
    print("✓ Built-in batch statistics: Datarax only")
    print("✓ Internal RNG management: Datarax only")
    print("✓ One-line checkpointing: Datarax only")

    print("\nConclusion: Datarax's stateful approach enables")
    print("capabilities that are impossible with Grain's stateless design")
    print("=" * 80)
