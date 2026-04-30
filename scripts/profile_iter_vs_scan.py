"""Local profiler comparing datarax iter-mode vs scan-mode vs jax-dataloader.

Goal: dissect throughput gaps between three measurement dimensions
on the same scenarios:

- ``Datarax`` adapter   — Python iterator over ``Pipeline.step()``.
- ``Datarax-scan``      — whole-epoch ``Pipeline.scan(...)`` body.
- ``jax-dataloader``    — numpy-host slicing baseline.

Probes whether per-batch NNX overhead is constant (suggesting
iterator-mode marshalling cost amortizable by scan) or scales with
stage count (suggesting per-stage JIT churn).

Scenarios:
  - NLP-1 (0 transforms) — pure iteration baseline.
  - TAB-1 (1 transform: Normalize) — minimal stage overhead.
  - PC-1 depth_1 / depth_5 / depth_10 — datarax-only scaling probe.

Run:
    source activate.sh && python scripts/profile_iter_vs_scan.py
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.adapters.datarax_scan_adapter import DataraxScanAdapter
from benchmarks.adapters.jax_dl_adapter import JaxDataloaderAdapter
from benchmarks.scenarios.nlp.nlp1_llm_pretraining import get_variant as get_nlp1
from benchmarks.scenarios.pipeline_complexity.pc1_chain_depth import get_variant as get_pc1
from benchmarks.scenarios.tabular.tab1_dense_features import get_variant as get_tab1


def _time_section(label: str, fn: Callable, repeats: int = 3) -> float:
    """Time *fn* across *repeats* and return the median seconds."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    median = sorted(durations)[len(durations) // 2]
    print(
        f"    {label:<42s} {median * 1000:>9.2f} ms  "
        f"(min={min(durations) * 1000:.2f}, reps={repeats})"
    )
    return median


def _profile_one(
    scenario_label: str,
    config: ScenarioConfig,
    data: dict,
    adapter_cls,
    num_batches: int = 100,
) -> dict:
    """Profile a single (scenario, adapter) pair and return timings.

    Re-runs setup() before each timed measurement so each call starts
    from a fresh iterator state. Calls ``adapter.warmup(num_batches)``
    so adapters that compile per-length (e.g. scan-mode) hit a warm
    XLA cache during the timed call.
    """
    print(f"\n  --- {scenario_label} ---")
    print(f"    transforms = {config.transforms}")
    print(f"    batch_size = {config.batch_size}, dataset = {config.dataset_size}")

    def _fresh_warm() -> Any:
        adapter = adapter_cls()
        adapter.setup(config, data)
        adapter.warmup(num_batches)
        # Reset pipeline position so the timed call sees a fresh source.
        # Without this, iter-mode finishes immediately because warmup
        # advanced position past source_length; scan-mode wraps but the
        # cached JIT graph is keyed on initial position state.
        pipeline = getattr(adapter, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "_position"):
            pipeline._position[...] = jnp.int32(0)
        return adapter

    adapter = adapter_cls()
    setup_t = _time_section("setup()", lambda: adapter.setup(config, data), repeats=1)
    adapter.teardown()

    full_durations = []
    for _ in range(3):
        a = _fresh_warm()
        s = time.perf_counter()
        a.iterate(num_batches)
        full_durations.append(time.perf_counter() - s)
        a.teardown()
    full_t = sorted(full_durations)[len(full_durations) // 2]
    print(
        f"    full iterate(num_batches) [warm]           "
        f"{full_t * 1000:>9.2f} ms  (min={min(full_durations) * 1000:.2f}, reps=3)"
    )

    elements_per_sec = (config.batch_size * num_batches) / full_t
    per_batch_us = (full_t / num_batches) * 1e6
    return {
        "setup_ms": setup_t * 1000,
        "iterate_only_ms": full_t * 1000,
        "full_ms": full_t * 1000,
        "per_batch_us": per_batch_us,
        "throughput_elem_per_sec": elements_per_sec,
    }


def _compare_pair(scenario_label: str, variant, num_batches: int = 100) -> dict:
    """Run datarax and jax-dataloader on one variant and report ratios."""
    print(f"\n{'=' * 72}\n{scenario_label}\n{'=' * 72}")
    config = variant.config
    data = variant.data_generator()

    results = {}
    for name, adapter_cls in [
        ("jax-dataloader", JaxDataloaderAdapter),
        ("datarax-iter", DataraxAdapter),
        ("datarax-scan", DataraxScanAdapter),
    ]:
        if not adapter_cls().is_available():
            print(f"  SKIP: {name} unavailable")
            continue
        print(f"\n  >>> {name}")
        results[name] = _profile_one(name, config, data, adapter_cls, num_batches)

    if {"datarax-iter", "datarax-scan", "jax-dataloader"} <= results.keys():
        di = results["datarax-iter"]
        ds = results["datarax-scan"]
        j = results["jax-dataloader"]
        print("\n  Ratios:")
        print(f"    iter / jax-dl  (full):  {di['full_ms'] / j['full_ms']:>8.2f}x")
        print(f"    scan / jax-dl  (full):  {ds['full_ms'] / j['full_ms']:>8.2f}x")
        print(f"    iter / scan    (speedup from scan): {di['full_ms'] / ds['full_ms']:>5.2f}x")
        print(
            f"    throughput jax-dl / iter:  "
            f"{j['throughput_elem_per_sec'] / di['throughput_elem_per_sec']:>8.2f}x"
        )
        print(
            f"    throughput jax-dl / scan:  "
            f"{j['throughput_elem_per_sec'] / ds['throughput_elem_per_sec']:>8.2f}x"
        )
    return results


def _profile_pc1_scaling(num_batches: int = 50) -> list[dict]:
    """Run only datarax on PC-1 across chain depths to measure stage scaling."""
    print(f"\n{'=' * 72}\nPC-1 scaling — datarax only, varying chain depth\n{'=' * 72}")
    rows = []
    for depth_name in ["depth_1", "depth_5", "depth_10"]:
        variant = get_pc1(depth_name)
        config = variant.config
        data = variant.data_generator()
        print(f"\n  --- PC-1 {depth_name} (transforms={len(config.transforms)}) ---")

        if not DataraxAdapter().is_available():
            print("  SKIP: datarax unavailable")
            continue

        result = _profile_one(f"datarax {depth_name}", config, data, DataraxAdapter, num_batches)
        result["depth_name"] = depth_name
        result["depth"] = len(config.transforms)
        rows.append(result)

    if rows:
        print("\n  Datarax per-batch overhead vs chain depth:")
        print(f"  {'depth':<10}{'per-batch us':>16}{'iterate-only ms':>20}{'throughput e/s':>20}")
        for r in rows:
            print(
                f"  {r['depth_name']:<10}{r['per_batch_us']:>14.1f}us"
                f"{r['iterate_only_ms']:>18.1f}ms"
                f"{r['throughput_elem_per_sec']:>18.0f}"
            )
        if len(rows) >= 2:
            base = rows[0]
            print("\n  Marginal cost per added stage (vs depth_1):")
            for r in rows[1:]:
                delta_depth = r["depth"] - base["depth"]
                delta_us = r["per_batch_us"] - base["per_batch_us"]
                print(
                    f"    +{delta_depth} stages -> +{delta_us:>7.1f} us/batch "
                    f"(={delta_us / max(delta_depth, 1):.1f} us per stage)"
                )
    return rows


def main() -> None:
    """Run main."""
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    summary = {}
    summary["NLP-1 small"] = _compare_pair("NLP-1 small (0 transforms)", get_nlp1("small"))
    summary["TAB-1 small"] = _compare_pair(
        "TAB-1 small (1 transform: Normalize)", get_tab1("small")
    )
    pc1_rows = _profile_pc1_scaling(num_batches=50)

    print(f"\n{'=' * 72}\nFINAL SUMMARY\n{'=' * 72}")
    print(f"{'scenario':<24}{'datarax/jaxdl iter':>22}{'jaxdl/dx throughput':>24}")
    for label, results in summary.items():
        if "datarax" in results and "jax-dataloader" in results:
            d, j = results["datarax"], results["jax-dataloader"]
            print(
                f"{label:<24}"
                f"{d['full_ms'] / j['full_ms']:>20.1f}x"
                f"{j['throughput_elem_per_sec'] / d['throughput_elem_per_sec']:>22.1f}x"
            )

    if pc1_rows and len(pc1_rows) >= 2:
        per_stage = []
        base = pc1_rows[0]
        for r in pc1_rows[1:]:
            delta_depth = r["depth"] - base["depth"]
            delta_us = r["per_batch_us"] - base["per_batch_us"]
            if delta_depth > 0:
                per_stage.append(delta_us / delta_depth)
        if per_stage:
            avg = sum(per_stage) / len(per_stage)
            print(f"\nDatarax per-stage marginal cost: ~{avg:.1f} us/batch/stage")
            print(
                f"Datarax fixed per-batch overhead (depth_1): ~{base['per_batch_us']:.1f} us/batch"
            )


if __name__ == "__main__":
    main()
