# Framework Comparison

This page describes the comparative analysis framework. Live results are available on the [W&B dashboard](dashboard.md) with interactive charts, comparison tables, and filtering.

---

## Metrics

Every benchmark measures three primary metric groups:

| Group | Metrics | Direction |
|-------|---------|-----------|
| **Throughput** | Elements processed per second | Higher is better |
| **Latency** | Per-batch time (p50, p95, p99) | Lower is better |
| **Memory** | Peak RSS, GPU memory | Lower is better |

The W&B dashboard groups these automatically using slash notation (`Throughput/throughput/Datarax`).

---

## Scenario Coverage

Each adapter supports only the scenarios where it implements the required transforms. Adapters that cannot implement a scenario's transforms are excluded from it rather than measured with less work. The checkmarks below show support on the 6 most widely covered scenarios; the **Total** column is the empirical full-catalog coverage count (of 37 scenarios) from the [Coverage Matrix](https://github.com/avitai/datarax/blob/main/benchmarks/COVERAGE_MATRIX.md).

| Framework | CV-1 | NLP-1 | TAB-1 | MM-1 | DIST-1 | PR-1 | Total |
|-----------|:----:|:-----:|:-----:|:----:|:------:|:----:|:-----:|
| **Datarax (iter)** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | **37** |
| **Datarax (scan)** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | **37** |
| Google Grain | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | 25 |
| tf.data | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | 25 |
| PyTorch DataLoader | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | 25 |
| SPDL | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | | 25 |
| NVIDIA DALI | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | | 9 |
| jax-dataloader | :white_check_mark: | | | | | | 13 |
| FFCV | :white_check_mark: | | | | | | 13 |
| Deep Lake | :white_check_mark: | | | | | | 13 |
| MosaicML Streaming | :white_check_mark: | :white_check_mark: | | | | | 2 |
| WebDataset | :white_check_mark: | :white_check_mark: | | | | | 2 |
| HuggingFace Datasets | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | 13 |
| Ray Data | | :white_check_mark: | :white_check_mark: | | | | 2 |
| LitData | :white_check_mark: | | | | | | 1 |
| Energon | | | | :white_check_mark: | | | 1 |

!!! note "Datarax supports all 37 scenarios"
    The table above shows the 6 most widely supported scenarios. Both Datarax adapters (iter-mode and scan-mode) support the full 37-scenario catalog, including AUG-1/AUG-2/AUG-3, PC-1 through PC-5, IO-1 through IO-4, NNX-1/XFMR-1, and the 9 heavy `H*` variants. **25 of 37 scenarios run on ≥3 frameworks** — the set with meaningful cross-framework comparison.

---

## Fair Comparison Views

Use two complementary views to avoid misleading conclusions:

1.  **Same-backend, shared-coverage view (canonical):** compare frameworks only on the scenario intersection they all support on the same backend/hardware profile.
2.  **Native-optimal view:** compare frameworks on scenarios that represent each framework's strongest native capabilities.

Canonical published cloud reports use on-demand Vast A100 runs, profile-controlled scenario sets, and manifest/backend-truth validation.

---

## Visualization

The W&B dashboard provides interactive versions of these chart types:

| Chart | What It Shows |
|-------|---------------|
| **Comparison Table** | All frameworks side-by-side with best values highlighted |
| **Throughput Bars** | Grouped bar chart — elem/s per scenario per framework |
| **Latency Distribution** | Per-batch latency distribution across frameworks |
| **Memory Profile** | Peak RSS comparison across frameworks |
| **Ranking Tables** | Per-metric rankings with delta-from-best percentages |

For local chart generation (offline use), the `benchmarks.visualization.charts` module can produce static plots:

```python
from pathlib import Path

from benchmarks.runners.full_runner import ComparativeResults
from benchmarks.visualization.charts import ChartGenerator

results = ComparativeResults.load(Path("benchmark-data/reports/latest"))
gen = ChartGenerator(results, Path("benchmark-data/charts"))
gen.generate_all()
```

---

## Comparative Analysis

### Capability coverage (not throughput contests)

Several scenarios run only on the Datarax adapters because they exercise
capabilities the transform-based competitor adapters do not express. These are
**capability-coverage differentiators, not head-to-head throughput wins**, and
per the [Coverage Matrix](https://github.com/avitai/datarax/blob/main/benchmarks/COVERAGE_MATRIX.md) must not be
presented as competitive leads:

-   **Pipeline Complexity (PC-2 branching DAG, PC-3 differentiable rebatch, PC-4 probabilistic, PC-5 differentiable/learnable)**: Datarax's DAG execution engine handles complex multi-branch and gradient-flowing pipelines that other frameworks cannot express, so there is no peer to compare against.
-   **Datarax Unique (NNX-1, XFMR-1)**: Flax NNX module integration and JIT+vmap transform acceleration are exclusive to Datarax.
-   The full Datarax-exclusive set (12 scenarios: CV-3, CV-4, IO-3, IO-4, NLP-2, NNX-1, PC-2, PC-3, PC-4, PC-5, PR-2, XFMR-1) is enumerated in the Coverage Matrix.

### Strengths

Scenarios where Datarax leads other frameworks by >1.2x on the shared
supported-scenario intersection. These represent areas where the JAX-native
architecture provides clear advantages on directly comparable workloads.

### Comparable Performance

Scenarios where performance is within 0.8x-1.2x of the closest alternative.

### Optimization Opportunities

Scenarios where other frameworks lead. Each gap is mapped to a prioritized optimization target. The gap detector generates an optimization backlog:

```python
from pathlib import Path

from benchmarks.analysis.gap_detection import GapDetector
from benchmarks.runners.full_runner import ComparativeResults

results = ComparativeResults.load(Path("benchmark-data/reports/latest"))
detector = GapDetector(results)
detector.generate_backlog(Path("benchmark-data/optimization_backlog.md"))
```

---

## Viewing Results

### W&B Dashboard

After running benchmarks, export to W&B for interactive exploration:

```bash
export WANDB_API_KEY="..."
calibrax export --data benchmark-data/
```

See [Dashboard & calibrax](dashboard.md) for setup details.

### Terminal Summary

For a quick local overview without W&B:

```bash
calibrax summary --data benchmark-data/
```
