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

Each adapter supports only the scenarios where it implements the required transforms. This ensures fair comparisons — every framework performs equivalent work on each scenario.

| Framework | CV-1 | NLP-1 | TAB-1 | MM-1 | DIST-1 | PR-1 | Total |
|-----------|:----:|:-----:|:-----:|:----:|:------:|:----:|:-----:|
| **Datarax** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | **25** |
| Google Grain | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | 5 |
| tf.data | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | 5 |
| PyTorch DataLoader | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | :white_check_mark: | 5 |
| NVIDIA DALI | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | | 4 |
| SPDL | :white_check_mark: | :white_check_mark: | :white_check_mark: | | :white_check_mark: | | 4 |
| MosaicML Streaming | :white_check_mark: | :white_check_mark: | | | | | 2 |
| WebDataset | :white_check_mark: | :white_check_mark: | | | | | 2 |
| HuggingFace Datasets | | :white_check_mark: | :white_check_mark: | | | | 2 |
| Ray Data | | :white_check_mark: | :white_check_mark: | | | | 2 |
| jax-dataloader | :white_check_mark: | | | | | | 1 |
| FFCV | :white_check_mark: | | | | | | 1 |
| LitData | :white_check_mark: | | | | | | 1 |
| Deep Lake | :white_check_mark: | | | | | | 1 |
| Energon | | | | :white_check_mark: | | | 1 |

!!! note "Datarax supports all 25 scenarios"
    The table above shows the 6 most widely supported scenarios. Datarax uniquely supports all 25 scenarios including PC-1 through PC-5 (pipeline complexity), IO-1 through IO-4 (I/O patterns), and NNX-1/XFMR-1 (Datarax-unique features).

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

### Strengths

Scenarios where Datarax leads other frameworks by >1.2x. These represent areas where the JAX-native architecture provides clear advantages:

-   **Pipeline Complexity (PC-\*)**: Datarax's DAG execution engine handles complex multi-branch pipelines that other frameworks cannot express
-   **Datarax Unique (NNX-1, XFMR-1)**: Features like Flax NNX module integration and JIT+vmap transform acceleration are exclusive to Datarax

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
benchkit export --data benchmark-data/
```

See [Dashboard & benchkit](dashboard.md) for setup details.

### Terminal Summary

For a quick local overview without W&B:

```bash
benchkit summary --data benchmark-data/
```
