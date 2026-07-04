# Benchmark Coverage Matrix

Empirical per-adapter scenario coverage, measured from each adapter's
`supported_scenarios()` across the 37 benchmark scenarios. Counts for adapters
whose framework is not installed reflect their *declared* support (transform +
capability requirements), i.e. what they would run once installed.

## Per-adapter scenario coverage

| Adapter | Scenarios | Notes |
|---|---|---|
| Datarax (iter) | 37 | Full coverage — reference implementation |
| Datarax (scan) | 37 | Whole-epoch `nnx.scan` variant |
| Google Grain | 25 | JAX-native |
| tf.data | 25 | |
| PyTorch DataLoader | 25 | Universal baseline |
| SPDL | 25 | Thread-based loading |
| HuggingFace Datasets | 13 | framework not installed |
| jax-dataloader | 13 | framework not installed |
| FFCV | 13 | framework not installed; vision-focused |
| Deep Lake | 13 | framework not installed |
| NVIDIA DALI | 9 | framework not installed; GPU preprocessing |
| MosaicML Streaming | 2 | framework not installed; cloud streaming |
| WebDataset | 2 | framework not installed |
| Ray Data | 2 | framework not installed; distributed |
| LitData | 1 | framework not installed |
| Energon | 1 | framework not installed; multimodal |

## Competitive coverage

**25 of 37 scenarios run on ≥3 frameworks** — the set with meaningful
cross-framework comparison. Best-covered: **CV-1** and **NLP-1** (14 frameworks
each), **TAB-1** (12). Most vision/NLP/tabular/distributed scenarios sit at 6–11.

## Datarax-exclusive scenarios (12)

Only the Datarax variants run these — they exercise capabilities the
transform-based competitor adapters do not express, so they are datarax's
differentiators rather than head-to-head comparisons:

`CV-3` (batch mixing) · `CV-4` (multi-scale resize) · `IO-3` (mixed source) ·
`IO-4` (caching) · `NLP-2` (dynamic padding) · `NNX-1` (NNX overhead) ·
`PC-2` (branching DAG) · `PC-3` (differentiable rebatch) · `PC-4` (probabilistic) ·
`PC-5` (differentiable/learnable) · `PR-2` (determinism) · `XFMR-1` (JIT+vmap).

Reported comparisons must not present these as competitive wins — they are
capability coverage, not throughput contests.

## Per-scenario framework counts

```
AUG-1:7  AUG-2:7  AUG-3:7  CV-1:14  CV-2:10  CV-3:2  CV-4:2  DIST-1:11  DIST-2:10
HCV-1:7  HCV-2:6  HDIST-1:6  HMM-1:6  HNLP-1:6  HNLP-2:6  HPC-1:7  HPC-2:6
HTAB-1:6  IO-1:10  IO-2:10  IO-3:2  IO-4:2  MM-1:11  MM-2:10  NLP-1:14  NLP-2:2
NNX-1:2  PC-1:10  PC-2:2  PC-3:2  PC-4:2  PC-5:2  PR-1:10  PR-2:2  TAB-1:12
TAB-2:10  XFMR-1:2
```
