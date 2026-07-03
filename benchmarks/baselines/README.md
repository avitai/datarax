# Benchmark Baselines

Datarax-adapter baselines in the calibrax result schema, one JSON per
scenario variant, named `{scenario_id}_{variant}.json`. Each file records
the environment it was measured on (GPU model, driver, JAX version); the
throughput regression guard only compares against baselines whose GPU
model matches the local machine.

Layout:

- `*.json` — local development baselines (NVIDIA GeForce RTX 4090,
  `gpu_rtx4090` profile). Regenerate with
  `python -m benchmarks.runners.benchmark_runner --generate-baselines
  --profile gpu_rtx4090 --baselines-dir benchmarks/baselines --force`.
- `gpu_a100/` — cloud baselines (A100, `gpu_a100` profile), exported from
  the automated Vast two-pass run's full-stage datarax results. Use with
  `--baselines-dir benchmarks/baselines/gpu_a100` on A100 hosts.

`BaselineStore` validates every file against the calibrax schema on load
and rejects legacy shapes with a regeneration hint.
