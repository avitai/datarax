#!/usr/bin/env bash
# Run full comparative benchmark suite via the datarax-bench CLI.
#
# Usage:
#   ./scripts/run_full_benchmark.sh           # CPU (default)
#   ./scripts/run_full_benchmark.sh gpu        # GPU
#   ./scripts/run_full_benchmark.sh tpu        # TPU
set -euo pipefail

PLATFORM="${1:-cpu}"
REPETITIONS="${2:-5}"

echo "Running full comparative benchmark: platform=${PLATFORM}, repetitions=${REPETITIONS}"

uv run datarax-bench run \
    --platform "${PLATFORM}" \
    --repetitions "${REPETITIONS}" \
    --profile ci_cpu

echo "Benchmark complete."
