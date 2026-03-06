#!/usr/bin/env bash
# Run full comparative benchmark suite via the benchmarks CLI module.
#
# Usage:
#   ./scripts/run_full_benchmark.sh           # CPU (default)
#   ./scripts/run_full_benchmark.sh gpu        # GPU
#   ./scripts/run_full_benchmark.sh tpu        # TPU
#   ./scripts/run_full_benchmark.sh gpu 5 gpu_a100  # Custom profile override
set -euo pipefail

PLATFORM="${1:-cpu}"
REPETITIONS="${2:-5}"
PROFILE="${3:-}"

if [[ -z "${PROFILE}" ]]; then
  case "${PLATFORM}" in
    cpu) PROFILE="ci_cpu" ;;
    gpu) PROFILE="gpu_rtx4090" ;;
    tpu) PROFILE="tpu_v5e" ;;
    *)
      echo "Unsupported platform: ${PLATFORM}" >&2
      exit 1
      ;;
  esac
fi

echo "Running full comparative benchmark: platform=${PLATFORM}, profile=${PROFILE}, repetitions=${REPETITIONS}"

uv run python -m benchmarks.cli run \
    --platform "${PLATFORM}" \
    --repetitions "${REPETITIONS}" \
    --profile "${PROFILE}"

echo "Benchmark complete."
