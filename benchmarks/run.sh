#!/usr/bin/env bash
# AdapterSentry benchmark runner
#
# Usage:
#   ./benchmarks/run.sh                      # cold + warm, 100 adapters, 4 workers
#   ./benchmarks/run.sh --n 1000 --workers 8 # custom size
#   ./benchmarks/run.sh --scenario cold      # single scenario
#   ./benchmarks/run.sh --ci                 # CI mode: run + check regression
#
# Output:
#   benchmarks/results/current.json          — metrics from this run
#   benchmarks/results/baseline.json         — committed reference (do not delete)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/root/lora_env/bin/python3}"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
CURRENT="${RESULTS_DIR}/current.json"
BASELINE="${RESULTS_DIR}/baseline.json"

# Defaults
N=100
WORKERS=4
SCENARIO="both"
CI_MODE=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)        N="$2";        shift 2 ;;
        --workers)  WORKERS="$2";  shift 2 ;;
        --scenario) SCENARIO="$2"; shift 2 ;;
        --ci)       CI_MODE=true;  shift   ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"

echo "==> Running benchmark: n=${N} workers=${WORKERS} scenario=${SCENARIO}"

cd "${REPO_ROOT}"
"${PYTHON}" -m benchmarks.harness \
    --n "${N}" \
    --workers "${WORKERS}" \
    --scenario "${SCENARIO}" \
    --output "${CURRENT}" \
    --verbose

echo ""
echo "==> Metrics written to ${CURRENT}"

if "${CI_MODE}"; then
    if [[ ! -f "${BASELINE}" ]]; then
        echo "WARNING: No baseline found at ${BASELINE}. Copying current run as baseline."
        cp "${CURRENT}" "${BASELINE}"
        echo "Baseline created. Commit ${BASELINE} to lock it."
        exit 0
    fi

    echo "==> Checking regression against baseline..."
    "${PYTHON}" benchmarks/check_regression.py \
        --current  "${CURRENT}" \
        --baseline "${BASELINE}" \
        --factor   2.0
fi
