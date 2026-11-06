#!/bin/bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: MODEL_PATH=PATH PYTHON_BIN=PYTHON BATCH_SIZE=64 FULL=1 OUTPUT=PATH bash $0"
  echo "Defaults match the paper: DeepSeekMoE, batch 64, six complete datasets. FULL=0 LIMIT=4 selects a smoke run."
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Use --help for experiment configuration" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/deepseek-moe-16b-base}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LIMIT="${LIMIT:-4}"
FULL="${FULL:-1}"
FORCE="${FORCE:-0}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/routing_fidelity_results.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 1
fi

run_args=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/routing_fidelity.py"
  --model "${MODEL_PATH}"
  --batch-size "${BATCH_SIZE}"
  --output "${OUTPUT}"
  --datasets
    openbookqa
    arc_easy
    arc_challenge
    hellaswag
    piqa
    winogrande
)

if [[ "${FULL}" == "1" ]]; then
  run_args+=(--full)
elif [[ "${FULL}" == "0" ]]; then
  run_args+=(--limit "${LIMIT}")
else
  echo "FULL must be 0 or 1" >&2
  exit 1
fi

if [[ "${FORCE}" == "1" ]]; then
  run_args+=(--force)
elif [[ "${FORCE}" != "0" ]]; then
  echo "FORCE must be 0 or 1" >&2
  exit 1
fi

"${run_args[@]}"
