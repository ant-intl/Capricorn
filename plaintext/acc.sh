#!/bin/bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  cat <<EOF
Usage: bash $0
Accuracy experiments use environment variables:
  MODEL=deepseek-moe-16b-base|Qwen1.5-MoE-A2.7B|OLMoE-1B-7B-0125-Instruct
  MODE=original|float-all-experts|capricorn-piecewise-fixed|capricorn-polynomial-fixed|cryptomoe-piecewise-fixed|cryptomoe-polynomial-fixed
  MODEL_PATH=PATH PYTHON_BIN=PYTHON LOG_DIR=PATH
  BATCH_SIZES_OVERRIDE="16 64" DATASETS_OVERRIDE="piqa arc_easy"
EOF
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Use --help for experiment configuration" >&2; exit 2; }

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-OLMoE-1B-7B-0125-Instruct}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/${MODEL}}"
MODE="${MODE:-capricorn-piecewise-fixed}"
ROUTING_TRACE_DIR="${ROUTING_TRACE_DIR:-}"
ROUTING_REFERENCE_DIR="${ROUTING_REFERENCE_DIR:-}"
FXP_STATS="${FXP_STATS:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/exms}"
RESET_LOGS="${RESET_LOGS:-0}"
# Avoid fragmentation when batch-64 evaluation materializes multi-GiB logits.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

case "${MODE}" in
  original|float-all-experts|\
capricorn-piecewise-fixed|capricorn-polynomial-fixed|\
cryptomoe-piecewise-fixed|cryptomoe-polynomial-fixed)
    ;;
  fixed-fast)
    echo "fixed-fast is deprecated; use capricorn-piecewise-fixed instead" >&2
    exit 1
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac
if [[ "${FXP_STATS}" != "0" && "${FXP_STATS}" != "1" ]]; then
  echo "FXP_STATS must be 0 or 1" >&2
  exit 1
fi
if [[ "${RESET_LOGS}" != "0" && "${RESET_LOGS}" != "1" ]]; then
  echo "RESET_LOGS must be 0 or 1" >&2
  exit 1
fi
if [[ "${MODE}" == "original" || "${MODE}" == "float-all-experts" ]]; then
  if [[ "${FXP_STATS}" != "0" ]]; then
    echo "${MODE} mode forbids FXP_STATS" >&2
    exit 1
  fi
  if [[ -n "${ROUTING_TRACE_DIR}" || -n "${ROUTING_REFERENCE_DIR}" ]]; then
    echo "${MODE} mode forbids fixed-point routing traces" >&2
    exit 1
  fi
fi

# 数据集列表；失败恢复时允许只重跑指定数据集。
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
else
  DATASETS=(
    "openbookqa"
    "arc_easy"
    "arc_challenge"
    "hellaswag"
    "piqa"
    "winogrande"
  )
fi
if (( ${#DATASETS[@]} == 0 )); then
  echo "DATASETS_OVERRIDE must contain at least one dataset" >&2
  exit 1
fi
for dataset in "${DATASETS[@]}"; do
  case "${dataset}" in
    openbookqa|arc_easy|arc_challenge|hellaswag|piqa|winogrande)
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
done

# All experiment profiles use the checkpoint k_base for capacity:
# ceil(tokens * k_base * 2 / total_experts).  Capricorn activates all
# experts; CryptoMoE routes only to k_base experts. float-all-experts changes
# only the native floating-point router top-k and keeps the same capacity k.
case "${MODEL}" in
  deepseek-moe-16b-base)
    TOTAL_EXPERTS=64
    BASE_EXPERTS=6
    ;;
  Qwen1.5-MoE-A2.7B)
    TOTAL_EXPERTS=60
    BASE_EXPERTS=4
    ;;
  OLMoE-1B-7B-0125-Instruct)
    TOTAL_EXPERTS=64
    BASE_EXPERTS=8
    ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  original)
    EXPERT_VALUES=("${BASE_EXPERTS}")
    ACTIVATIONS=("silu")
    LOG_PROFILE="${MODEL}_original_silu"
    ;;
  float-all-experts)
    EXPERT_VALUES=("${TOTAL_EXPERTS}")
    ACTIVATIONS=("silu")
    LOG_PROFILE="${MODEL}_float_all_experts_silu"
    ;;
  capricorn-piecewise-fixed)
    EXPERT_VALUES=("${TOTAL_EXPERTS}")
    ACTIVATIONS=("silu_piecewise_fixed_16_-8_8")
    LOG_PROFILE="${MODEL}_capricorn_piecewise_fixed_silu_16_-8_8"
    ;;
  capricorn-polynomial-fixed)
    EXPERT_VALUES=("${TOTAL_EXPERTS}")
    ACTIVATIONS=("silu_poly_approx")
    LOG_PROFILE="${MODEL}_capricorn_polynomial_fixed_silu_poly_approx"
    ;;
  cryptomoe-piecewise-fixed)
    EXPERT_VALUES=("${BASE_EXPERTS}")
    ACTIVATIONS=("silu_piecewise_fixed_16_-8_8")
    LOG_PROFILE="${MODEL}_cryptomoe_piecewise_fixed_silu_16_-8_8"
    ;;
  cryptomoe-polynomial-fixed)
    EXPERT_VALUES=("${BASE_EXPERTS}")
    ACTIVATIONS=("silu_poly_approx")
    LOG_PROFILE="${MODEL}_cryptomoe_polynomial_fixed_silu_poly_approx"
    ;;

esac

# batch_size 的两个默认值；失败恢复时允许只重跑指定 batch。
if [[ -n "${BATCH_SIZES_OVERRIDE:-}" ]]; then
  read -r -a BATCH_SIZES <<< "${BATCH_SIZES_OVERRIDE}"
else
  BATCH_SIZES=(16 64)
fi
if (( ${#BATCH_SIZES[@]} == 0 )); then
  echo "BATCH_SIZES_OVERRIDE must contain at least one batch size" >&2
  exit 1
fi
for batch_size in "${BATCH_SIZES[@]}"; do
  case "${batch_size}" in
    16|64)
      ;;
    *)
      echo "Unsupported batch size: ${batch_size}" >&2
      exit 1
      ;;
  esac
done

# 创建日志目录
mkdir -p "${LOG_DIR}"
if [[ -n "${ROUTING_TRACE_DIR}" ]]; then
  mkdir -p "${ROUTING_TRACE_DIR}"
fi

# 循环 激活函数
for activation in "${ACTIVATIONS[@]}"; do
  echo "------------------------------------------------------------------------------------------"
  echo "Running tests with activation=${activation}"
  echo "------------------------------------------------------------------------------------------"

  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

  # 循环 batch size
  for batch_size in "${BATCH_SIZES[@]}"; do
    echo "------------------------------------------------------------------------------------------"
    echo "Running tests with batch_size=${batch_size} and activation=${activation}"
    echo "------------------------------------------------------------------------------------------"

    LOGFILE="${LOG_DIR}/${LOG_PROFILE}_batch_${batch_size}_acc.txt"
    if [[ "${RESET_LOGS}" == "1" ]]; then
      : > "${LOGFILE}"
    fi

    {
      echo "==================== Model: ${MODEL}, mode: ${MODE}, activation: ${activation}, batch_size: ${batch_size} ===================="
    } >> "$LOGFILE" 2>&1

    # 循环执行 expert values
    for n_val in "${EXPERT_VALUES[@]}"; do
      {
        echo "=========================================================================================="
        echo "Running tests with num_experts_per_tok=${n_val}, batch_size=${batch_size}, and activation=${activation}"
        echo "=========================================================================================="
      } >> "$LOGFILE" 2>&1

      # 循环数据集
      for dataset in "${DATASETS[@]}"; do
        {
          echo ">>> Testing dataset: ${dataset} with batch_size=${batch_size}, activation=${activation}, experts=${n_val}"
        } >> "$LOGFILE" 2>&1

        model_args=(
          --model "${MODEL_PATH}"
          --dataset "${dataset}"
          --batch_size "${batch_size}"
          --mode "${MODE}"
        )
        if [[ "${FXP_STATS}" == "1" ]]; then
          model_args+=(--fxp_stats)
        fi
        if [[ "${MODE}" != "original" ]]; then
          model_args+=(
            --num_experts_per_tok "${n_val}"
            --hidden_act "${activation}"
          )
          trace_name="${MODEL}_${dataset}_batch_${batch_size}.pt"
          if [[ -n "${ROUTING_TRACE_DIR}" ]]; then
            model_args+=(
              --routing_trace_out "${ROUTING_TRACE_DIR}/${trace_name}"
            )
          fi
          if [[ -n "${ROUTING_REFERENCE_DIR}" ]]; then
            model_args+=(
              --routing_trace_reference \
              "${ROUTING_REFERENCE_DIR}/${trace_name}"
            )
          fi
        fi

        if "${PYTHON_BIN}" "${SCRIPT_DIR}/test.py" "${model_args[@]}" 2>&1 | tee -a "$LOGFILE"; then
          echo "✅ ${dataset} finished successfully with batch_size=${batch_size}, activation=${activation}, experts=${n_val}" >> "$LOGFILE" 2>&1
        else
          echo "❌ ${dataset} failed with batch_size=${batch_size}, activation=${activation}, experts=${n_val}" >> "$LOGFILE" 2>&1
          exit 1
        fi
      done
    done
  done
done

echo "All tests completed."
