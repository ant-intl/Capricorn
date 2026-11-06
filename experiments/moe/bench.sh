#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
usage() {
  cat <<EOF
Usage: bash $0 [--dry-run] [OUTPUT_ROOT]
Paper routing and MoE-layer experiments. Configuration uses environment variables:
  TOKEN_COUNTS="16 32 64"  CAPRICORN_NETWORKS=lan,wan  RUNS=5
  IMPLEMENTATIONS=cryptomoe_serial,cryptomoe_parallel,capricorn
  ROUTING_ONLY=0           Set 1 for routing-only measurements
  EMBEDDING_DIMENSION=2048 EXPERT_DIMENSION=1408 EXPERT_NUMS=64 TOP_K=6
  COT_DATA_DIR=PATH        CAPRICORN_SKIP_BUILD=1      TIMEOUT_SECONDS=3600
  CAPRICORN_CRYPTO_MOE_BIN=PATH  CAPRICORN_MOE_BIN=PATH
EOF
}

dry_run=0
output_root=""
for arg in "$@"; do
  case "${arg}" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=1 ;;
    -*) usage >&2; exit 2 ;;
    *) [[ -z "${output_root}" ]] || { usage >&2; exit 2; }; output_root=${arg} ;;
  esac
done
runs=${RUNS:-5}
networks=${CAPRICORN_NETWORKS:-lan,wan}
implementations=${IMPLEMENTATIONS:-cryptomoe_serial,cryptomoe_parallel,capricorn}
routing_only=${ROUTING_ONLY:-0}
embedding=${EMBEDDING_DIMENSION:-2048}
expert_dimension=${EXPERT_DIMENSION:-1408}
experts=${EXPERT_NUMS:-64}
top_k=${TOP_K:-6}
timeout_seconds=${TIMEOUT_SECONDS:-3600}
read -r -a tokens <<< "${TOKEN_COUNTS:-16 32 64}"
IFS=, read -r -a profiles <<< "${networks}"
IFS=, read -r -a variants <<< "${implementations}"
for value in "${runs}" "${embedding}" "${expert_dimension}" "${experts}" "${top_k}" "${timeout_seconds}" "${tokens[@]}"; do
  [[ ${value} =~ ^[1-9][0-9]*$ ]] || { echo 'Dimensions, runs and timeout must be positive integers' >&2; exit 2; }
done
[[ ${routing_only} == 0 || ${routing_only} == 1 ]] || { echo 'ROUTING_ONLY must be 0 or 1' >&2; exit 2; }
((top_k <= experts)) || { echo 'TOP_K exceeds EXPERT_NUMS' >&2; exit 2; }
shaped=0
for network in "${profiles[@]}"; do
  case "${network}" in native) ;; lan|wan) shaped=1 ;; *) echo "Unsupported network: ${network}" >&2; exit 2 ;; esac
done
targets=()
for implementation in "${variants[@]}"; do
  case "${implementation}" in
    cryptomoe_serial|cryptomoe_parallel) targets+=(//benchmarks:cryptomoe_bench) ;;
    capricorn) targets+=(//benchmarks:capricorn_bench) ;;
    *) echo "Unsupported implementation: ${implementation}" >&2; exit 2 ;;
  esac
done
[[ ${#tokens[@]} -gt 0 && ${#profiles[@]} -gt 0 && ${#variants[@]} -gt 0 ]] || { echo 'Experiment selection must be nonempty' >&2; exit 2; }
output_root=${output_root:-${repo_root}/results/moe_layer_$(date +%Y%m%d_%H%M%S)}
[[ ${output_root} == /* ]] || output_root="${PWD}/${output_root}"
cot_data_dir=${COT_DATA_DIR:-${repo_root}/capricorn/internal/cot_data}
[[ ${cot_data_dir} == /* ]] || cot_data_dir="${PWD}/${cot_data_dir}"
crypto_bin=${CAPRICORN_CRYPTO_MOE_BIN:-${repo_root}/bazel-bin/benchmarks/cryptomoe_bench}
capricorn_bin=${CAPRICORN_MOE_BIN:-${repo_root}/bazel-bin/benchmarks/capricorn_bench}

if [[ ${dry_run} == 0 ]]; then
  if [[ -d "${output_root}" && -n "$(ls -A -- "${output_root}")" ]]; then
    echo "Output directory is not empty: ${output_root}" >&2; exit 2
  fi
  if [[ ${CAPRICORN_SKIP_BUILD:-0} != 1 ]]; then
    (cd -- "${repo_root}" && bazelisk build -c opt "${targets[@]}")
  fi
  for implementation in "${variants[@]}"; do
    binary=${crypto_bin}; [[ ${implementation} != capricorn ]] || binary=${capricorn_bin}
    [[ -x "${binary}" ]] || { echo "Benchmark is not executable: ${binary}" >&2; exit 2; }
  done
  command -v timeout >/dev/null || { echo 'GNU timeout is required' >&2; exit 2; }
  if [[ ${shaped} == 1 && ${CAPRICORN_MOE_NAMESPACE:-0} != 1 ]]; then
    exec unshare --user --map-root-user --net env CAPRICORN_MOE_NAMESPACE=1 \
      CAPRICORN_MOE_PARENT_NETNS="$(readlink /proc/self/ns/net)" CAPRICORN_SKIP_BUILD=1 \
      COT_DATA_DIR="${cot_data_dir}" bash "${repo_root}/experiments/moe/bench.sh" "${output_root}"
  fi
  if [[ ${shaped} == 1 ]]; then
    own_namespace=$(readlink /proc/self/ns/net)
    [[ -n ${CAPRICORN_MOE_PARENT_NETNS:-} && ${own_namespace} != "${CAPRICORN_MOE_PARENT_NETNS}" ]] || {
      echo 'Refusing network shaping outside the child namespace' >&2; exit 2;
    }
    ip link set lo up
  fi
  mkdir -p -- "${output_root}" "${cot_data_dir}"
fi

party0=""; party1=""
cleanup() {
  for pid in "${party0}" "${party1}"; do
    if [[ -n ${pid} ]]; then kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

moe_task() {
  local network=$1 implementation=$2 token_count=$3 run=$4
  local binary=${crypto_bin}
  local -a extra_args=()
  case "${implementation}" in
    cryptomoe_serial) extra_args+=(--expert_top_t_mode=serial) ;;
    cryptomoe_parallel) extra_args+=(--expert_top_t_mode=parallel) ;;
    capricorn) binary=${capricorn_bin} ;;
  esac
  local case_dir="${output_root}/${network}/${implementation}/tokens_${token_count}/run_${run}"
  local -a args=(--parties=127.0.0.1:10086,127.0.0.1:10087 --token_nums="${token_count}"
    --embedding_dimension="${embedding}" --expert_nums="${experts}" --expert_dimension="${expert_dimension}"
    --active_expert_nums="${top_k}" --capacity_factor=2 --query_count=1 --routing_only="${routing_only}"
    ${extra_args[@]+"${extra_args[@]}"})
  if [[ ${dry_run} == 1 ]]; then
    printf '%s %s tokens=%s run=%s\n' "${network}" "${implementation}" "${token_count}" "${run}"
    printf '%q ' "${binary}" --role=true "${args[@]}"; printf '\n'
    printf '%q ' "${binary}" --role=false "${args[@]}"; printf '\n'
    return
  fi
  mkdir -p -- "${case_dir}"
  printf '%q ' "${binary}" "${args[@]}" >"${case_dir}/command.txt"
  printf '\n' >>"${case_dir}/command.txt"
  COT_DATA_DIR="${cot_data_dir}" RUNFILES_DIR="${binary}.runfiles" timeout --kill-after=5 "${timeout_seconds}" "${binary}" --role=true "${args[@]}" >"${case_dir}/server.log" 2>&1 &
  party0=$!
  COT_DATA_DIR="${cot_data_dir}" RUNFILES_DIR="${binary}.runfiles" timeout --kill-after=5 "${timeout_seconds}" "${binary}" --role=false "${args[@]}" >"${case_dir}/client.log" 2>&1 &
  party1=$!
  local status0=0 status1=0
  while kill -0 "${party0}" 2>/dev/null || kill -0 "${party1}" 2>/dev/null; do
    if ! kill -0 "${party0}" 2>/dev/null && ! wait "${party0}"; then kill "${party1}" 2>/dev/null || true; break; fi
    if ! kill -0 "${party1}" 2>/dev/null && ! wait "${party1}"; then kill "${party0}" 2>/dev/null || true; break; fi
    sleep 0.05
  done
  wait "${party0}" || status0=$?
  wait "${party1}" || status1=$?
  party0=""; party1=""
  printf 'rank0=%s rank1=%s\n' "${status0}" "${status1}" >"${case_dir}/exit_status"
  if [[ ${status0} != 0 || ${status1} != 0 ]]; then echo "FAILED: ${case_dir}" >&2; return 1; fi
  echo "DONE: ${case_dir}"
}

for network in "${profiles[@]}"; do
  rate=""; delay=""
  case "${network}" in lan) rate=3gbit; delay=0.1ms ;; wan) rate=400mbit; delay=20ms ;; esac
  if [[ ${dry_run} == 0 ]]; then
    if [[ ${network} == native && ${shaped} == 1 ]]; then
      tc qdisc del dev lo root 2>/dev/null || true
    elif [[ ${network} != native ]]; then
      tc qdisc replace dev lo root netem rate "${rate}" delay "${delay}"
    fi
    printf 'network=%s rate=%s one_way_delay=%s\n' "${network}" "${rate}" "${delay}" >>"${output_root}/network_profiles.txt"
  fi
  for token_count in "${tokens[@]}"; do
    for implementation in "${variants[@]}"; do
      for ((run=1; run<=runs; run++)); do moe_task "${network}" "${implementation}" "${token_count}" "${run}"; done
    done
  done
done
