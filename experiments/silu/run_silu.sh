#!/usr/bin/env bash

set -euo pipefail

accuracy=0
if [[ ${1:-} == --accuracy ]]; then accuracy=1; shift; fi

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: RUNS=5 CAPRICORN_NETWORKS=lan,wan COT_DATA_DIR=PATH bash $0 [--accuracy] OUTPUT_ROOT"
  echo "CAPRICORN_NETWORKS accepts lan, wan, or lan,wan; OUTPUT_ROOT must be empty."
  echo "--accuracy runs the existing plaintext fixed-point approximation reference for ranges 8, 32 and 64, without MPC or network shaping."
  exit 0
fi

if [[ $# -ne 1 || $1 == -* ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
[[ "${output_root}" == /* ]] || output_root="${PWD}/${output_root}"
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runs=${RUNS:-5}
networks=${CAPRICORN_NETWORKS:-lan,wan}
if ! [[ ${runs} =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS must be a positive integer" >&2
  exit 2
fi
case "${networks}" in
  lan) profiles=('lan 3gbit 0.1ms') ;;
  wan) profiles=('wan 400mbit 20ms') ;;
  lan,wan) profiles=('lan 3gbit 0.1ms' 'wan 400mbit 20ms') ;;
  *)
    echo "Unsupported CAPRICORN_NETWORKS: ${networks}" >&2
    exit 2
    ;;
esac
cot_data_dir=${COT_DATA_DIR:-"${repo_root}/capricorn/internal/cot_data"}

if [[ -d "${output_root}" && -n "$(ls -A -- "${output_root}")" ]]; then
  echo "Output directory is not empty: ${output_root}" >&2; exit 2
fi
if [[ ${accuracy} == 1 ]]; then
  mkdir -p -- "${output_root}"
  "${CXX:-c++}" -O2 -std=c++17 "${repo_root}/plaintext/activation/SiLU_a_12_d_18.cpp" -o "${output_root}/silu_accuracy"
  for exponent in 18 20 22; do
    for input_range in 8 32 64; do
      "${output_root}/silu_accuracy" "$((1 << exponent))" "${input_range}" "${runs}" >"${output_root}/accuracy_2p${exponent}_range${input_range}.log"
    done
  done
  exit 0
fi

baseline_bin="${CAPRICORN_SILU_BASELINE_BIN:-${repo_root}/bazel-bin/src/common/nn/spu_activation_test}"
secure_bin="${CAPRICORN_SILU_SECURE_BIN:-${repo_root}/bazel-bin/src/capricorn/activation/secure_silu_test}"

if [[ ! -x "${baseline_bin}" || ! -x "${secure_bin}" ]]; then
  echo "Build first: bazelisk build -c opt //src/common/nn:spu_activation_test //src/capricorn/activation:secure_silu_test" >&2
  exit 2
fi

if [[ ${CAPRICORN_NETEM_NAMESPACE:-0} != 1 ]]; then
  exec unshare --user --map-root-user --net env \
    CAPRICORN_NETEM_NAMESPACE=1 CAPRICORN_SILU_PARENT_NETNS="$(readlink /proc/self/ns/net)" \
    RUNS="${runs}" CAPRICORN_NETWORKS="${networks}" COT_DATA_DIR="${cot_data_dir}" \
    bash "$0" "${output_root}"
fi

[[ -n ${CAPRICORN_SILU_PARENT_NETNS:-} && $(readlink /proc/self/ns/net) != "${CAPRICORN_SILU_PARENT_NETNS}" ]] || {
  echo "Refusing network shaping outside the child namespace" >&2
  exit 2
}
mkdir -p "${output_root}" "${cot_data_dir}"
ip link set lo up

write_network_profile() {
  local path=$1 name=$2 rate=$3 delay=$4
  cat >"${path}" <<EOF
{"name":"${name}","rate":"${rate}","one_way_delay":"${delay}","interface":"lo","isolation":"unshare --user --map-root-user --net"}
EOF
}

run_case() {
  local network=$1 rate=$2 delay=$3 implementation=$4 exponent=$5 run=$6
  local case_dir="${output_root}/${network}/${implementation}/2p${exponent}/run_$(printf '%03d' "${run}")"
  local status_file="${case_dir}/exit_status"

  mkdir -p "${case_dir}"
  write_network_profile "${case_dir}/network_profile.json" "${network}" "${rate}" "${delay}"
  tc -s qdisc show dev lo >"${case_dir}/tc_qdisc_before.txt"

  local index=$(((exponent - 18) / 2))
  local -a command
  if [[ ${implementation} == silu ]]; then
    command=("${baseline_bin}" --gtest_also_run_disabled_tests
      "--gtest_filter=DISABLED_Table3/SiLUTest.Work/${index}")
  else
    command=(env "COT_DATA_DIR=${cot_data_dir}" "${secure_bin}"
      --gtest_also_run_disabled_tests
      "--gtest_filter=DISABLED_Table3/SecureSiluTest.RandomDataTest/${index}")
  fi

  echo "RUN  ${network} ${implementation} 2^${exponent} run ${run}"
  set +e
  /usr/bin/time -v "${command[@]}" >"${case_dir}/test.log" 2>"${case_dir}/time.log"
  local status=$?
  set -e
  printf '%s\n' "${status}" >"${status_file}"
  tc -s qdisc show dev lo >"${case_dir}/tc_qdisc_after.txt"
  if [[ ${status} -ne 0 ]]; then
    echo "FAILED ${network} ${implementation} 2^${exponent} run ${run}" >&2
    exit "${status}"
  fi
}

for profile in "${profiles[@]}"; do
  read -r network rate delay <<<"${profile}"
  tc qdisc replace dev lo root netem rate "${rate}" delay "${delay}"
  for exponent in 18 20 22; do
    for implementation in silu secure_silu; do
      for ((run = 1; run <= runs; ++run)); do
        run_case "${network}" "${rate}" "${delay}" "${implementation}" "${exponent}" "${run}"
      done
    done
  done
done

echo "All SiLU experiments completed: ${output_root}"
