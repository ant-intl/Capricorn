#!/usr/bin/env bash
set -euo pipefail

mode=$1
bazel=$2
target=//src/common/mpc:rot_context_test
umask 077
mkdir -p .cache/cot-smoke-v1
cache_root=$(cd .cache/cot-smoke-v1 && pwd -P)
files=(server_send0.txt server_receive0.txt client_send0.txt client_receive0.txt)

checksum() (
  cd "$1" || exit 1
  for file in "${files[@]}"; do
    [[ -f $file && ! -L $file && -s $file ]] || exit 1
  done
  shasum -a 256 "${files[@]}"
)

run_rot() {
  local dir=$1
  shift
  "$bazel" test -c opt --jobs=2 --nocache_test_results \
    --test_env="COT_DATA_DIR=$dir" --sandbox_writable_path="$dir" "$@" "$target"
}

generation=
if [[ -f $cache_root/current && ! -L $cache_root/current ]]; then
  read -r generation < "$cache_root/current" || generation=
fi
dir=$cache_root/$generation
if [[ $generation =~ ^generation\.[a-zA-Z0-9]+$ && -f $dir/SHA256SUMS ]] &&
   sums=$(checksum "$dir") && [[ $sums == "$(<"$dir/SHA256SUMS")" ]]; then
  printf 'Reusing ROT smoke-test COT cache: %s\n' "$dir"
else
  dir=$(mktemp -d "$cache_root/generation.XXXXXX")
  printf 'Initializing ROT smoke-test COT cache: %s\n' "$dir"
  run_rot "$dir" --test_filter=RotContextTest.ValidateROT
  checksum "$dir" > "$dir/SHA256SUMS"

  # Publish only a complete generation. Keep previous generations untouched,
  # including any generation another smoke test may still be using.
  pointer=$(mktemp "$cache_root/.current.XXXXXX")
  printf '%s\n' "${dir##*/}" > "$pointer"
  mv -f "$pointer" "$cache_root/current"
fi

if [[ $mode == test ]]; then
  run_rot "$dir"
fi
