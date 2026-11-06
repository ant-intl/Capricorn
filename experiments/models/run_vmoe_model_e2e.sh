#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CAPRICORN_JAX_PYTHON:-python3}" -m capricorn_models.run_vmoe_e2e "$@"
