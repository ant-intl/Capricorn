.DEFAULT_GOAL := help

BAZEL ?= bazelisk
PYTHON_BIN ?= python3
export PYTHON_BIN

.PHONY: help build cot_init test moe-plan bench-routing bench-moe bench-silu \
        silu-summary silu-accuracy vmoe-single vmoe-batch plaintext-accuracy \
        plaintext-summary routing-fidelity

help:
	@printf '%s\n' \
	  'Development:' \
	  '  make build             Build the five experiment binaries (Bazelisk).' \
	  '  make cot_init          Check or initialize the local ROT smoke-test cache.' \
	  '  make test              Run the lightweight ROT tests; not the full suite.' \
	  '' \
	  'MPC benchmarks (see README for Linux tools and build prerequisites):' \
	  '  make moe-plan          Preview the routing/MoE matrix without running it.' \
	  '  make bench-routing RUNS=1 CAPRICORN_NETWORKS=lan TOKEN_COUNTS=16' \
	  '  make bench-moe RUNS=1 CAPRICORN_NETWORKS=lan TOKEN_COUNTS=16' \
	  '  make bench-silu RUNS=1 CAPRICORN_NETWORKS=lan' \
	  '  make silu-summary [OUTPUT=results/silu]' \
	  '' \
	  'V-MoE inference (prepare artifacts first; LAN/WAN require Linux):' \
	  '  make vmoe-single ARTIFACT_DIR=path/to/artifact [NETWORK=native]' \
	  '  make vmoe-batch MANIFEST=path/to/manifest.json ARTIFACT_ROOT=path/to/artifacts [NETWORK=native]' \
	  '  See README for checkpoint export, Imagenette preparation and batch summary.' \
	  '' \
	  'Plaintext evaluation (accuracy and fidelity require local models/datasets):' \
	  '  make silu-accuracy [RUNS=1]' \
	  '  make plaintext-accuracy MODEL_PATH=path/to/model' \
	  '  make plaintext-summary [LOG_DIR=results/plaintext_accuracy]' \
	  '  make routing-fidelity MODEL_PATH=path/to/model FULL=0 LIMIT=4' \
	  '' \
	  'Set OUTPUT=path to override an experiment output; see README for other options.'

build:
	"$(BAZEL)" build -c opt \
	  //benchmarks:capricorn_bench \
	  //benchmarks:cryptomoe_bench \
	  //src/common/nn:spu_activation_test \
	  //src/capricorn/activation:secure_silu_test \
	  //src/adapters/vmoe:pphlo_runner_main

cot_init:
	@bash tools/rot_smoke.sh init "$(BAZEL)"

test:
	@bash tools/rot_smoke.sh test "$(BAZEL)"

moe-plan:
	bash experiments/moe/bench.sh --dry-run

bench-routing:
	ROUTING_ONLY=1 bash experiments/moe/bench.sh "$${OUTPUT:-results/routing}"

bench-moe:
	ROUTING_ONLY=0 bash experiments/moe/bench.sh "$${OUTPUT:-results/moe_layer}"

bench-silu:
	bash experiments/silu/run_silu.sh "$${OUTPUT:-results/silu}"

silu-summary:
	"$$PYTHON_BIN" experiments/silu/summarize_silu_table3.py "$${OUTPUT:-results/silu}"

silu-accuracy:
	bash experiments/silu/run_silu.sh --accuracy "$${OUTPUT:-results/silu_accuracy}"

vmoe-single:
	@: "$${ARTIFACT_DIR:?Set ARTIFACT_DIR to an exported V-MoE artifact}"
	bash experiments/models/run_vmoe_model_e2e.sh --mode single \
	  --network "$${NETWORK:-native}" --artifact-dir "$$ARTIFACT_DIR" \
	  --run-dir "$${OUTPUT:-results/vmoe_single}"

vmoe-batch:
	@: "$${MANIFEST:?Set MANIFEST to the sample manifest}" \
	   "$${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to the prepared artifact tree}"
	bash experiments/models/run_vmoe_model_e2e.sh --mode batch \
	  --network "$${NETWORK:-native}" --routing "$${ROUTING:-both}" \
	  --manifest "$$MANIFEST" --artifact-root "$$ARTIFACT_ROOT" \
	  --run-dir "$${OUTPUT:-results/vmoe_batch}"

plaintext-accuracy:
	@: "$${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
	LOG_DIR="$${LOG_DIR:-$${OUTPUT:-results/plaintext_accuracy}}" bash plaintext/acc.sh

plaintext-summary:
	"$$PYTHON_BIN" plaintext/parse_results.py --log-dir "$${LOG_DIR:-$${OUTPUT:-results/plaintext_accuracy}}"

routing-fidelity:
	@: "$${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
	OUTPUT="$${OUTPUT:-results/routing_fidelity.json}" bash plaintext/run_routing_fidelity.sh
