# Copyright 2026 Ant International, Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses reproducible V-MoE profiling logs and validates revealed outputs."""

import argparse
import json
import pathlib
import re

import numpy as np

_RECORD_PATTERN = re.compile(r"\[VMOE_(METRIC|STAGE|OPERATOR)\]\s+(.*)$")
_INTEGER_FIELDS = {
    "rank",
    "count",
    "sent_bytes",
    "sent_actions",
    "recv_bytes",
    "recv_actions",
    "current_rss_bytes",
    "peak_rss_bytes",
}
_FLOAT_FIELDS = {"elapsed_ms"}
_FULL_MODEL_MOE_TARGETS = (
    "spu.capricorn.vmoe_s32_e8_k2_gelu_bias.eval.v2",
    "spu.capricorn.vmoe_s32_e8_t28_gelu_bias.expert_choice.v2",
    "spu.capricorn.vmoe_s32_e8_k2_silu_bias.eval.v3",
    "spu.capricorn.vmoe_s32_e8_t28_secure_silu_bias.expert_choice.v3",
)


def _parse_log(path):
    records = {"METRIC": [], "STAGE": [], "OPERATOR": []}
    for line in pathlib.Path(path).read_text().splitlines():
        match = _RECORD_PATTERN.search(line)
        if not match:
            continue
        kind, payload = match.groups()
        record = {}
        for field in payload.split():
            key, value = field.split("=", 1)
            if key in _INTEGER_FIELDS:
                value = int(value)
            elif key in _FLOAT_FIELDS:
                value = float(value)
            record[key] = value
        records[kind].append(record)
    return records


def _top_indices(value, count=5):
    flattened = np.asarray(value).reshape(-1)
    count = min(count, flattened.size)
    return np.argsort(flattened)[-count:][::-1].astype(int).tolist()


def _rss_summary(records):
    snapshots = [
        record
        for kind in ("METRIC", "STAGE")
        for record in records[kind]
        if "peak_rss_bytes" in record
    ]
    if not snapshots:
        return {"recorded": False}
    return {
        "recorded": True,
        "peak_rss_bytes": max(record["peak_rss_bytes"] for record in snapshots),
        "max_current_rss_bytes": max(
            record["current_rss_bytes"] for record in snapshots
        ),
    }


def summarize_run(artifact_dir, run_dir):
    artifact_dir = pathlib.Path(artifact_dir)
    run_dir = pathlib.Path(run_dir)
    metadata_path = artifact_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    stablehlo = (artifact_dir / "model.stablehlo.mlir").read_text()
    logs = {str(rank): _parse_log(run_dir / f"rank{rank}.log") for rank in (0, 1)}

    expected = np.load(artifact_dir / "expected.npy", allow_pickle=False)
    rank0 = np.load(run_dir / "output_rank0.npy", allow_pickle=False)
    rank1 = np.load(run_dir / "output_rank1.npy", allow_pickle=False)
    absolute_error = np.abs(rank0 - expected)
    difference = rank0.astype(np.float64) - expected.astype(np.float64)
    expected_flat = expected.astype(np.float64).reshape(-1)
    rank0_flat = rank0.astype(np.float64).reshape(-1)
    cosine_denominator = np.linalg.norm(expected_flat) * np.linalg.norm(rank0_flat)

    return {
        "routing_mode": metadata.get("routing_mode"),
        "activation": metadata.get("activation"),
        "sample": metadata.get("sample"),
        "checkpoint": metadata.get("checkpoint"),
        "parameter_validation": metadata.get("parameters"),
        "preprocess": metadata.get("preprocess"),
        "instrumentation": {
            "marker_count": stablehlo.count("spu.capricorn.vmoe_trace."),
            "moe_custom_call_count": sum(
                stablehlo.count(target) for target in _FULL_MODEL_MOE_TARGETS
            ),
            "stage_count_by_rank": {
                rank: len(records["STAGE"]) for rank, records in logs.items()
            },
            "operator_aggregate_count_by_rank": {
                rank: len(records["OPERATOR"]) for rank, records in logs.items()
            },
        },
        "correctness": {
            "shape": list(rank0.shape),
            "max_abs_error": float(np.max(absolute_error)),
            "mean_abs_error": float(np.mean(absolute_error)),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "cosine_similarity": float(
                np.dot(expected_flat, rank0_flat) / cosine_denominator
            ),
            "rank0_rank1_max_abs": float(np.max(np.abs(rank0 - rank1))),
            "plaintext_top1": int(np.argmax(expected)),
            "rank0_top1": int(np.argmax(rank0)),
            "rank1_top1": int(np.argmax(rank1)),
            "plaintext_top5": _top_indices(expected),
            "rank0_top5": _top_indices(rank0),
            "rank1_top5": _top_indices(rank1),
        },
        "runner_metrics": {rank: records["METRIC"] for rank, records in logs.items()},
        "rss": {rank: _rss_summary(records) for rank, records in logs.items()},
        "stages": {rank: records["STAGE"] for rank, records in logs.items()},
        "operators": {rank: records["OPERATOR"] for rank, records in logs.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=pathlib.Path, required=True)
    parser.add_argument("--run_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    summary = summarize_run(args.artifact_dir, args.run_dir)
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
