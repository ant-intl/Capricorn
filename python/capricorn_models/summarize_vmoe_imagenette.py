# Copyright 2026 Ant International, Ant Group. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Aggregates Imagenette V-MoE results without retaining raw metric records."""

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile

import numpy as np

from capricorn_models.vmoe_imagenette import write_json_atomic


def validate_sample_summary(summary, sample, routing_mode):
    if summary.get("routing_mode") != routing_mode:
        raise ValueError(f"summary routing does not match {routing_mode}")
    if summary.get("sample", {}).get("id") != sample.get("id"):
        raise ValueError("summary sample id does not match manifest")
    instrumentation = summary.get("instrumentation", {})
    if instrumentation.get("marker_count") != 55:
        raise ValueError("summary must contain 55 trace markers")
    if instrumentation.get("moe_custom_call_count") != 2:
        raise ValueError("summary must contain two MoE custom calls")
    if instrumentation.get("stage_count_by_rank") != {"0": 54, "1": 54}:
        raise ValueError("summary must contain 54 stages for both ranks")
    if instrumentation.get("operator_aggregate_count_by_rank") != {
        "0": 350,
        "1": 350,
    }:
        raise ValueError("summary must contain 350 operator aggregates for both ranks")
    correctness = summary.get("correctness", {})
    if correctness.get("rank0_rank1_max_abs") != 0.0:
        raise ValueError("P0 and P1 outputs must be identical")
    if correctness.get("rank0_top1") != correctness.get("rank1_top1"):
        raise ValueError("P0 and P1 top-1 predictions must match")
    if correctness.get("rank0_top5") != correctness.get("rank1_top5"):
        raise ValueError("P0 and P1 top-5 predictions must match")


def wilson_interval(successes, total, confidence=0.95):
    if total == 0:
        return (0.0, 0.0)
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    if confidence != 0.95:
        raise ValueError("only the predeclared 95% Wilson interval is supported")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return (center - radius, center + radius)


def _describe(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "stddev": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _top_indices(logits, count=5):
    flat = np.asarray(logits).reshape(-1)
    return np.argsort(flat)[-count:][::-1].astype(int).tolist()


def _records_by_id(records):
    output = {}
    for record in records:
        sample_id = record["summary"].get("sample", {}).get("id")
        if not sample_id or sample_id in output:
            raise ValueError(f"duplicate or missing result sample id {sample_id!r}")
        output[sample_id] = record
    return output


def _metric(summary, operator):
    matches = [
        record
        for record in summary["runner_metrics"]["0"]
        if record["operator"] == operator
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one rank-0 {operator} metric")
    return matches[0]


def _accuracy(samples, records):
    sources = {"plaintext": "expected", "rank0": "rank0", "rank1": "rank1"}
    result = {}
    for source_name, record_key in sources.items():
        top1_successes = 0
        top5_successes = 0
        per_class = {}
        for sample, record in zip(samples, records):
            label = int(sample["class_index"])
            top5 = _top_indices(record[record_key])
            top1_successes += top5[0] == label
            top5_successes += label in top5
            class_entry = per_class.setdefault(
                sample["synset"], {"count": 0, "top1_correct": 0, "top5_correct": 0}
            )
            class_entry["count"] += 1
            class_entry["top1_correct"] += top5[0] == label
            class_entry["top5_correct"] += label in top5
        for entry in per_class.values():
            entry["top1"] = entry["top1_correct"] / entry["count"]
            entry["top5"] = entry["top5_correct"] / entry["count"]
        result[source_name] = {
            "top1_correct": int(top1_successes),
            "top5_correct": int(top5_successes),
            "top1": top1_successes / len(samples),
            "top5": top5_successes / len(samples),
            "top1_wilson_95": list(wilson_interval(top1_successes, len(samples))),
            "per_class": per_class,
        }
    result["top1"] = result["rank0"]["top1"]
    result["top5"] = result["rank0"]["top5"]
    return result


def _performance(records):
    total = [_metric(record["summary"], "total_after_link") for record in records]
    execute = [_metric(record["summary"], "execute_pphlo") for record in records]
    return {
        "total_after_link_ms": _describe([record["elapsed_ms"] for record in total]),
        "execute_pphlo_ms": _describe([record["elapsed_ms"] for record in execute]),
        "bidirectional_bytes": _describe(
            [record["sent_bytes"] + record["recv_bytes"] for record in total]
        ),
        "bidirectional_actions": _describe(
            [record["sent_actions"] + record["recv_actions"] for record in total]
        ),
    }


def _stage_statistics(records):
    grouped = {}
    for record in records:
        for stage in record["summary"]["stages"]["0"]:
            entry = grouped.setdefault(
                stage["stage"], {"elapsed_ms": [], "bytes": [], "actions": []}
            )
            entry["elapsed_ms"].append(stage["elapsed_ms"])
            entry["bytes"].append(stage["sent_bytes"] + stage["recv_bytes"])
            entry["actions"].append(stage["sent_actions"] + stage["recv_actions"])
    return {
        stage: {
            "elapsed_ms": _describe(values["elapsed_ms"]),
            "bidirectional_bytes": _describe(values["bytes"]),
            "bidirectional_actions": _describe(values["actions"]),
        }
        for stage, values in sorted(grouped.items())
    }


def _operator_statistics(records):
    grouped = {}
    for record in records:
        for operator in record["summary"]["operators"]["0"]:
            key = (operator["stage"], operator["operator"])
            entry = grouped.setdefault(
                key, {"count": [], "elapsed_ms": [], "bytes": [], "actions": []}
            )
            entry["count"].append(operator["count"])
            entry["elapsed_ms"].append(operator["elapsed_ms"])
            entry["bytes"].append(operator["sent_bytes"] + operator["recv_bytes"])
            entry["actions"].append(
                operator["sent_actions"] + operator["recv_actions"]
            )
    output = {}
    for (stage, operator), values in sorted(grouped.items()):
        output.setdefault(stage, {})[operator] = {
            "count": _describe(values["count"]),
            "elapsed_ms": _describe(values["elapsed_ms"]),
            "bidirectional_bytes": _describe(values["bytes"]),
            "bidirectional_actions": _describe(values["actions"]),
        }
    return output


def _moe_statistics(records):
    grouped = {}
    for record in records:
        for metric in record["summary"]["runner_metrics"]["0"]:
            if metric["stage"] not in ("moe", "custom_call"):
                continue
            entry = grouped.setdefault(metric["operator"], [])
            entry.append(metric["elapsed_ms"])
    return {operator: _describe(values) for operator, values in sorted(grouped.items())}


def aggregate_imagenette_results(manifest, token_records, expert_records):
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("benchmark manifest has no samples")
    token_by_id = _records_by_id(token_records)
    expert_by_id = _records_by_id(expert_records)
    expected_ids = [sample["id"] for sample in samples]
    if set(token_by_id) != set(expected_ids) or set(expert_by_id) != set(expected_ids):
        raise ValueError("Token/Expert result sets do not exactly match the manifest")
    token = [token_by_id[sample_id] for sample_id in expected_ids]
    expert = [expert_by_id[sample_id] for sample_id in expected_ids]
    for sample, record in zip(samples, token):
        validate_sample_summary(record["summary"], sample, "token_choice")
    for sample, record in zip(samples, expert):
        validate_sample_summary(record["summary"], sample, "expert_choice")

    result_samples = []
    for sample, token_record, expert_record in zip(samples, token, expert):
        label = int(sample["class_index"])
        token_top5 = _top_indices(token_record["rank0"])
        expert_top5 = _top_indices(expert_record["rank0"])
        result_samples.append(
            {
                "id": sample["id"],
                "synset": sample["synset"],
                "class_index": label,
                "label": sample.get("label"),
                "token_choice": {
                    "top1": token_top5[0],
                    "top5": token_top5,
                    "top1_correct": token_top5[0] == label,
                    "top5_correct": label in token_top5,
                },
                "expert_choice": {
                    "top1": expert_top5[0],
                    "top5": expert_top5,
                    "top1_correct": expert_top5[0] == label,
                    "top5_correct": label in expert_top5,
                },
            }
        )

    token_totals = [
        _metric(record["summary"], "total_after_link")["elapsed_ms"]
        for record in token
    ]
    expert_totals = [
        _metric(record["summary"], "total_after_link")["elapsed_ms"]
        for record in expert
    ]
    results = {"schema_version": 1, "sample_count": len(samples), "samples": result_samples}
    metrics = {
        "schema_version": 1,
        "sample_count": len(samples),
        "accuracy": {
            "token_choice": _accuracy(samples, token),
            "expert_choice": _accuracy(samples, expert),
            "routing_top1_agreement": sum(
                _top_indices(left["rank0"], 1) == _top_indices(right["rank0"], 1)
                for left, right in zip(token, expert)
            )
            / len(samples),
        },
        "fidelity": {
            mode: {
                field: _describe(
                    [record["summary"]["correctness"][field] for record in records]
                )
                for field in (
                    "max_abs_error",
                    "mean_abs_error",
                    "rmse",
                    "cosine_similarity",
                )
            }
            for mode, records in (("token_choice", token), ("expert_choice", expert))
        },
        "performance": {
            "token_choice": _performance(token),
            "expert_choice": _performance(expert),
            "paired": {
                "time_saved_ms": _describe(
                    [left - right for left, right in zip(token_totals, expert_totals)]
                ),
                "speedup": _describe(
                    [left / right for left, right in zip(token_totals, expert_totals)]
                ),
            },
        },
        "stages": {
            "token_choice": _stage_statistics(token),
            "expert_choice": _stage_statistics(expert),
        },
        "operators": {
            "token_choice": _operator_statistics(token),
            "expert_choice": _operator_statistics(expert),
        },
        "moe": {
            "token_choice": _moe_statistics(token),
            "expert_choice": _moe_statistics(expert),
        },
    }
    return results, metrics


def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_result_archive(path, manifest, token_records, expert_records):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = manifest["samples"]
    token_by_id = _records_by_id(token_records)
    expert_by_id = _records_by_id(expert_records)
    sample_ids = [sample["id"] for sample in samples]
    if set(sample_ids) != set(token_by_id) or set(sample_ids) != set(expert_by_id):
        raise ValueError("result records do not match manifest for archive")

    arrays = {
        "sample_ids": np.asarray(sample_ids),
        "labels": np.asarray([sample["class_index"] for sample in samples], np.int64),
    }
    for prefix, records in (("token", token_by_id), ("expert", expert_by_id)):
        for source, key in (
            ("plaintext", "expected"),
            ("rank0", "rank0"),
            ("rank1", "rank1"),
        ):
            arrays[f"{prefix}_{source}_logits"] = np.stack(
                [np.asarray(records[sample_id][key], np.float32).reshape(1000) for sample_id in sample_ids]
            )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    _, _ = aggregate_imagenette_results(manifest, token_records, expert_records)
    sample_index = []
    for sample_id, label in zip(sample_ids, arrays["labels"]):
        entry = {"id": sample_id, "class_index": int(label)}
        for prefix in ("token", "expert"):
            top5 = _top_indices(arrays[f"{prefix}_rank0_logits"][len(sample_index)])
            entry[prefix] = {
                "top1": top5[0],
                "top5": top5,
                "top1_correct": top5[0] == int(label),
                "top5_correct": int(label) in top5,
            }
        sample_index.append(entry)
    return {
        "schema_version": 1,
        "archive_file": path.name,
        "archive_sha256": _sha256_file(path),
        "sample_count": len(samples),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "samples": sample_index,
    }


def load_result_records(manifest, artifact_root, run_root):
    """Loads complete logits and validated summaries from a server run tree."""
    artifact_root = pathlib.Path(artifact_root)
    run_root = pathlib.Path(run_root)
    output = {}
    for routing_mode in ("token_choice", "expert_choice"):
        records = []
        for sample in manifest.get("samples", []):
            artifact_dir = artifact_root / routing_mode / sample["id"]
            run_dir = run_root / routing_mode / sample["id"]
            required = {
                "summary": run_dir / "summary.json",
                "expected": artifact_dir / "expected.npy",
                "rank0": run_dir / "output_rank0.npy",
                "rank1": run_dir / "output_rank1.npy",
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                raise ValueError(
                    f"missing {routing_mode}/{sample['id']} result files: {missing}"
                )
            summary = json.loads(required["summary"].read_text())
            validate_sample_summary(summary, sample, routing_mode)
            arrays = {
                name: np.load(path, allow_pickle=False)
                for name, path in required.items()
                if name != "summary"
            }
            for name, value in arrays.items():
                if value.size != 1000:
                    raise ValueError(
                        f"{routing_mode}/{sample['id']} {name} has {value.size} logits"
                    )
            records.append({"summary": summary, **arrays})
        output[routing_mode] = records
    return output


def write_aggregate_outputs(manifest_path, artifact_root, run_root, output_dir):
    """Writes full inference results separately from aggregate-only metrics."""
    manifest_path = pathlib.Path(manifest_path)
    output_dir = pathlib.Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    records = load_result_records(manifest, artifact_root, run_root)
    results, metrics = aggregate_imagenette_results(
        manifest, records["token_choice"], records["expert_choice"]
    )
    archive_name = "vmoe_imagenette_50_results.npz"
    index = write_result_archive(
        output_dir / archive_name,
        manifest,
        records["token_choice"],
        records["expert_choice"],
    )
    # Keep all per-sample predictions/correctness in the result index.  The
    # metrics file intentionally contains aggregate statistics only.
    index["samples"] = results["samples"]
    write_json_atomic(output_dir / "vmoe_imagenette_50_results.json", index)
    write_json_atomic(output_dir / "vmoe_imagenette_50_metrics.json", metrics)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--artifact_root", type=pathlib.Path, required=True)
    parser.add_argument("--run_root", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    index = write_aggregate_outputs(
        args.manifest, args.artifact_root, args.run_root, args.output_dir
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
