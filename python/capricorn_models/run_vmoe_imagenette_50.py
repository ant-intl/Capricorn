# Copyright 2026 Ant International, Ant Group. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Internal batch scheduling and resume validation for V-MoE runs."""

import json
import pathlib

from capricorn_models.summarize_vmoe_imagenette import validate_sample_summary
from capricorn_models.summarize_vmoe_run import summarize_run
from capricorn_models.vmoe_imagenette import write_json_atomic


def planned_jobs(samples, routing_modes=("token_choice", "expert_choice")):
    jobs = []
    for index, sample in enumerate(samples):
        order = tuple(routing_modes) if index % 2 == 0 else tuple(reversed(routing_modes))
        jobs.extend((sample["id"], routing_mode) for routing_mode in order)
    return jobs


def _load_or_create_summary(artifact_dir, run_dir):
    summary_path = pathlib.Path(run_dir) / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    summary = summarize_run(artifact_dir, run_dir)
    write_json_atomic(summary_path, summary)
    return summary


def completed_job(artifact_dir, run_dir, sample, routing_mode):
    artifact_dir = pathlib.Path(artifact_dir)
    run_dir = pathlib.Path(run_dir)
    required = (
        "run_status.json",
        "rank0.log",
        "rank1.log",
        "output_rank0.npy",
        "output_rank1.npy",
    )
    if not artifact_dir.is_dir() or any(
        not (run_dir / name).is_file() for name in required
    ):
        return False
    try:
        status = json.loads((run_dir / "run_status.json").read_text())
        if status.get("rank0_status") != 0 or status.get("rank1_status") != 0:
            return False
        summary = _load_or_create_summary(artifact_dir, run_dir)
        validate_sample_summary(summary, sample, routing_mode)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def run_benchmark(
    manifest_path,
    artifact_root,
    run_root,
    *,
    artifact_runner,
    routing_modes=("token_choice", "expert_choice"),
):
    manifest_path = pathlib.Path(manifest_path)
    artifact_root = pathlib.Path(artifact_root)
    run_root = pathlib.Path(run_root)
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    sample_by_id = {sample["id"]: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("benchmark manifest contains duplicate sample ids")

    run_root.mkdir(parents=True, exist_ok=True)
    jobs = planned_jobs(samples, routing_modes)
    progress = {
        "schema_version": 1,
        "job_count": len(jobs),
        "completed": 0,
        "skipped": 0,
        "failed": [],
    }

    for job_index, (sample_id, routing_mode) in enumerate(jobs):
        sample = sample_by_id[sample_id]
        artifact_dir = artifact_root / routing_mode / sample_id
        run_dir = run_root / routing_mode / sample_id
        if completed_job(artifact_dir, run_dir, sample, routing_mode):
            progress["completed"] += 1
            progress["skipped"] += 1
            progress["last_job"] = {
                "index": job_index,
                "sample_id": sample_id,
                "routing_mode": routing_mode,
                "status": "skipped_valid",
            }
            write_json_atomic(run_root / "progress.json", progress)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        # An invalid cached summary must not mask the fresh run.  The runner
        # writes logs and logits; completed_job regenerates this file from
        # those authoritative artifacts.
        (run_dir / "summary.json").unlink(missing_ok=True)
        runner_status = artifact_runner(artifact_dir, run_dir)
        valid = runner_status == 0 and completed_job(
            artifact_dir, run_dir, sample, routing_mode
        )
        if valid:
            progress["completed"] += 1
            status = "completed"
        else:
            progress["failed"].append(
                {
                    "sample_id": sample_id,
                    "routing_mode": routing_mode,
                    "runner_status": runner_status,
                }
            )
            status = "failed"
        progress["last_job"] = {
            "index": job_index,
            "sample_id": sample_id,
            "routing_mode": routing_mode,
            "status": status,
        }
        write_json_atomic(run_root / "progress.json", progress)

    progress["success"] = progress["completed"] == len(jobs) and not progress["failed"]
    write_json_atomic(run_root / "progress.json", progress)
    return progress
