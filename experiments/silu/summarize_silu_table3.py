#!/usr/bin/env python3

"""Summarize paper SiLU runs using max(server, client) total runtime.

Offline/online times remain server-side measurements. Communication uses
server send+receive after checking byte/action conservation with the client.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


BASELINE_COMM = re.compile(
    r"MOE_SILU_(Server|Client) send bytes: (\d+).*?send actions: (\d+)"
    r".*?recv bytes: (\d+).*?recv actions: (\d+)")
BASELINE_TIME = re.compile(r"MOE_SILU_(Server|Client) need ([0-9.]+) ms")
SECURE_TOTAL = re.compile(
    r"\[(server|client)_SILU\] send bytes: (\d+).*?send actions: (\d+)"
    r".*?recv bytes: (\d+).*?recv actions: (\d+).*?time: ([0-9.]+) ms")
SECURE_OFFLINE = re.compile(r"\[server_SILU_OFFLINE\].*?time: ([0-9.]+) ms")
SECURE_ONLINE = re.compile(r"\[server_SILU_ONLINE\].*?time: ([0-9.]+) ms")
MAX_ERROR = re.compile(r"(?:Max error: |max_ulp_error=)(\d+)")


def _both_parties(matches, metric):
    if len(matches) != 2 or {match[0].lower() for match in matches} != {"server", "client"}:
        raise ValueError(f"expected exactly one server and client {metric}")
    return {match[0].lower(): match[1:] for match in matches}


def parse_log(text: str, implementation: str) -> dict[str, float | int | None]:
    if implementation == "silu":
        comm = _both_parties(BASELINE_COMM.findall(text), "SiLU communication record")
        elapsed = _both_parties(BASELINE_TIME.findall(text), "SiLU timer")
        result = {
            "offline_seconds": None,
            "online_seconds": None,
            "max_error": None,
        }
    else:
        total = _both_parties(SECURE_TOTAL.findall(text), "_SiLU total record")
        comm = {party: values[:4] for party, values in total.items()}
        elapsed = {party: values[4:] for party, values in total.items()}
        offline = SECURE_OFFLINE.search(text)
        online = SECURE_ONLINE.search(text)
        error = MAX_ERROR.search(text)
        result = {
            "offline_seconds": float(offline.group(1)) / 1000.0 if offline else None,
            "online_seconds": float(online.group(1)) / 1000.0 if online else None,
            "max_error": int(error.group(1)) if error else None,
        }
    server_sent, server_send_actions, server_recv, server_recv_actions = map(int, comm["server"])
    client_sent, client_send_actions, client_recv, client_recv_actions = map(int, comm["client"])
    if (server_sent, server_recv, server_send_actions, server_recv_actions) != (
        client_recv, client_sent, client_recv_actions, client_send_actions
    ):
        raise ValueError("SiLU party byte/action counters do not conserve")
    result["sent_bytes"] = server_sent
    result["recv_bytes"] = server_recv
    result["server_elapsed_seconds"] = float(elapsed["server"][0]) / 1000.0
    result["client_elapsed_seconds"] = float(elapsed["client"][0]) / 1000.0
    result["elapsed_seconds"] = max(result["server_elapsed_seconds"], result["client_elapsed_seconds"])
    result["communication_bytes"] = result["sent_bytes"] + result["recv_bytes"]
    result["communication_mib"] = result["communication_bytes"] / (1024 * 1024)
    result["communication_decimal_mb"] = result["communication_bytes"] / 1_000_000
    return result


def load_runs(root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(root.glob("*/*/2p*/run_*")):
        if not run_dir.is_dir():
            continue
        log_path = run_dir / "test.log"
        status_path = run_dir / "exit_status"
        if not status_path.is_file() or status_path.read_text().strip() != "0":
            raise ValueError(f"missing or nonzero exit status: {run_dir}")
        text = log_path.read_text(errors="replace")
        if "[  PASSED  ] 1 test." not in text:
            raise ValueError(f"missing successful gtest result: {log_path}")
        network, implementation, length = log_path.relative_to(root).parts[:3]
        parsed = parse_log(text, implementation)
        parsed.update(
            network=network,
            implementation=implementation,
            exponent=int(length.removeprefix("2p")),
            run=int(run_dir.name.removeprefix("run_")),
            passed=True,
        )
        runs.append(parsed)
    return runs


def aggregate(runs: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(r["network"], r["implementation"], r["exponent"]) for r in runs})
    for key in keys:
        group = [r for r in runs if (r["network"], r["implementation"], r["exponent"]) == key]
        times = [r["elapsed_seconds"] for r in group]
        comms = [r["communication_mib"] for r in group]
        offline = [r["offline_seconds"] for r in group if r["offline_seconds"] is not None]
        online = [r["online_seconds"] for r in group if r["online_seconds"] is not None]
        errors = [r["max_error"] for r in group if r["max_error"] is not None]
        rows.append(
            {
                "network": key[0],
                "implementation": key[1],
                "exponent": key[2],
                "runs": len(group),
                "runtime_mean_s": statistics.mean(times),
                "runtime_stddev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "offline_mean_s": statistics.mean(offline) if offline else None,
                "online_mean_s": statistics.mean(online) if online else None,
                "communication_mean_mib": statistics.mean(comms),
                "all_passed": all(r["passed"] for r in group),
                "max_error": max(errors, default=None),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"result directory does not exist: {args.root}")
    try:
        runs = load_runs(args.root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not runs:
        parser.error("no successful SiLU runs found")
    rows = aggregate(runs)
    (args.root / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (args.root / "summary.csv").open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        markdown = [
            "| Network | Implementation | Length | Runs | Runtime (s) | Communication (MiB) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            markdown.append(
                f"| {row['network'].upper()} | {row['implementation']} | 2^{row['exponent']} | "
                f"{row['runs']} | {row['runtime_mean_s']:.3f} ± {row['runtime_stddev_s']:.3f} | "
                f"{row['communication_mean_mib']:.2f} |"
            )
        (args.root / "summary.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
