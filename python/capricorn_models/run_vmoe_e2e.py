"""One V-MoE entrypoint for single/batch runs on native, LAN or WAN."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time


NETWORKS = {"lan": ("3gbit", "0.1ms"), "wan": ("400mbit", "20ms")}
ROUTINGS = ("token_choice", "expert_choice")
MODULE = "capricorn_models.run_vmoe_e2e"


class RunError(RuntimeError):
    pass


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("single", "batch"), default="single")
    p.add_argument("--network", choices=("native", "lan", "wan"), default="native")
    p.add_argument("--routing", choices=(*ROUTINGS, "both"))
    p.add_argument("--artifact-dir", type=Path, help="Single: existing exported model; otherwise export a synthetic model")
    p.add_argument("--manifest", type=Path, help="Batch: sample manifest")
    p.add_argument("--artifact-root", type=Path, help="Batch: ROUTING/SAMPLE_ID artifact tree")
    p.add_argument("--run-dir", type=Path, help="Output/resume directory; default: a new results/vmoe-* directory")
    p.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--python", default=sys.executable, help="Python used for model export and verification")
    p.add_argument("--runner-bin", type=Path, default=os.environ.get("CAPRICORN_VMOE_RUNNER_BIN"))
    p.add_argument("--cot-data-dir", type=Path, default=os.environ.get("COT_DATA_DIR"))
    p.add_argument("--expert-top-t", choices=("serial", "parallel"), default=os.environ.get("CAPRICORN_VMOE_EXPERT_TOP_T_MODE", "serial"))
    p.add_argument("--profile-operators", choices=("0", "1"), default=os.environ.get("CAPRICORN_VMOE_PROFILE_OPERATORS"))
    p.add_argument("--skip-build", action="store_true", default=os.environ.get("CAPRICORN_SKIP_BUILD") == "1")
    p.add_argument("--timeout", type=float, default=3600.0, help="Maximum seconds per two-party run")
    p.add_argument("--_isolated", action="store_true", help=argparse.SUPPRESS)
    return p


def normalize(args):
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise RunError("--timeout must be finite and positive")
    if args.profile_operators not in (None, "0", "1"):
        raise RunError("operator profiling must be 0 or 1")
    if args.expert_top_t not in ("serial", "parallel"):
        raise RunError("expert Top-t must be serial or parallel")
    for key in ("workspace", "artifact_dir", "manifest", "artifact_root", "run_dir", "runner_bin", "cot_data_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, Path(value).resolve())
    if args.mode == "batch":
        if args.artifact_dir or not args.manifest or not args.artifact_root:
            raise RunError("batch requires --manifest and --artifact-root, without --artifact-dir")
        manifest = json.loads(args.manifest.read_text())
        samples = manifest.get("samples", [])
        ids = [s.get("id") for s in samples]
        if any(not isinstance(x, str) or not x or x in (".", "..") or Path(x).name != x for x in ids):
            raise RunError("sample ids must be single directory names")
        if not ids or len(set(ids)) != len(ids):
            raise RunError("manifest must contain nonempty, unique sample ids")
    elif args.manifest or args.artifact_root:
        raise RunError("--manifest and --artifact-root require --mode batch")
    if args.artifact_dir and not args.artifact_dir.is_dir():
        raise RunError(f"artifact directory is missing: {args.artifact_dir}")
    args.routes = ROUTINGS if args.routing == "both" or (args.mode == "batch" and args.routing is None) else (args.routing or "token_choice",)
    if args.artifact_dir:
        recorded = args.artifact_dir / "routing_mode.txt"
        if recorded.is_file():
            routing = recorded.read_text().strip()
            if routing not in ROUTINGS:
                raise RunError("artifact routing_mode.txt is invalid")
            if args.routing is None:
                args.routes = (routing,)
            elif args.routes != (routing,):
                raise RunError("--routing does not match the exported single artifact")
        elif len(args.routes) != 1:
            raise RunError("one existing artifact cannot represent both routing modes")
    if args.network != "native" and sys.platform != "linux":
        raise RunError("LAN/WAN network namespaces require Linux")
    args.profile_operators = args.profile_operators or ("1" if args.mode == "batch" else "0")
    args.runner_bin = args.runner_bin or args.workspace / "bazel-bin/src/adapters/vmoe/pphlo_runner_main"
    args.cot_data_dir = args.cot_data_dir or args.workspace / "capricorn/internal/cot_data"
    if args.run_dir is None:
        parent = args.workspace / "results"
        parent.mkdir(parents=True, exist_ok=True)
        args.run_dir = Path(tempfile.mkdtemp(prefix="vmoe-", dir=parent))
    return args


def ensure_run_config(args):
    config = {k: str(getattr(args, k)) if getattr(args, k) is not None else None for k in ("workspace", "artifact_dir", "manifest", "artifact_root", "runner_bin", "expert_top_t", "profile_operators", "mode", "network")}
    config["routing"] = list(args.routes)
    if args.manifest:
        config["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    path = args.run_dir / "invocation.json"
    if path.is_file():
        if json.loads(path.read_text()) != config:
            raise RunError("output directory belongs to another configuration; choose a different --run-dir")
    else:
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise RunError("output directory has no invocation record; choose an empty --run-dir")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(path, config)


def setup_network(args):
    if args.network == "native":
        return
    own_ns = os.readlink("/proc/self/ns/net")
    try:
        namespace_fd = int(os.environ.get("CAPRICORN_VMOE_PARENT_NETNS_FD", "-1"))
        parent_ns = os.readlink(f"/proc/self/fd/{namespace_fd}")
    except (ValueError, OSError) as ex:
        raise RunError("parent network namespace handle is missing") from ex
    if not parent_ns.startswith("net:[") or own_ns == parent_ns or parent_ns != os.environ.get("CAPRICORN_VMOE_PARENT_NETNS"):
        raise RunError("refusing network shaping outside the child network namespace")
    rate, delay = NETWORKS[args.network]
    subprocess.run(["ip", "link", "set", "lo", "up"], check=True)
    subprocess.run(["tc", "qdisc", "replace", "dev", "lo", "root", "netem", "rate", rate, "delay", delay], check=True)
    with (args.run_dir / "tc_qdisc.txt").open("w") as out:
        subprocess.run(["tc", "qdisc", "show", "dev", "lo"], stdout=out, check=True)
    with (args.run_dir / "network.rtt.txt").open("w") as out:
        subprocess.run(["ping", "-n", "-c", "8", "-i", "0.2", "127.0.0.1"], stdout=out, check=True, timeout=15)
    write_json(args.run_dir / "network_profile.json", {"name": args.network, "rate": rate, "delay": delay, "interface": "lo", "isolation": "unshare_user_network_namespace"})


def isolated_command(argv, args):
    return ["unshare", "--user", "--map-root-user", "--net", sys.executable, "-m", MODULE, *argv, "--run-dir", str(args.run_dir), "--skip-build", "--_isolated"]


def stop(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class Runner:
    def __init__(self, args):
        self.args = args
        self.environment = dict(os.environ)
        self.environment["COT_DATA_DIR"] = str(args.cot_data_dir)
        if args.profile_operators == "1":
            self.environment["CAPRICORN_VMOE_PROFILE_OPERATORS"] = "1"
        else:
            self.environment.pop("CAPRICORN_VMOE_PROFILE_OPERATORS", None)

    def build(self):
        if not self.args.skip_build:
            subprocess.run(["bazelisk", "build", "-c", "opt", "--noshow_progress", "//src/adapters/vmoe:pphlo_runner_main"], cwd=self.args.workspace, check=True)
        if not os.access(self.args.runner_bin, os.X_OK):
            raise RunError(f"runner is not executable: {self.args.runner_bin}")

    def execute(self, artifact, run_dir):
        artifact, run_dir = Path(artifact), Path(run_dir)
        for name in ("model.stablehlo.mlir", "input_order.txt", "expected.npy"):
            if not (artifact / name).is_file():
                raise RunError(f"missing artifact file: {artifact / name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.args.cot_data_dir.mkdir(parents=True, exist_ok=True)
        if self.args.network != "native":
            profile = json.loads((self.args.run_dir / "network_profile.json").read_text())
            write_json(run_dir / "network_profile.json", profile)
        processes = []
        begin = time.monotonic()
        try:
            with contextlib.ExitStack() as stack:
                sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
                for sock in sockets:
                    sock.bind(("127.0.0.1", 0))
                parties = ",".join(f"127.0.0.1:{s.getsockname()[1]}" for s in sockets)
            with contextlib.ExitStack() as logs:
                for role in (0, 1):
                    out = logs.enter_context((run_dir / f"rank{role}.log").open("w"))
                    command = [str(self.args.runner_bin), f"--role={role}", f"--parties={parties}", f"--artifact_dir={artifact}", f"--expert_top_t_mode={self.args.expert_top_t}", f"--output={run_dir / f'output_rank{role}.npy'}"]
                    processes.append(subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=self.environment, start_new_session=True))
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in (None, 0) for p in processes):
                        raise RunError("a V-MoE party failed")
                    if time.monotonic() - begin > self.args.timeout:
                        raise RunError("two-party run timed out")
                    time.sleep(0.02)
                if any(p.returncode != 0 for p in processes):
                    raise RunError("a V-MoE party failed")
        finally:
            for process in processes:
                stop(process)
            status = {"artifact_dir": str(artifact), "elapsed_seconds": time.monotonic() - begin}
            status.update({f"rank{role}_status": processes[role].returncode if role < len(processes) else None for role in (0, 1)})
            write_json(run_dir / "run_status.json", status)
        return 0


def run_single(args, runner):
    for routing in args.routes:
        run_dir = args.run_dir / routing
        artifact = args.artifact_dir or run_dir / "artifact"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "verification.json").unlink(missing_ok=True)
        if args.artifact_dir is None:
            subprocess.run([args.python, "-m", "capricorn_models.export_vmoe_model", f"--output_dir={artifact}", f"--routing_mode={routing}"], check=True)
        runner.execute(artifact, run_dir)
        completed = subprocess.run([args.python, "-m", "capricorn_models.verify_vmoe_model", f"--artifact_dir={artifact}", f"--run_dir={run_dir}"], capture_output=True, text=True, check=True)
        write_json(run_dir / "verification.json", json.loads(completed.stdout))
    return 0


def run_batch(args, runner):
    from capricorn_models.run_vmoe_imagenette_50 import run_benchmark

    def execute(artifact, run_dir):
        try:
            return runner.execute(artifact, run_dir)
        except RunError as ex:
            print(f"{run_dir}: {ex}", file=sys.stderr)
            return 1

    progress = run_benchmark(args.manifest, args.artifact_root, args.run_dir, artifact_runner=execute, routing_modes=args.routes)
    print(json.dumps(progress, sort_keys=True))
    return 0 if progress["success"] else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    args = cli.parse_args(argv)
    try:
        args = normalize(args)
        ensure_run_config(args)
        runner = Runner(args)
        runner.build()
        if args.network != "native" and not args._isolated:
            environment = dict(os.environ)
            environment["CAPRICORN_VMOE_PARENT_NETNS"] = os.readlink("/proc/self/ns/net")
            namespace_fd = os.open("/proc/self/ns/net", os.O_RDONLY)
            environment["CAPRICORN_VMOE_PARENT_NETNS_FD"] = str(namespace_fd)
            try:
                child = subprocess.Popen(isolated_command(argv, args), env=environment, start_new_session=True, pass_fds=(namespace_fd,))
                try:
                    return child.wait()
                finally:
                    stop(child)
            finally:
                os.close(namespace_fd)
        setup_network(args)
        result = run_single(args, runner) if args.mode == "single" else run_batch(args, runner)
        print(f"V-MoE {args.mode} finished; outputs: {args.run_dir}")
        return result
    except (RunError, OSError, ValueError, subprocess.SubprocessError) as ex:
        print(f"V-MoE: {ex}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("V-MoE interrupted; current outputs preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
