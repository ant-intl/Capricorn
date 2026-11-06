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

"""Reproducible Imagenette-50 assets for secure V-MoE inference."""

import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image


IMAGENETTE_ARCHIVE_URL = (
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
)
IMAGENETTE_ARCHIVE_BYTES = 341_663_724
IMAGENETTE_SELECTION_VERSION = "capricorn-imagenette-50-v1"
IMAGENETTE_CLASSES = {
    "n01440764": {"class_index": 0, "label": "tench"},
    "n02102040": {"class_index": 217, "label": "English springer"},
    "n02979186": {"class_index": 482, "label": "cassette player"},
    "n03000684": {"class_index": 491, "label": "chain saw"},
    "n03028079": {"class_index": 497, "label": "church"},
    "n03394916": {"class_index": 566, "label": "French horn"},
    "n03417042": {"class_index": 569, "label": "garbage truck"},
    "n03425413": {"class_index": 571, "label": "gas pump"},
    "n03445777": {"class_index": 574, "label": "golf ball"},
    "n03888257": {"class_index": 701, "label": "parachute"},
}
VMOE_TARGETS = {
    "token_choice": "spu.capricorn.vmoe_s32_e8_k2_silu_bias.eval.v3",
    "expert_choice": (
        "spu.capricorn.vmoe_s32_e8_t28_secure_silu_bias.expert_choice.v3"
    ),
}
VMOE_EXPERT_IMPLEMENTATIONS = {
    "token_choice": "SiLU",
    "expert_choice": "_SiLU",
}


def _selection_digest(relative_path):
    payload = f"{IMAGENETTE_SELECTION_VERSION}\n{relative_path}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_imagenette_members(member_names, per_class=5):
    """Selects a stable balanced validation subset from archive member names."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    candidates = {synset: [] for synset in IMAGENETTE_CLASSES}
    for raw_name in member_names:
        path = pathlib.PurePosixPath(raw_name)
        parts = path.parts
        if len(parts) != 4 or parts[:2] != ("imagenette2-320", "val"):
            continue
        synset = parts[2]
        if synset not in candidates or path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        relative_path = path.as_posix()
        candidates[synset].append((_selection_digest(relative_path), relative_path))

    selected = []
    for synset, class_info in IMAGENETTE_CLASSES.items():
        ranked = sorted(set(candidates[synset]))
        if len(ranked) < per_class:
            raise ValueError(
                f"Imagenette class {synset} has {len(ranked)} valid members; "
                f"expected at least {per_class}"
            )
        for selection_digest, relative_path in ranked[:per_class]:
            stem = pathlib.PurePosixPath(relative_path).stem
            selected.append(
                {
                    "id": f"{synset}_{stem}",
                    "relative_path": relative_path,
                    "selection_digest": selection_digest,
                    "synset": synset,
                    **class_info,
                }
            )
    return selected


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_download(path, expected_bytes, expected_sha256):
    path = pathlib.Path(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Imagenette archive byte count is {path.stat().st_size}; "
            f"expected {expected_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Imagenette archive SHA256 is {actual_sha256}; "
            f"expected {expected_sha256}"
        )


def fetch_imagenette_archive(
    url, destination, expected_bytes=IMAGENETTE_ARCHIVE_BYTES, expected_sha256=None
):
    """Downloads an archive through a validated atomic cache install."""
    destination = pathlib.Path(destination)
    if destination.is_file():
        try:
            _validate_download(destination, expected_bytes, expected_sha256)
            return destination
        except ValueError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(url) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        _validate_download(temporary_path, expected_bytes, expected_sha256)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _validate_archive_members(members):
    by_name = {}
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe Imagenette archive path {member.name!r}")
        if member.name in by_name:
            raise ValueError(f"duplicate Imagenette archive member {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are forbidden in Imagenette archive: {member.name!r}")
        if not member.isdir() and not member.isreg():
            raise ValueError(
                f"unsupported Imagenette archive member type: {member.name!r}"
            )
        by_name[member.name] = member
    return by_name


def _write_bytes_atomic(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_npy_atomic(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.save(output, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_imagenette_50(archive_path, output_dir):
    """Safely extracts the deterministic Imagenette-50 validation subset."""
    archive_path = pathlib.Path(archive_path)
    output_dir = pathlib.Path(output_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        by_name = _validate_archive_members(archive.getmembers())
        selected = select_imagenette_members(
            name for name, member in by_name.items() if member.isreg()
        )
        samples = []
        for record in selected:
            member = by_name[record["relative_path"]]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"unable to read archive member {member.name!r}")
            payload = extracted.read()
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    rgb = image.convert("RGB")
                    width, height = rgb.size
            except Exception as error:
                raise ValueError(f"invalid image {member.name!r}") from error
            if min(width, height) < 256:
                raise ValueError(
                    f"image {member.name!r} is too small for eval preprocessing: "
                    f"{width}x{height}"
                )
            suffix = pathlib.PurePosixPath(member.name).suffix
            local_path = pathlib.Path("images") / f"{record['id']}{suffix}"
            _write_bytes_atomic(output_dir / local_path, payload)
            samples.append(
                {
                    **record,
                    "local_path": local_path.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "decoded_size": [width, height],
                }
            )

    manifest = {
        "schema_version": 1,
        "dataset": "Imagenette",
        "source": {
            "url": IMAGENETTE_ARCHIVE_URL,
            "archive_file": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_file(archive_path),
        },
        "selection": {
            "version": IMAGENETTE_SELECTION_VERSION,
            "split": "val",
            "per_class": 5,
            "sample_count": len(samples),
        },
        "classes": IMAGENETTE_CLASSES,
        "samples": samples,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def validate_base_artifact(path, routing_mode):
    """Validates a paper SiLU research artifact without loading parameters."""
    path = pathlib.Path(path)
    if routing_mode not in VMOE_TARGETS:
        raise ValueError(f"unknown routing mode {routing_mode!r}")
    required = (
        "model.stablehlo.mlir",
        "input_order.txt",
        "parameter_paths.txt",
        "routing_mode.txt",
        "activation.txt",
        "dense_activation.txt",
        "moe_activation.txt",
        "expert_implementation.txt",
        "metadata.json",
        "arg_000.npy",
        "expected.npy",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"base artifact is missing files: {missing}")

    file_routing = (path / "routing_mode.txt").read_text().strip()
    metadata = json.loads((path / "metadata.json").read_text())
    if file_routing != routing_mode or metadata.get("routing_mode") != routing_mode:
        raise ValueError(
            f"base artifact routing is {file_routing!r}/"
            f"{metadata.get('routing_mode')!r}; expected {routing_mode!r}"
        )
    dense_activation = (path / "dense_activation.txt").read_text().strip()
    moe_activation = (path / "moe_activation.txt").read_text().strip()
    implementation = (path / "expert_implementation.txt").read_text().strip()
    if dense_activation != "gelu" or metadata.get("dense_activation") != "gelu":
        raise ValueError("base artifact dense backbone must use GELU")
    if (
        moe_activation != "silu"
        or metadata.get("moe_activation") != "silu"
        or (path / "activation.txt").read_text().strip() != "silu"
    ):
        raise ValueError("base artifact MoE must use the SiLU research profile")
    expected_implementation = VMOE_EXPERT_IMPLEMENTATIONS[routing_mode]
    if (
        implementation != expected_implementation
        or metadata.get("expert_implementation") != expected_implementation
    ):
        raise ValueError(
            f"base artifact must use {expected_implementation} for {routing_mode}"
        )
    if metadata.get("expert_biases") is not True:
        raise ValueError("base artifact must consume both official expert biases")

    parameters = metadata.get("parameters", {})
    if (
        parameters.get("coverage") != 1.0
        or parameters.get("expected_arrays") != 140
        or parameters.get("actual_arrays") != 140
        or any(
            parameters.get(field)
            for field in (
                "missing",
                "unexpected",
                "shape_mismatches",
                "dtype_mismatches",
            )
        )
    ):
        raise ValueError("base artifact does not have complete official parameters")

    manifest_lines = [
        line.split()
        for line in (path / "input_order.txt").read_text().splitlines()
        if line.strip()
    ]
    if not manifest_lines or manifest_lines[0] != [
        "arg_000.npy",
        "image",
        "PRIVATE",
        "0",
    ]:
        raise ValueError("base artifact image ownership must be PRIVATE rank 0")
    for fields in manifest_lines[1:]:
        if len(fields) != 4 or fields[2:] != ["PRIVATE", "1"]:
            raise ValueError("base artifact parameters must be PRIVATE rank 1")
        if not (path / fields[0]).is_file():
            raise ValueError(f"base artifact is missing operand {fields[0]!r}")

    target = VMOE_TARGETS[routing_mode]
    custom_call_count = (path / "model.stablehlo.mlir").read_text().count(target)
    if custom_call_count != 2:
        raise ValueError(
            f"base artifact must contain two {routing_mode} custom calls; "
            f"got {custom_call_count}"
        )
    return {
        "routing_mode": routing_mode,
        "custom_call_count": custom_call_count,
        "operand_count": len(manifest_lines),
        "metadata": metadata,
    }


def create_artifact_overlay(base_dir, overlay_dir, image_npy, expected, metadata):
    """Creates a lightweight immutable hardlink overlay for one sample."""
    base_dir = pathlib.Path(base_dir)
    overlay_dir = pathlib.Path(overlay_dir)
    image_npy = pathlib.Path(image_npy)
    routing_mode = metadata.get("routing_mode")
    report = validate_base_artifact(base_dir, routing_mode)
    if not image_npy.is_file():
        raise ValueError(f"preprocessed image NPY is missing: {image_npy}")
    if overlay_dir.exists():
        raise FileExistsError(f"artifact overlay already exists: {overlay_dir}")

    overlay_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{overlay_dir.name}.", dir=overlay_dir.parent)
    )
    try:
        excluded = {"arg_000.npy", "expected.npy", "metadata.json"}
        for source in base_dir.iterdir():
            if source.name in excluded:
                continue
            if not source.is_file():
                raise ValueError(f"base artifact contains non-file entry: {source}")
            os.link(source, temporary_dir / source.name)
        os.link(image_npy, temporary_dir / "arg_000.npy")
        np.save(
            temporary_dir / "expected.npy",
            np.asarray(expected, dtype=np.float32),
            allow_pickle=False,
        )
        combined_metadata = {**report["metadata"], **metadata}
        write_json_atomic(temporary_dir / "metadata.json", combined_metadata)
        os.replace(temporary_dir, overlay_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def prepare_imagenette_50(
    checkpoint_prefix,
    manifest_path,
    image_root,
    token_base_dir,
    expert_base_dir,
    output_root,
    *,
    preprocess_fn=None,
    load_checkpoint_fn=None,
    token_forward=None,
    expert_forward=None,
):
    """Builds hardlinked Token/Expert artifacts and matching plaintext oracles."""
    manifest_path = pathlib.Path(manifest_path)
    image_root = pathlib.Path(image_root)
    output_root = pathlib.Path(output_root)
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    sample_ids = [sample.get("id") for sample in samples]
    if len(samples) != 50 or len(set(sample_ids)) != 50 or not all(sample_ids):
        raise ValueError("Imagenette-50 manifest must contain 50 unique sample ids")

    token_base_report = validate_base_artifact(token_base_dir, "token_choice")
    expert_base_report = validate_base_artifact(expert_base_dir, "expert_choice")
    if preprocess_fn is None:
        from capricorn_models.official_vmoe import preprocess_eval_image

        preprocess_fn = preprocess_eval_image
    if load_checkpoint_fn is None:
        from capricorn_models.official_vmoe import load_official_checkpoint

        load_checkpoint_fn = load_official_checkpoint

    params, checkpoint_report = load_checkpoint_fn(checkpoint_prefix)
    if token_forward is None or expert_forward is None:
        import functools

        import jax

        from capricorn_models.vmoe_model import vmoe_forward

        token_forward = token_forward or jax.jit(
            functools.partial(
                vmoe_forward,
                routing_mode="token_choice",
                dense_activation="gelu",
                moe_activation="silu",
                expert_biases=True,
            )
        )
        expert_forward = expert_forward or jax.jit(
            functools.partial(
                vmoe_forward,
                routing_mode="expert_choice",
                dense_activation="gelu",
                moe_activation="silu",
                expert_biases=True,
            )
        )

    started = time.perf_counter()
    prepared_samples = []
    for sample in samples:
        image_path = image_root / sample["local_path"]
        image, preprocess_report = preprocess_fn(image_path)
        image_npy = output_root / "images" / f"{sample['id']}.npy"
        _save_npy_atomic(image_npy, image)
        token_expected = np.asarray(token_forward(image, params), dtype=np.float32)
        expert_expected = np.asarray(expert_forward(image, params), dtype=np.float32)
        if token_expected.shape != (1, 1000) or expert_expected.shape != (1, 1000):
            raise ValueError(
                f"plaintext logits for {sample['id']} must have shape [1,1000]"
            )

        common_metadata = {
            "kind": "official_vmoe_imagenette_50",
            "activation": "silu",
            "dense_activation": "gelu",
            "moe_activation": "silu",
            "expert_biases": True,
            "checkpoint": checkpoint_report,
            "sample": sample,
            "preprocess": preprocess_report,
            "benchmark": {
                "selection_version": IMAGENETTE_SELECTION_VERSION,
                "manifest_sha256": sha256_file(manifest_path),
            },
        }
        create_artifact_overlay(
            token_base_dir,
            output_root / "artifacts" / "token_choice" / sample["id"],
            image_npy,
            token_expected,
            {**common_metadata, "routing_mode": "token_choice"},
        )
        create_artifact_overlay(
            expert_base_dir,
            output_root / "artifacts" / "expert_choice" / sample["id"],
            image_npy,
            expert_expected,
            {**common_metadata, "routing_mode": "expert_choice"},
        )
        prepared_samples.append(sample["id"])

    report = {
        "schema_version": 1,
        "sample_count": len(prepared_samples),
        "sample_ids": prepared_samples,
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint": checkpoint_report,
        "token_base": token_base_report["routing_mode"],
        "expert_base": expert_base_report["routing_mode"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output_root / "preparation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--archive", type=pathlib.Path, required=True)
    dataset.add_argument("--output_dir", type=pathlib.Path, required=True)
    dataset.add_argument("--url", default=IMAGENETTE_ARCHIVE_URL)
    dataset.add_argument("--archive_bytes", type=int, default=IMAGENETTE_ARCHIVE_BYTES)
    dataset.add_argument("--archive_sha256")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--checkpoint_prefix", type=pathlib.Path, required=True)
    prepare.add_argument("--manifest", type=pathlib.Path, required=True)
    prepare.add_argument("--image_root", type=pathlib.Path, required=True)
    prepare.add_argument("--token_base_dir", type=pathlib.Path, required=True)
    prepare.add_argument("--expert_base_dir", type=pathlib.Path, required=True)
    prepare.add_argument("--output_root", type=pathlib.Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "dataset":
        archive = fetch_imagenette_archive(
            args.url, args.archive, args.archive_bytes, args.archive_sha256
        )
        result = extract_imagenette_50(archive, args.output_dir)
    else:
        result = prepare_imagenette_50(
            args.checkpoint_prefix,
            args.manifest,
            args.image_root,
            args.token_base_dir,
            args.expert_base_dir,
            args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
