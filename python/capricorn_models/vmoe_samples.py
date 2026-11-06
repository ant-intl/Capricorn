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

"""Reproducible real-image sample manifest for official V-MoE inference."""

import argparse
import hashlib
import json
import pathlib
import tempfile
import urllib.request

_MANIFEST_PATH = pathlib.Path(__file__).with_name("vmoe_samples.json")


def load_sample_manifest(path=_MANIFEST_PATH):
    manifest = json.loads(pathlib.Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported V-MoE sample manifest schema")
    samples = manifest.get("samples", [])
    identifiers = [sample.get("id") for sample in samples]
    if len(identifiers) != len(set(identifiers)) or not all(identifiers):
        raise ValueError("V-MoE sample ids must be non-empty and unique")
    return manifest


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_by_id(sample_id, path=_MANIFEST_PATH):
    for sample in load_sample_manifest(path)["samples"]:
        if sample["id"] == sample_id:
            return sample
    raise KeyError(f"unknown V-MoE sample {sample_id!r}")


def sample_by_sha256(sha256, path=_MANIFEST_PATH):
    for sample in load_sample_manifest(path)["samples"]:
        if sample["sha256"] == sha256:
            return sample
    return None


def fetch_sample(sample_id, output_dir, path=_MANIFEST_PATH):
    sample = sample_by_id(sample_id, path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / sample["file_name"]
    if destination.exists() and sha256_file(destination) == sample["sha256"]:
        return destination

    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as temporary:
        temporary_path = pathlib.Path(temporary.name)
        with urllib.request.urlopen(sample["source_url"]) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
    try:
        actual_sha256 = sha256_file(temporary_path)
        if actual_sha256 != sample["sha256"]:
            raise ValueError(f"sample {sample_id} SHA256 mismatch: {actual_sha256}")
        if temporary_path.stat().st_size != sample["bytes"]:
            raise ValueError(f"sample {sample_id} byte size mismatch")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="tensorflow_yellow_labrador")
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(fetch_sample(args.sample_id, args.output_dir))


if __name__ == "__main__":
    main()
