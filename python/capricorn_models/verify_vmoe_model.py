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

"""Verifies full-model two-party Capricorn logits against the CPU oracle."""

import argparse
import json
import pathlib

import numpy as np


# Full CHEETAH/FM64 runs have observed up to 0.01197 max error while producing
# identical party outputs and the same top-1 class.  Use a measured research
# margin while retaining exact party consistency and top-1 checks below.
_MAX_ORACLE_ABS_ERROR = 2e-2


def verify_vmoe_model(artifact_dir: pathlib.Path, run_dir=None) -> dict[str, float | int]:
  artifact_dir = pathlib.Path(artifact_dir)
  run_dir = artifact_dir if run_dir is None else pathlib.Path(run_dir)
  expected = np.load(artifact_dir / "expected.npy", allow_pickle=False)
  rank0 = np.load(run_dir / "output_rank0.npy", allow_pickle=False)
  rank1 = np.load(run_dir / "output_rank1.npy", allow_pickle=False)

  required_shape = (1, 1000)
  if expected.shape != required_shape:
    raise ValueError(f"expected has shape {expected.shape}, want {required_shape}")
  if rank0.shape != required_shape or rank1.shape != required_shape:
    raise ValueError(
        f"party outputs have shapes {rank0.shape} and {rank1.shape}, "
        f"want {required_shape}"
    )
  if not all(np.all(np.isfinite(value)) for value in (expected, rank0, rank1)):
    raise ValueError("V-MoE logits must contain only finite values")
  output_max = float(np.max(np.abs(rank0)))
  if output_max <= 1e-3:
    raise ValueError(f"secure V-MoE logits are trivial: max={output_max}")

  np.testing.assert_allclose(rank0, rank1, atol=1e-6, rtol=0.0)
  np.testing.assert_allclose(
      rank0, expected, atol=_MAX_ORACLE_ABS_ERROR, rtol=0.0
  )
  secure_top1 = int(np.argmax(rank0, axis=-1)[0])
  oracle_top1 = int(np.argmax(expected, axis=-1)[0])
  if secure_top1 != oracle_top1:
    raise ValueError(
        f"V-MoE top-1 mismatch: secure={secure_top1}, oracle={oracle_top1}"
    )
  return {
      "output_max_abs": output_max,
      "party_max_abs_diff": float(np.max(np.abs(rank0 - rank1))),
      "oracle_max_abs_diff": float(np.max(np.abs(rank0 - expected))),
      "secure_top1": secure_top1,
      "oracle_top1": oracle_top1,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifact_dir", type=pathlib.Path, required=True)
  parser.add_argument("--run_dir", type=pathlib.Path)
  args = parser.parse_args()
  print(json.dumps(verify_vmoe_model(args.artifact_dir, args.run_dir), sort_keys=True))


if __name__ == "__main__":
  main()
