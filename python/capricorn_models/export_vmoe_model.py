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

"""Exports the complete inference-only synthetic V-MoE model."""

import argparse
import functools
import json
import pathlib
import time

import jax
import numpy as np

from capricorn_models.vmoe_model import make_synthetic_image
from capricorn_models.vmoe_model import make_synthetic_parameters
from capricorn_models.vmoe_model import vmoe_forward
from capricorn_models.vmoe_model import vmoe_secure_forward
from capricorn_models.official_vmoe import load_official_checkpoint
from capricorn_models.official_vmoe import preprocess_eval_image
from capricorn_models.official_vmoe import validate_official_parameters
from capricorn_models.vmoe_samples import sample_by_sha256


def _path_text(path):
    return "/".join(str(getattr(component, "key", component)) for component in path)


def export_vmoe_model(
    output_dir: pathlib.Path,
    routing_mode: str = "token_choice",
    activation: str = "silu",
    dense_activation=None,
    moe_activation=None,
    expert_biases=None,
    image=None,
    params=None,
    metadata=None,
) -> None:
    """Writes full-model StableHLO, positional operands, and CPU logits."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = make_synthetic_image() if image is None else np.asarray(image)
    params = make_synthetic_parameters() if params is None else params
    dense_activation = activation if dense_activation is None else dense_activation
    moe_activation = activation if moe_activation is None else moe_activation
    expert_biases = moe_activation == "gelu" if expert_biases is None else expert_biases
    expert_implementation = (
        "_SiLU"
        if moe_activation == "silu" and routing_mode == "expert_choice"
        else "SiLU"
        if moe_activation == "silu"
        else "GELU"
    )
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)

    secure_forward = functools.partial(
        vmoe_secure_forward,
        routing_mode=routing_mode,
        activation=activation,
        dense_activation=dense_activation,
        moe_activation=moe_activation,
        expert_biases=expert_biases,
        trace_stages=True,
    )
    plaintext_forward = functools.partial(
        vmoe_forward,
        routing_mode=routing_mode,
        activation=activation,
        dense_activation=dense_activation,
        moe_activation=moe_activation,
        expert_biases=expert_biases,
    )
    # The positional manifest includes every parameter, including unused biases.
    lowered = jax.jit(secure_forward, keep_unused=True).lower(image, params)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    (output_dir / "model.stablehlo.mlir").write_text(stablehlo)

    manifest_lines = ["arg_000.npy image PRIVATE 0"]
    parameter_path_lines = []
    np.save(output_dir / "arg_000.npy", image, allow_pickle=False)
    for parameter_index, (path, leaf) in enumerate(path_leaves):
        argument_index = parameter_index + 1
        runtime_name = f"param_{parameter_index:03d}"
        np.save(
            output_dir / f"arg_{argument_index:03d}.npy",
            np.asarray(leaf, dtype=np.float32),
            allow_pickle=False,
        )
        manifest_lines.append(f"arg_{argument_index:03d}.npy {runtime_name} PRIVATE 1")
        parameter_path_lines.append(f"{runtime_name} {_path_text(path)}")

    (output_dir / "input_order.txt").write_text("\n".join(manifest_lines) + "\n")
    (output_dir / "parameter_paths.txt").write_text(
        "\n".join(parameter_path_lines) + "\n"
    )
    (output_dir / "routing_mode.txt").write_text(routing_mode + "\n")
    (output_dir / "activation.txt").write_text(activation + "\n")
    (output_dir / "dense_activation.txt").write_text(dense_activation + "\n")
    (output_dir / "moe_activation.txt").write_text(moe_activation + "\n")
    (output_dir / "expert_implementation.txt").write_text(
        expert_implementation + "\n"
    )
    if metadata is not None:
        metadata = {
            **metadata,
            "activation": moe_activation,
            "dense_activation": dense_activation,
            "moe_activation": moe_activation,
            "expert_biases": bool(expert_biases),
            "expert_implementation": expert_implementation,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    expected = np.asarray(jax.jit(plaintext_forward)(image, params))
    np.save(output_dir / "expected.npy", expected, allow_pickle=False)


def export_official_vmoe_model(
    output_dir,
    checkpoint_prefix,
    image_path,
    routing_mode="token_choice",
    profile="research",
):
    """Exports the real checkpoint and eval-preprocessed image for SPU."""
    started = time.perf_counter()
    params, checkpoint_report = load_official_checkpoint(checkpoint_prefix)
    restored = time.perf_counter()
    image, preprocess_report = preprocess_eval_image(image_path)
    preprocessed = time.perf_counter()
    parameter_report = validate_official_parameters(params)
    if parameter_report["coverage"] != 1.0:
        raise ValueError(f"official checkpoint validation failed: {parameter_report}")
    if profile not in ("research", "official_fidelity"):
        raise ValueError(f"unsupported official V-MoE profile {profile!r}")
    moe_activation = "silu" if profile == "research" else "gelu"
    metadata = {
        "kind": "official_vmoe_inference",
        "profile": profile,
        "routing_mode": routing_mode,
        "checkpoint": checkpoint_report,
        "parameters": parameter_report,
        "preprocess": preprocess_report,
        "sample": sample_by_sha256(preprocess_report["sha256"]),
        "timing_seconds": {
            "checkpoint_restore": restored - started,
            "preprocess": preprocessed - restored,
        },
    }
    export_vmoe_model(
        output_dir,
        routing_mode=routing_mode,
        activation=moe_activation,
        dense_activation="gelu",
        moe_activation=moe_activation,
        expert_biases=True,
        image=image,
        params=params,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--routing_mode",
        choices=("token_choice", "expert_choice"),
        default="token_choice",
    )
    parser.add_argument("--activation", choices=("silu", "gelu"), default="silu")
    parser.add_argument(
        "--profile",
        choices=("research", "official_fidelity"),
        default="research",
    )
    parser.add_argument("--checkpoint_prefix", type=pathlib.Path)
    parser.add_argument("--image", type=pathlib.Path)
    args = parser.parse_args()
    if (args.checkpoint_prefix is None) != (args.image is None):
        parser.error("--checkpoint_prefix and --image must be provided together")
    if args.checkpoint_prefix is not None:
        export_official_vmoe_model(
            args.output_dir,
            args.checkpoint_prefix,
            args.image,
            routing_mode=args.routing_mode,
            profile=args.profile,
        )
    else:
        export_vmoe_model(
            args.output_dir, routing_mode=args.routing_mode, activation=args.activation
        )


if __name__ == "__main__":
    main()
