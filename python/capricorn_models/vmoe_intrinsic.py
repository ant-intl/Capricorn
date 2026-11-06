# Copyright 2026 Ant International, Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX lowering for Capricorn's fixed-shape SiLU V-MoE blocks."""

import re

import jax
import numpy as np

try:
    from jax import ffi as jax_ffi
except ImportError:
    from jax.extend import ffi as jax_ffi


VMOE_S32_SILU_TARGET = "spu.capricorn.vmoe_s32_e8_k2_silu.v1"
VMOE_S32_EXPERT_CHOICE_SILU_TARGET = (
    "spu.capricorn.vmoe_s32_e8_t16_silu.expert_choice.v1"
)
VMOE_S32_SILU_BIAS_TARGET = (
    "spu.capricorn.vmoe_s32_e8_k2_silu_bias.eval.v3"
)
VMOE_S32_EXPERT_CHOICE_SECURE_SILU_BIAS_TARGET = (
    "spu.capricorn.vmoe_s32_e8_t28_secure_silu_bias.expert_choice.v3"
)
VMOE_S32_GELU_BIAS_TARGET = "spu.capricorn.vmoe_s32_e8_k2_gelu_bias.eval.v2"
VMOE_S32_EXPERT_CHOICE_GELU_BIAS_TARGET = (
    "spu.capricorn.vmoe_s32_e8_t28_gelu_bias.expert_choice.v2"
)
VMOE_TRACE_TARGET_PREFIX = "spu.capricorn.vmoe_trace."
_TRACE_STAGE_PATTERN = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$")

_ROUTING_TARGETS = {
    ("token_choice", "silu"): VMOE_S32_SILU_TARGET,
    ("expert_choice", "silu"): VMOE_S32_EXPERT_CHOICE_SILU_TARGET,
    ("token_choice", "gelu"): VMOE_S32_GELU_BIAS_TARGET,
    ("expert_choice", "gelu"): VMOE_S32_EXPERT_CHOICE_GELU_BIAS_TARGET,
}
_SILU_BIAS_TARGETS = {
    "token_choice": VMOE_S32_SILU_BIAS_TARGET,
    "expert_choice": VMOE_S32_EXPERT_CHOICE_SECURE_SILU_BIAS_TARGET,
}

_TOKEN_SHAPE = (1, 50, 512)
_ROUTER_SHAPE = (512, 8)
_EXPERT_W1_SHAPE = (8, 512, 2048)
_EXPERT_W2_SHAPE = (8, 2048, 512)
_EXPERT_B1_SHAPE = (8, 2048)
_EXPERT_B2_SHAPE = (8, 512)


def vmoe_trace_marker(value, *, stage):
    """Emits an identity custom call delimiting one profiled model stage."""
    if not isinstance(stage, str) or not _TRACE_STAGE_PATTERN.fullmatch(stage):
        raise ValueError(f"invalid V-MoE trace stage {stage!r}")
    target = f"{VMOE_TRACE_TARGET_PREFIX}{stage}.v1"
    result = jax.ShapeDtypeStruct(tuple(value.shape), value.dtype)
    return jax_ffi.ffi_call(
        target,
        result,
        has_side_effect=True,
        vmap_method="sequential",
    )(value)


def _validate_operand(name, operand, expected_shape, expected_dtype):
    if tuple(operand.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(operand.shape)}"
        )
    if not np.issubdtype(operand.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {operand.dtype}")
    if expected_dtype is not None and operand.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {operand.dtype}")


def vmoe_block(
    token_embeddings,
    router_weights,
    expert_w1,
    *expert_operands,
    routing_mode="token_choice",
    activation="silu",
):
    """Emits one fixed V-MoE S/32 SiLU StableHLO custom call.

    ``routing_mode`` is static because token-choice and expert-choice have the
    same external shapes but different secure routing semantics. Secure routing,
    expert computation, and combination execute in Capricorn's PPHLO handler
    with the current runtime's already shared ``spu::Value`` operands.
    """
    try:
        target = _ROUTING_TARGETS[(routing_mode, activation)]
    except (KeyError, TypeError) as error:
        supported = ", ".join(
            f"{routing}/{active}" for routing, active in sorted(_ROUTING_TARGETS)
        )
        raise ValueError(
            f"unsupported routing_mode/activation V-MoE profile "
            f"{routing_mode!r}/{activation!r}; "
            "expected one of: "
            f"{supported}"
        ) from error

    _validate_operand("token_embeddings", token_embeddings, _TOKEN_SHAPE, None)
    dtype = token_embeddings.dtype
    _validate_operand("router_weights", router_weights, _ROUTER_SHAPE, dtype)
    _validate_operand("expert_w1", expert_w1, _EXPERT_W1_SHAPE, dtype)
    if activation == "silu":
        if len(expert_operands) == 1:
            (expert_w2,) = expert_operands
            _validate_operand("expert_w2", expert_w2, _EXPERT_W2_SHAPE, dtype)
            operands = (token_embeddings, router_weights, expert_w1, expert_w2)
        elif len(expert_operands) == 3:
            expert_b1, expert_w2, expert_b2 = expert_operands
            _validate_operand("expert_b1", expert_b1, _EXPERT_B1_SHAPE, dtype)
            _validate_operand("expert_w2", expert_w2, _EXPERT_W2_SHAPE, dtype)
            _validate_operand("expert_b2", expert_b2, _EXPERT_B2_SHAPE, dtype)
            operands = (
                token_embeddings,
                router_weights,
                expert_w1,
                expert_b1,
                expert_w2,
                expert_b2,
            )
            target = _SILU_BIAS_TARGETS[routing_mode]
        else:
            raise ValueError("SiLU V-MoE expects W2 or b1, W2, b2 after W1")
    else:
        if len(expert_operands) != 3:
            raise ValueError("GELU V-MoE expects b1, W2, b2 after W1")
        expert_b1, expert_w2, expert_b2 = expert_operands
        _validate_operand("expert_b1", expert_b1, _EXPERT_B1_SHAPE, dtype)
        _validate_operand("expert_w2", expert_w2, _EXPERT_W2_SHAPE, dtype)
        _validate_operand("expert_b2", expert_b2, _EXPERT_B2_SHAPE, dtype)
        operands = (
            token_embeddings,
            router_weights,
            expert_w1,
            expert_b1,
            expert_w2,
            expert_b2,
        )

    result = jax.ShapeDtypeStruct(_TOKEN_SHAPE, dtype)
    return jax_ffi.ffi_call(
        target,
        result,
        has_side_effect=False,
        vmap_method="sequential",
    )(*operands)
