from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from execution_modes import FIXED_MODES, validate_mode

from .spec import (
    IO_Q18,
    SILU_INPUT_Q18,
    SILU_INTERCEPT_Q18,
    SILU_SLOPE_Q12,
    FixedPointSpec,
)


def is_enabled(config) -> bool:
    mode = getattr(config, "fxp_mode", "off")
    if mode == "off":
        return False
    if mode == "fixed-fast":
        raise ValueError(
            "fixed-fast is deprecated; use "
            "capricorn-piecewise-fixed instead"
        )
    return validate_mode(mode) in FIXED_MODES


def _stats_enabled(config) -> bool:
    return bool(getattr(config, "fxp_stats", False))


def _round_half_away_from_zero(value: Tensor) -> Tensor:
    return torch.copysign(torch.floor(torch.abs(value) + 0.5), value)


def _record(config, tag: str, value: Tensor, spec: FixedPointSpec) -> None:
    if not _stats_enabled(config) or value.numel() == 0:
        return
    detached = value.detach()
    finite = torch.isfinite(detached)
    nonfinite_count = int((~finite).sum().item())
    finite_value = detached[finite]
    max_abs = (
        float(finite_value.abs().max().item()) if finite_value.numel() else 0.0
    )
    overflow_count = int(
        ((finite_value < spec.min_value) | (finite_value > spec.max_value))
        .sum()
        .item()
    )
    stats = getattr(config, "_fxp_stats_data", None)
    if stats is None:
        stats = {}
        setattr(config, "_fxp_stats_data", stats)
    previous = stats.get(tag, {})
    stats[tag] = {
        "spec": spec.name,
        "calls": int(previous.get("calls", 0)) + 1,
        "max_abs": max(float(previous.get("max_abs", 0.0)), max_abs),
        "overflow_count": int(previous.get("overflow_count", 0)) + overflow_count,
        "nonfinite_count": int(previous.get("nonfinite_count", 0))
        + nonfinite_count,
    }
    if nonfinite_count:
        raise FloatingPointError(f"{tag}: found {nonfinite_count} NaN/Inf values")
    if overflow_count:
        raise OverflowError(
            f"{tag}: {overflow_count} values overflow {spec.name} "
            f"[{spec.min_value}, {spec.max_value}]"
        )


def encode(
    value: Tensor,
    config=None,
    tag: str = "encode",
    spec: FixedPointSpec = IO_Q18,
) -> Tensor:
    """Encode a floating tensor onto a fixed-point grid using std::llround semantics."""
    if not torch.is_floating_point(value):
        raise TypeError(f"{tag}: fixed-point encoding requires a floating tensor")
    _record(config, f"{tag}.input", value, spec)
    scale = float(spec.scale)
    if torch.is_grad_enabled() and value.requires_grad:
        # Preserve the differentiable expression for training/debug callers.
        # Mutating CopysignBackward0's output later would invalidate autograd's
        # version counter even though the mathematical gradient is zero.
        encoded = _round_half_away_from_zero(value * scale) / scale
    else:
        # Full-model inference activations can be several GiB.  The direct
        # expression above materializes abs/add/floor intermediates of the
        # same size and can exceed GPU memory even though the final tensor
        # fits.  Keep encode() non-mutating by allocating the scaled result
        # once, then reuse that fresh tensor for every pointwise step.  Since
        # scale is positive, value has the same sign bit as value * scale,
        # including signed zero.
        encoded = value.mul(scale)
        encoded.abs_().add_(0.5).floor_().copysign_(value).div_(scale)
    _record(config, tag, encoded, spec)
    return encoded


def requantize_after_mul(
    value: Tensor,
    config=None,
    tag: str = "requant_mul",
    spec: FixedPointSpec = IO_Q18,
    *,
    inplace: bool = False,
) -> Tensor:
    """Drop low product bits with C++ integer-division (toward-zero) semantics."""
    if not torch.is_floating_point(value):
        raise TypeError(f"{tag}: requantization requires a floating tensor")
    _record(config, f"{tag}.input", value, spec)
    scale = float(spec.scale)
    # Multiplication outputs can be several GiB for full-vocabulary logits.
    # Reuse those fresh tensors at internal call sites to avoid materializing
    # multiple equally large temporaries. Public callers remain non-mutating
    # unless they explicitly opt in.
    result = value if inplace else value.clone()
    result.mul_(scale).trunc_().div_(scale)
    _record(config, tag, result, spec)
    return result


def is_on_grid(
    value: Tensor,
    spec: FixedPointSpec = IO_Q18,
    atol: Optional[float] = None,
) -> bool:
    if value.numel() == 0:
        return True
    if atol is None:
        atol = 1e-6 if value.dtype == torch.float32 else 1e-12
    scaled = value.detach().to(torch.float64) * spec.scale
    return bool(torch.all(torch.abs(scaled - torch.round(scaled)) <= atol).item())


def _require_prepared(config, tag: str) -> None:
    if is_enabled(config) and not bool(
        getattr(config, "fxp_parameters_prepared", False)
    ):
        raise RuntimeError(
            f"{tag}: fixed-point parameters are not prepared; call "
            "prepare_model_for_fixed_point(model) after loading the checkpoint"
        )


def _release_cached_cuda_memory_for_large_linear(
    value: Tensor, weight: Tensor
) -> None:
    """Defragment the target device before materializing multi-GiB outputs."""
    if not value.is_cuda:
        return
    output_elements = math.prod(value.shape[:-1]) * weight.shape[0]
    output_bytes = output_elements * value.element_size()
    if output_bytes < (1 << 30):
        return
    with torch.cuda.device(value.device):
        torch.cuda.empty_cache()


def linear_with_weight(
    value: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    config,
    tag: str,
) -> Tensor:
    if not is_enabled(config):
        return F.linear(value, weight, bias)
    _require_prepared(config, tag)
    value = encode(value, config, f"{tag}.input")
    _release_cached_cuda_memory_for_large_linear(value, weight)
    output = F.linear(value, weight, None)
    output = requantize_after_mul(
        output, config, f"{tag}.product", inplace=True
    )
    if bias is not None:
        output = add(output, bias, config, f"{tag}.bias")
    return output


def linear(value: Tensor, layer: nn.Linear, config, tag: str) -> Tensor:
    return linear_with_weight(value, layer.weight, layer.bias, config, tag)


def embedding(
    input_ids: Tensor, layer: nn.Embedding, config, tag: str = "embedding"
) -> Tensor:
    if not is_enabled(config):
        return layer(input_ids)
    _require_prepared(config, tag)
    return encode(layer(input_ids), config, tag)


def matmul(left: Tensor, right: Tensor, config, tag: str) -> Tensor:
    if not is_enabled(config):
        return torch.matmul(left, right)
    left = encode(left, config, f"{tag}.left")
    right = encode(right, config, f"{tag}.right")
    return requantize_after_mul(
        torch.matmul(left, right),
        config,
        f"{tag}.product",
        inplace=True,
    )


def mul(left: Tensor, right: Tensor, config, tag: str) -> Tensor:
    if not is_enabled(config):
        return left * right
    left = encode(left, config, f"{tag}.left")
    right = encode(right, config, f"{tag}.right")
    return requantize_after_mul(
        left * right, config, f"{tag}.product", inplace=True
    )


def add(
    left: Tensor,
    right: Tensor,
    config,
    tag: str,
    spec: FixedPointSpec = IO_Q18,
) -> Tensor:
    if not is_enabled(config):
        return left + right
    return encode(left + right, config, tag, spec)


def div(numerator: Tensor, denominator: Tensor, config, tag: str) -> Tensor:
    if not is_enabled(config):
        return numerator / denominator
    numerator = encode(numerator, config, f"{tag}.numerator")
    denominator = encode(denominator, config, f"{tag}.denominator")
    if torch.any(denominator == 0):
        raise ZeroDivisionError(f"{tag}: fixed-point denominator quantized to zero")
    return encode(numerator / denominator, config, tag)


def softmax(value: Tensor, dim: int, config, tag: str) -> Tensor:
    if not is_enabled(config):
        return F.softmax(value, dim=dim, dtype=torch.float32).to(value.dtype)
    finite = torch.isfinite(value)
    quantizable = (
        finite & (value >= IO_Q18.min_value) & (value <= IO_Q18.max_value)
    )
    encoded_finite = encode(
        torch.where(quantizable, value, torch.zeros_like(value)),
        config,
        f"{tag}.input",
    )
    value = torch.where(quantizable, encoded_finite, value)
    shifted = value - value.max(dim=dim, keepdim=True).values
    shifted_finite = torch.isfinite(shifted)
    shifted_quantizable = (
        shifted_finite
        & (shifted >= IO_Q18.min_value)
        & (shifted <= IO_Q18.max_value)
    )
    shifted = torch.where(
        shifted_quantizable,
        encode(
            torch.where(
                shifted_quantizable, shifted, torch.zeros_like(shifted)
            ),
            config,
            f"{tag}.shifted",
        ),
        shifted,
    )
    exponent = encode(torch.exp(shifted), config, f"{tag}.exp")
    denominator = encode(
        exponent.sum(dim=dim, keepdim=True), config, f"{tag}.sum"
    )
    reciprocal = encode(1.0 / denominator, config, f"{tag}.reciprocal")
    return mul(exponent, reciprocal, config, tag)


def sigmoid(value: Tensor, config, tag: str) -> Tensor:
    if not is_enabled(config):
        return torch.sigmoid(value)
    value = encode(value, config, f"{tag}.input")
    return encode(torch.sigmoid(value), config, tag)


def quantized_rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    config,
    tag: str,
) -> Tensor:
    if not is_enabled(config):
        input_dtype = hidden_states.dtype
        value = hidden_states.to(torch.float32)
        variance = value.pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + epsilon)
        return weight * value.to(input_dtype)

    _require_prepared(config, tag)
    value = encode(hidden_states.to(torch.float32), config, f"{tag}.input")
    square = mul(value, value, config, f"{tag}.square")
    variance = encode(square.mean(-1, keepdim=True), config, f"{tag}.mean")
    variance_eps = add(
        variance,
        torch.as_tensor(epsilon, dtype=value.dtype, device=value.device),
        config,
        f"{tag}.epsilon",
    )
    inv_rms = encode(torch.rsqrt(variance_eps), config, f"{tag}.rsqrt")
    normalized = mul(value, inv_rms, config, f"{tag}.normalize")
    return mul(normalized, weight, config, f"{tag}.weight")


def quantized_attention_scale(head_dim: int, device, dtype, config, tag: str):
    scale = torch.as_tensor(
        1.0 / math.sqrt(head_dim), device=device, dtype=dtype
    )
    return encode(scale, config, tag) if is_enabled(config) else scale


def apply_rotary(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    config,
    tag: str,
    *,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
):
    if position_ids is not None and cos.dim() == 2:
        cos = cos[position_ids]
        sin = sin[position_ids]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(value: Tensor) -> Tensor:
        first = value[..., : value.shape[-1] // 2]
        second = value[..., value.shape[-1] // 2 :]
        return torch.cat((-second, first), dim=-1)

    if not is_enabled(config):
        return (
            (query * cos) + (rotate_half(query) * sin),
            (key * cos) + (rotate_half(key) * sin),
        )
    cos = encode(cos, config, f"{tag}.cos")
    sin = encode(sin, config, f"{tag}.sin")
    query_output = add(
        mul(query, cos, config, f"{tag}.query_cos"),
        mul(rotate_half(query), sin, config, f"{tag}.query_sin"),
        config,
        f"{tag}.query",
    )
    key_output = add(
        mul(key, cos, config, f"{tag}.key_cos"),
        mul(rotate_half(key), sin, config, f"{tag}.key_sin"),
        config,
        f"{tag}.key",
    )
    return query_output, key_output


__all__ = [
    "IO_Q18",
    "SILU_INPUT_Q18",
    "SILU_INTERCEPT_Q18",
    "SILU_SLOPE_Q12",
    "FixedPointSpec",
    "add",
    "apply_rotary",
    "div",
    "embedding",
    "encode",
    "is_enabled",
    "is_on_grid",
    "linear",
    "linear_with_weight",
    "matmul",
    "mul",
    "quantized_attention_scale",
    "quantized_rms_norm",
    "requantize_after_mul",
    "sigmoid",
    "softmax",
]
