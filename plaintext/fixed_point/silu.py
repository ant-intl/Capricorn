from __future__ import annotations

import torch
from torch import Tensor, nn

from .core import (
    IO_Q18,
    SILU_INPUT_Q18,
    add,
    encode,
    is_enabled,
    mul,
    requantize_after_mul,
)
from .reference import (
    ring_reduce_signed,
    sign_extend_signed,
    trunc_div_pow2_toward_zero,
    wrap_signed,
)


# Exact integer encodings used by plaintext/activation/SiLU_a_12_d_18.cpp.
SLOPE_Q12 = (
    -15,
    -34,
    -76,
    -157,
    -288,
    -399,
    -143,
    1087,
    3009,
    4239,
    4495,
    4384,
    4253,
    4172,
    4130,
    4111,
)

INTERCEPT_Q18 = (
    -8302,
    -16897,
    -32648,
    -58373,
    -91807,
    -113650,
    -83687,
    -9909,
    -9909,
    -83687,
    -113650,
    -91807,
    -58373,
    -32649,
    -16896,
    -8303,
)

# Frozen Q18 encodings of the CryptoMoE polynomial coefficients.  These are
# deliberately integers so the plaintext path mirrors the MPC arithmetic
# instead of quantizing Python floating-point literals at each invocation.
CRYPTO_POLY_Q18 = {
    "a2": -1454,
    "a1": -21490,
    "a0": -80414,
    "b6": 72,
    "b4": -2913,
    "b2": 59806,
    "b1": 131072,
    "b0": 2230,
}

CRYPTO_POLY_FLOAT = {
    "a2": -0.0055465625580307,
    "a1": -0.0819767021525476,
    "a0": -0.3067541139982155,
    "b6": 0.0002743776353465,
    "b4": -0.011113046708173,
    "b2": 0.2281430841728270,
    "b1": 0.5,
    "b0": 0.0085064025895951,
}


def _coefficient_tensors(value: Tensor):
    slopes = torch.as_tensor(SLOPE_Q12, device=value.device, dtype=value.dtype)
    intercepts = torch.as_tensor(
        INTERCEPT_Q18, device=value.device, dtype=value.dtype
    )
    return slopes / float(1 << 12), intercepts / float(1 << 18)


def capricorn_piecewise_silu_float(value: Tensor) -> Tensor:
    slopes, intercepts = _coefficient_tensors(value)
    index = torch.floor(value + 8.0).to(torch.long).clamp_(0, 15)
    candidate = slopes[index] * value + intercepts[index]
    return torch.where(
        value < -8.0,
        torch.zeros_like(value),
        torch.where(value >= 8.0, value, candidate),
    )


def capricorn_silu(value: Tensor, config, tag: str = "silu") -> Tensor:
    if not is_enabled(config):
        return capricorn_piecewise_silu_float(value)

    input_q64 = encode(value, config, f"{tag}.input", IO_Q18)
    input_q40 = encode(input_q64, config, f"{tag}.ring_reduce", SILU_INPUT_Q18)
    slopes, intercepts = _coefficient_tensors(input_q40)
    index = torch.floor(input_q40 + 8.0).to(torch.long).clamp_(0, 15)
    product = requantize_after_mul(
        slopes[index] * input_q40,
        config,
        f"{tag}.slope_product",
        SILU_INPUT_Q18,
        inplace=True,
    )
    candidate = add(
        product,
        intercepts[index],
        config,
        f"{tag}.intercept",
        SILU_INPUT_Q18,
    )
    output_q40 = torch.where(
        input_q40 < -8.0,
        torch.zeros_like(input_q40),
        torch.where(input_q40 >= 8.0, input_q40, candidate),
    )
    return encode(output_q40, config, f"{tag}.ring_extend", IO_Q18)


def capricorn_silu_integer(input_q18: int) -> int:
    input_q40 = ring_reduce_signed(input_q18, 40)
    lower = -8 * (1 << 18)
    upper = 8 * (1 << 18)
    if input_q40 < lower:
        output_q40 = 0
    elif input_q40 >= upper:
        output_q40 = input_q40
    else:
        index = (input_q40 - lower) // (1 << 18)
        product = trunc_div_pow2_toward_zero(
            SLOPE_Q12[index] * input_q40, 12
        )
        output_q40 = wrap_signed(product + INTERCEPT_Q18[index], 40)
    return sign_extend_signed(output_q40, 40, 64)


def cryptomoe_polynomial_silu_float(value: Tensor) -> Tensor:
    """Pure floating-point form used by the CryptoMoE float-eager baseline."""
    left_x = value.clamp(-8.0, -4.0)
    left = (
        CRYPTO_POLY_FLOAT["a2"] * left_x.square()
        + CRYPTO_POLY_FLOAT["a1"] * left_x
        + CRYPTO_POLY_FLOAT["a0"]
    )

    middle_x = value.clamp(-4.0, 4.0)
    middle_x2 = middle_x.square()
    middle_x4 = middle_x2.square()
    middle_x6 = middle_x4 * middle_x2
    middle = (
        CRYPTO_POLY_FLOAT["b6"] * middle_x6
        + CRYPTO_POLY_FLOAT["b4"] * middle_x4
        + CRYPTO_POLY_FLOAT["b2"] * middle_x2
        + CRYPTO_POLY_FLOAT["b1"] * middle_x
        + CRYPTO_POLY_FLOAT["b0"]
    )
    return torch.where(
        value < -8.0,
        torch.zeros_like(value),
        torch.where(
            value <= -4.0,
            left,
            torch.where(value <= 4.0, middle, value),
        ),
    )


def _q18_coefficient(value: Tensor, name: str) -> Tensor:
    return torch.as_tensor(
        CRYPTO_POLY_Q18[name] / float(IO_Q18.scale),
        device=value.device,
        dtype=value.dtype,
    )


def _require_finite_q64(value: Tensor, tag: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{tag}: found NaN/Inf input")
    out_of_range = (value < IO_Q18.min_value) | (value > IO_Q18.max_value)
    if bool(out_of_range.any().item()):
        raise OverflowError(
            f"{tag}: input overflows Q64.18 "
            f"[{IO_Q18.min_value}, {IO_Q18.max_value}]"
        )


def cryptomoe_silu(value: Tensor, config, tag: str = "silu") -> Tensor:
    """Q64.18 CryptoMoE polynomial with truncation after every multiply."""
    if not is_enabled(config):
        return cryptomoe_polynomial_silu_float(value)

    _require_finite_q64(value, tag)
    output_dtype = value.dtype
    # FP32 cannot represent every Q18 multiply result closely enough to
    # preserve truncation direction.  Use FP64 only for this bounded
    # polynomial, then cast the Q18 result back for the surrounding model.
    working_value = value.to(torch.float64)
    input_q18 = encode(working_value, config, f"{tag}.input", IO_Q18)

    # Clamp before evaluating inactive branches.  torch.where evaluates both
    # operands, so this prevents an out-of-range input from overflowing x^6
    # without changing any value in a branch that can actually be selected.
    left_x = input_q18.clamp(-8.0, -4.0)
    left_x2 = mul(left_x, left_x, config, f"{tag}.poly.left.x2")
    left = mul(
        _q18_coefficient(input_q18, "a2"),
        left_x2,
        config,
        f"{tag}.poly.left.a2",
    )
    left = add(
        left,
        mul(
            _q18_coefficient(input_q18, "a1"),
            left_x,
            config,
            f"{tag}.poly.left.a1",
        ),
        config,
        f"{tag}.poly.left.sum1",
    )
    left = add(
        left,
        _q18_coefficient(input_q18, "a0"),
        config,
        f"{tag}.poly.left.sum2",
    )

    middle_x = input_q18.clamp(-4.0, 4.0)
    middle_x2 = mul(
        middle_x, middle_x, config, f"{tag}.poly.middle.x2"
    )
    middle_x4 = mul(
        middle_x2, middle_x2, config, f"{tag}.poly.middle.x4"
    )
    middle_x6 = mul(
        middle_x4, middle_x2, config, f"{tag}.poly.middle.x6"
    )
    middle = mul(
        _q18_coefficient(input_q18, "b6"),
        middle_x6,
        config,
        f"{tag}.poly.middle.b6",
    )
    for name, power in (
        ("b4", middle_x4),
        ("b2", middle_x2),
        ("b1", middle_x),
    ):
        middle = add(
            middle,
            mul(
                _q18_coefficient(input_q18, name),
                power,
                config,
                f"{tag}.poly.middle.{name}",
            ),
            config,
            f"{tag}.poly.middle.sum_{name}",
        )
    middle = add(
        middle,
        _q18_coefficient(input_q18, "b0"),
        config,
        f"{tag}.poly.middle.sum_b0",
    )

    output = torch.where(
        input_q18 < -8.0,
        torch.zeros_like(input_q18),
        torch.where(
            input_q18 <= -4.0,
            left,
            torch.where(input_q18 <= 4.0, middle, input_q18),
        ),
    )
    _require_finite_q64(output, f"{tag}.output")
    return encode(output, config, f"{tag}.output", IO_Q18).to(output_dtype)


def _mul_q18_integer(left: int, right: int) -> int:
    return wrap_signed(
        trunc_div_pow2_toward_zero(left * right, IO_Q18.fractional_bits),
        IO_Q18.total_bits,
    )


def _add_q18_integer(left: int, right: int) -> int:
    return wrap_signed(left + right, IO_Q18.total_bits)


def cryptomoe_silu_integer(input_q18: int) -> int:
    """Bit-exact integer-ring reference for the CryptoMoE polynomial."""
    input_q18 = wrap_signed(input_q18, IO_Q18.total_bits)
    lower = -8 * IO_Q18.scale
    left_upper = -4 * IO_Q18.scale
    middle_upper = 4 * IO_Q18.scale
    if input_q18 < lower:
        return 0
    if input_q18 > middle_upper:
        return input_q18

    if input_q18 <= left_upper:
        x = min(max(input_q18, lower), left_upper)
        x2 = _mul_q18_integer(x, x)
        result = _mul_q18_integer(CRYPTO_POLY_Q18["a2"], x2)
        result = _add_q18_integer(
            result, _mul_q18_integer(CRYPTO_POLY_Q18["a1"], x)
        )
        return _add_q18_integer(result, CRYPTO_POLY_Q18["a0"])

    x = min(max(input_q18, left_upper), middle_upper)
    x2 = _mul_q18_integer(x, x)
    x4 = _mul_q18_integer(x2, x2)
    x6 = _mul_q18_integer(x4, x2)
    result = _mul_q18_integer(CRYPTO_POLY_Q18["b6"], x6)
    for name, power in (("b4", x4), ("b2", x2), ("b1", x)):
        result = _add_q18_integer(
            result, _mul_q18_integer(CRYPTO_POLY_Q18[name], power)
        )
    return _add_q18_integer(result, CRYPTO_POLY_Q18["b0"])


def fixed_silu(value: Tensor, config, tag: str = "silu") -> Tensor:
    """Dispatch the fixed activation without changing model-specific code."""
    profile = getattr(config, "fxp_silu_profile", None)
    if profile is None:
        profile = {
            "capricorn-piecewise-fixed": "piecewise",
            "capricorn-polynomial-fixed": "polynomial",
            "cryptomoe-piecewise-fixed": "piecewise",
            "cryptomoe-polynomial-fixed": "polynomial",
        }.get(getattr(config, "fxp_mode", None))
    if profile == "piecewise":
        return capricorn_silu(value, config, tag)
    if profile == "polynomial":
        return cryptomoe_silu(value, config, tag)
    raise ValueError(f"unsupported fixed-point SiLU profile: {profile!r}")


class CapricornPiecewiseSiLU(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return capricorn_piecewise_silu_float(value)

    def extra_repr(self) -> str:
        return "segments=16, range=[-8, 8], slope=Q14.12, intercept=Q20.18"


class CryptoMoePolynomialSiLU(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return cryptomoe_polynomial_silu_float(value)

    def extra_repr(self) -> str:
        return "segments=[-8,-4,4], coefficients=frozen-polynomial"
