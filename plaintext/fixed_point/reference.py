from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import Tensor

from .spec import IO_Q18, FixedPointSpec


def round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def wrap_signed(value: int, total_bits: int) -> int:
    if total_bits <= 0:
        raise ValueError("total_bits must be positive")
    modulus = 1 << total_bits
    unsigned = value % modulus
    sign_bit = 1 << (total_bits - 1)
    return unsigned - modulus if unsigned & sign_bit else unsigned


def ring_reduce_signed(value: int, target_bits: int) -> int:
    return wrap_signed(value, target_bits)


def sign_extend_signed(value: int, source_bits: int, target_bits: int) -> int:
    if target_bits < source_bits:
        raise ValueError("target_bits must be greater than or equal to source_bits")
    return wrap_signed(wrap_signed(value, source_bits), target_bits)


def encode_integer(value: float, spec: FixedPointSpec = IO_Q18) -> int:
    return wrap_signed(
        round_half_away_from_zero(value * spec.scale), spec.total_bits
    )


def decode_integer(value: int, spec: FixedPointSpec = IO_Q18) -> float:
    return wrap_signed(value, spec.total_bits) / spec.scale


def trunc_div_pow2_toward_zero(value: int, shift: int) -> int:
    if shift < 0:
        raise ValueError("shift must be non-negative")
    divisor = 1 << shift
    if value >= 0:
        return value // divisor
    return -((-value) // divisor)


def mul_integer(
    left: int,
    right: int,
    spec: FixedPointSpec = IO_Q18,
    right_fractional_bits: int | None = None,
) -> int:
    shift = (
        spec.fractional_bits
        if right_fractional_bits is None
        else right_fractional_bits
    )
    product = trunc_div_pow2_toward_zero(left * right, shift)
    return wrap_signed(product, spec.total_bits)


def matmul_integer(
    left: Iterable[Iterable[int]],
    right: Iterable[Iterable[int]],
    spec: FixedPointSpec = IO_Q18,
) -> List[List[int]]:
    left_rows = [list(row) for row in left]
    right_rows = [list(row) for row in right]
    if not left_rows or not right_rows:
        return []
    if len(left_rows[0]) != len(right_rows):
        raise ValueError("incompatible matrix dimensions")
    right_cols = list(zip(*right_rows))
    output: List[List[int]] = []
    for row in left_rows:
        output_row = []
        for column in right_cols:
            accumulator = sum(a * b for a, b in zip(row, column))
            value = trunc_div_pow2_toward_zero(
                accumulator, spec.fractional_bits
            )
            output_row.append(wrap_signed(value, spec.total_bits))
        output.append(output_row)
    return output


def encode_fp64(
    value: Tensor,
    spec: FixedPointSpec = IO_Q18,
) -> Tensor:
    if not torch.is_floating_point(value):
        raise TypeError("FP64 reference encoding requires a floating tensor")
    value = value.to(torch.float64)
    scaled = value * float(spec.scale)
    rounded = torch.copysign(torch.floor(torch.abs(scaled) + 0.5), scaled)
    return rounded / float(spec.scale)


def requantize_fp64(
    value: Tensor,
    spec: FixedPointSpec = IO_Q18,
) -> Tensor:
    if not torch.is_floating_point(value):
        raise TypeError("FP64 reference requantization requires a floating tensor")
    value = value.to(torch.float64)
    return torch.trunc(value * float(spec.scale)) / float(spec.scale)


def mul_fp64(
    left: Tensor,
    right: Tensor,
    spec: FixedPointSpec = IO_Q18,
) -> Tensor:
    return requantize_fp64(
        encode_fp64(left, spec) * encode_fp64(right, spec),
        spec,
    )


def matmul_fp64(
    left: Tensor,
    right: Tensor,
    spec: FixedPointSpec = IO_Q18,
) -> Tensor:
    return requantize_fp64(
        torch.matmul(encode_fp64(left, spec), encode_fp64(right, spec)),
        spec,
    )
