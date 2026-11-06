"""Activation registry used by the paper's plaintext model evaluation."""

from transformers.activations import ACT2FN as TRANSFORMERS_ACT2FN
from transformers.activations import ClassInstantier

from fixed_point.silu import CapricornPiecewiseSiLU, CryptoMoePolynomialSiLU


ACT2FN = ClassInstantier({
    **dict(TRANSFORMERS_ACT2FN),
    "silu_poly_approx": CryptoMoePolynomialSiLU,
    "silu_piecewise_fixed_16_-8_8": CapricornPiecewiseSiLU,
})
