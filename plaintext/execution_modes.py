from __future__ import annotations


ORIGINAL_MODE = "original"
FLOAT_ALL_EXPERTS_MODE = "float-all-experts"
NATIVE_FLOAT_MODES = frozenset({ORIGINAL_MODE, FLOAT_ALL_EXPERTS_MODE})
FIXED_MODES = frozenset(
    {
        "capricorn-piecewise-fixed",
        "capricorn-polynomial-fixed",
        "cryptomoe-piecewise-fixed",
        "cryptomoe-polynomial-fixed",
    }
)
EXPERIMENT_MODES = frozenset({*NATIVE_FLOAT_MODES, *FIXED_MODES})
DIAGNOSTIC_MODES = frozenset({"float-eager"})
CONTROLLED_MODES = frozenset({*DIAGNOSTIC_MODES, *FIXED_MODES})
VALID_MODES = frozenset({*EXPERIMENT_MODES, *DIAGNOSTIC_MODES})


def validate_mode(mode: str) -> str:
    if mode == "fixed-fast":
        raise ValueError(
            "fixed-fast is deprecated; use "
            "capricorn-piecewise-fixed instead"
        )
    if mode not in VALID_MODES:
        raise ValueError(
            f"unsupported mode {mode!r}; expected one of {VALID_MODES}"
        )
    return mode
