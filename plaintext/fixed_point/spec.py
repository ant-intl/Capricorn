from dataclasses import dataclass


@dataclass(frozen=True)
class FixedPointSpec:
    total_bits: int
    fractional_bits: int
    name: str

    def __post_init__(self) -> None:
        if not 1 <= self.total_bits <= 64:
            raise ValueError(f"{self.name}: total_bits must be in [1, 64]")
        if not 0 <= self.fractional_bits < self.total_bits:
            raise ValueError(
                f"{self.name}: fractional_bits must be in [0, total_bits)"
            )

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def min_integer(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def max_integer(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    @property
    def min_value(self) -> float:
        return self.min_integer / self.scale

    @property
    def max_value(self) -> float:
        return self.max_integer / self.scale


IO_Q18 = FixedPointSpec(64, 18, "Q64.18")
SILU_INPUT_Q18 = FixedPointSpec(40, 18, "Q40.18")
SILU_SLOPE_Q12 = FixedPointSpec(14, 12, "Q14.12")
SILU_INTERCEPT_Q18 = FixedPointSpec(20, 18, "Q20.18")
