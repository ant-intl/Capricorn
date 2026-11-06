from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch

from execution_modes import (
    CONTROLLED_MODES,
    FIXED_MODES,
    NATIVE_FLOAT_MODES,
    ORIGINAL_MODE,
    VALID_MODES,
    validate_mode,
)
from model_profiles import MODEL_ARCHITECTURE_PROFILES, model_kind

from .core import IO_Q18, _round_half_away_from_zero, is_enabled


MODEL_PROFILES = {
    kind: {
        "total": int(profile["total_experts"]),
        "base_k": int(profile["native_top_k"]),
        "total_attr": str(profile["total_attr"]),
    }
    for kind, profile in MODEL_ARCHITECTURE_PROFILES.items()
}

MODE_PROFILES = {
    "capricorn-piecewise-fixed": {
        "routing_profile": "capricorn",
        "silu_profile": "piecewise",
        "hidden_act": "silu_piecewise_fixed_16_-8_8",
        "active": "all",
    },
    "capricorn-polynomial-fixed": {
        "routing_profile": "capricorn",
        "silu_profile": "polynomial",
        "hidden_act": "silu_poly_approx",
        "active": "all",
    },
    "cryptomoe-piecewise-fixed": {
        "routing_profile": "cryptomoe",
        "silu_profile": "piecewise",
        "hidden_act": "silu_piecewise_fixed_16_-8_8",
        "active": "base",
    },
    "cryptomoe-polynomial-fixed": {
        "routing_profile": "cryptomoe",
        "silu_profile": "polynomial",
        "hidden_act": "silu_poly_approx",
        "active": "base",
    },
}


def configure_fixed_point(
    config,
    mode: str,
    *,
    stats: bool = False,
    hidden_act: str | None = None,
    num_experts_per_tok: int | None = None,
):
    mode = validate_mode(mode)
    if mode in NATIVE_FLOAT_MODES:
        raise ValueError(
            f"{mode} mode must use its native-float configurator; "
            "fixed-point configuration is forbidden"
        )
    config.fxp_mode = mode
    config.fxp_stats = bool(stats)
    config.fxp_parameters_prepared = False
    profile = MODE_PROFILES.get(mode)
    if profile is not None:
        model = MODEL_PROFILES[model_kind(config)]
        if hidden_act is None:
            hidden_act = profile["hidden_act"]
        if num_experts_per_tok is None:
            num_experts_per_tok = (
                int(getattr(config, model["total_attr"]))
                if profile["active"] == "all"
                else int(model["base_k"])
            )
    if hidden_act is not None:
        config.hidden_act = hidden_act
    if num_experts_per_tok is not None:
        if hasattr(config, "num_experts_per_tok"):
            config.num_experts_per_tok = num_experts_per_tok
        else:
            raise AttributeError("model config has no num_experts_per_tok")
    if mode in CONTROLLED_MODES:
        config._attn_implementation = "eager"
    return config


def configure_routing_trace(
    config,
    *,
    output_path: str | None = None,
    reference_path: str | None = None,
):
    mode = validate_mode(getattr(config, "fxp_mode", ORIGINAL_MODE))
    if mode in NATIVE_FLOAT_MODES:
        raise ValueError(
            f"{mode} mode cannot configure fixed-point routing traces"
        )
    enabled = output_path is not None or reference_path is not None
    if enabled and mode not in CONTROLLED_MODES:
        raise ValueError(
            "routing trace is only supported in controlled modes"
        )

    config.fxp_routing_trace = enabled
    config.fxp_routing_trace_out = output_path
    config._fxp_routing_records = []
    config._fxp_routing_reference = []
    config._fxp_routing_reference_index = 0
    config._fxp_routing_summary = {
        "calls": 0,
        "tokens": 0,
        "slots": 0,
        "matched_slots": 0,
        "exact_set_tokens": 0,
        "top1_matched_tokens": 0,
    }

    if reference_path is not None:
        payload = torch.load(
            reference_path,
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("version") != 1:
            raise ValueError("unsupported routing trace version")
        if payload.get("model_type") != getattr(config, "model_type", None):
            raise ValueError(
                "routing trace model_type mismatch: "
                f"{payload.get('model_type')!r} != "
                f"{getattr(config, 'model_type', None)!r}"
            )
        config._fxp_routing_reference = payload.get("records", [])
    return config


def _routing_profile(config, active: int, total: int, base_k: int) -> str:
    mode = validate_mode(getattr(config, "fxp_mode", "original"))
    if mode in MODE_PROFILES:
        return str(MODE_PROFILES[mode]["routing_profile"])
    if mode == "float-eager":
        hidden_act = getattr(config, "hidden_act", None)
        if (
            hidden_act == "silu_piecewise_fixed_16_-8_8"
            and active == total
        ):
            return "capricorn"
        if hidden_act == "silu_poly_approx" and active == base_k:
            return "cryptomoe"
        raise ValueError(
            "float-eager must match one complete profile: "
            "Capricorn uses all experts with "
            "silu_piecewise_fixed_16_-8_8; CryptoMoE uses the checkpoint "
            "top-k with silu_poly_approx"
        )
    return "original"


def validate_moe_routing(config) -> Dict[str, int | float | str]:
    mode = validate_mode(getattr(config, "fxp_mode", ORIGINAL_MODE))
    if mode in NATIVE_FLOAT_MODES:
        raise ValueError(
            f"{mode} mode cannot use fixed-point routing validation"
        )
    kind = model_kind(config)
    expected = MODEL_PROFILES[kind]
    total = int(getattr(config, expected["total_attr"]))
    active = int(getattr(config, "num_experts_per_tok"))
    capacity = float(getattr(config, "moe_capacity_factor", 2.0))
    base_k = int(getattr(config, "moe_base_k", expected["base_k"]))
    profile = _routing_profile(config, active, total, base_k)
    errors = []
    if total != expected["total"]:
        errors.append(f"total experts={total}, expected {expected['total']}")
    expected_active = total if profile == "capricorn" else base_k
    if profile in {"capricorn", "cryptomoe"} and active != expected_active:
        errors.append(
            f"num_experts_per_tok={active}, expected {expected_active} "
            f"for {profile}"
        )
    mode_profile = MODE_PROFILES.get(mode)
    if (
        mode_profile is not None
        and getattr(config, "hidden_act", None) != mode_profile["hidden_act"]
    ):
        errors.append(
            f"hidden_act={getattr(config, 'hidden_act', None)!r}, expected "
            f"{mode_profile['hidden_act']!r} for {mode}"
        )
    if base_k != expected["base_k"]:
        errors.append(f"capacity base k={base_k}, expected {expected['base_k']}")
    if capacity != 2.0:
        errors.append(f"capacity factor={capacity}, expected 2.0")
    if errors:
        raise ValueError(
            f"{kind} {profile} routing mismatch: " + "; ".join(errors)
        )
    config.moe_base_k = base_k
    config.moe_capacity_factor = capacity
    config.fxp_mode = mode
    config.fxp_routing_profile = profile
    config.fxp_silu_profile = (
        str(mode_profile["silu_profile"])
        if mode_profile is not None
        else (
            "piecewise"
            if getattr(config, "hidden_act", None)
            == "silu_piecewise_fixed_16_-8_8"
            else "polynomial"
        )
    )
    return {
        "model": kind,
        "routing_profile": profile,
        "silu_profile": config.fxp_silu_profile,
        "total_experts": total,
        "active_experts": active,
        "base_k": base_k,
        "capacity_factor": capacity,
    }


def moe_capacity(num_tokens: int, config) -> int:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    mode = validate_mode(getattr(config, "fxp_mode", ORIGINAL_MODE))
    if mode in NATIVE_FLOAT_MODES:
        raise ValueError(
            f"{mode} mode must use the native model capacity path; "
            "fixed-point capacity helpers are forbidden"
        )
    kind = model_kind(config)
    if mode in CONTROLLED_MODES:
        routing = validate_moe_routing(config)
    else:
        defaults = MODEL_PROFILES[kind]
        routing = {
            "model": kind,
            "total_experts": int(getattr(config, defaults["total_attr"])),
            "base_k": int(
                getattr(config, "moe_base_k", defaults["base_k"])
            ),
            "capacity_factor": float(
                getattr(config, "moe_capacity_factor", 2.0)
            ),
        }
    raw_capacity = (
        num_tokens
        * int(routing["base_k"])
        * float(routing["capacity_factor"])
        / int(routing["total_experts"])
    )
    return math.ceil(raw_capacity)


def select_top_capacity(weights: torch.Tensor, capacity: int) -> torch.Tensor:
    """Return assignment indices for the largest per-expert routing weights."""
    if weights.ndim != 1:
        raise ValueError("routing weights must be one-dimensional")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if capacity >= weights.numel():
        return torch.arange(weights.numel(), device=weights.device)
    if capacity == 0:
        return torch.empty(0, device=weights.device, dtype=torch.long)
    return torch.topk(weights, capacity, dim=0, largest=True, sorted=True).indices


def record_routing(config, tag: str, scores: torch.Tensor) -> None:
    if not bool(getattr(config, "fxp_routing_trace", False)):
        return
    routing = validate_moe_routing(config)
    base_k = int(routing["base_k"])
    selected = (
        torch.topk(scores.detach().to(torch.float32), base_k, dim=-1, sorted=True)
        .indices.to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )
    token_count = selected.shape[0]
    summary = config._fxp_routing_summary
    summary["calls"] += 1
    summary["tokens"] += token_count
    summary["slots"] += token_count * base_k

    output_path = getattr(config, "fxp_routing_trace_out", None)
    if output_path is not None:
        config._fxp_routing_records.append(
            {
                "tag": tag,
                "indices": selected,
            }
        )

    reference = getattr(config, "_fxp_routing_reference", [])
    if reference:
        index = int(getattr(config, "_fxp_routing_reference_index", 0))
        if index >= len(reference):
            raise RuntimeError("routing trace has more calls than its reference")
        expected_entry = reference[index]
        expected = expected_entry["indices"]
        if expected_entry["tag"] != tag:
            raise RuntimeError(
                f"routing trace tag mismatch at call {index}: "
                f"{tag!r} != {expected_entry['tag']!r}"
            )
        if tuple(expected.shape) != tuple(selected.shape):
            raise RuntimeError(
                f"routing trace shape mismatch at call {index}: "
                f"{tuple(selected.shape)} != {tuple(expected.shape)}"
            )

        selected_sets = torch.sort(selected, dim=-1).values
        expected_sets = torch.sort(expected, dim=-1).values
        matched = (
            selected_sets.unsqueeze(-1)
            .eq(expected_sets.unsqueeze(-2))
            .any(dim=-1)
            .sum(dim=-1)
        )
        summary["matched_slots"] += int(matched.sum().item())
        summary["exact_set_tokens"] += int((matched == base_k).sum().item())
        summary["top1_matched_tokens"] += int(
            selected[:, 0].eq(expected[:, 0]).sum().item()
        )
        config._fxp_routing_reference_index = index + 1


def finalize_routing_trace(config) -> Dict[str, int | float | str | None]:
    if not bool(getattr(config, "fxp_routing_trace", False)):
        return {"enabled": False}

    reference = getattr(config, "_fxp_routing_reference", [])
    consumed = int(getattr(config, "_fxp_routing_reference_index", 0))
    if reference and consumed != len(reference):
        raise RuntimeError(
            "routing trace reference was not fully consumed: "
            f"{consumed}/{len(reference)} calls"
        )

    output_path = getattr(config, "fxp_routing_trace_out", None)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "model_type": getattr(config, "model_type", None),
                "records": getattr(config, "_fxp_routing_records", []),
            },
            path,
        )

    raw = dict(getattr(config, "_fxp_routing_summary", {}))
    tokens = int(raw.get("tokens", 0))
    slots = int(raw.get("slots", 0))
    has_reference = bool(reference)
    raw.update(
        {
            "enabled": True,
            "output_path": output_path,
            "reference_calls": len(reference),
            "slot_overlap": (
                float(raw.get("matched_slots", 0)) / slots
                if has_reference and slots
                else None
            ),
            "exact_set_rate": (
                float(raw.get("exact_set_tokens", 0)) / tokens
                if has_reference and tokens
                else None
            ),
            "top1_agreement": (
                float(raw.get("top1_matched_tokens", 0)) / tokens
                if has_reference and tokens
                else None
            ),
        }
    )
    return raw


def validate_fixed_point_config(config) -> None:
    mode = validate_mode(getattr(config, "fxp_mode", ORIGINAL_MODE))
    if mode in NATIVE_FLOAT_MODES:
        raise ValueError(
            f"{mode} mode must load the native model implementation; "
            "the fixed-capable model implementation is forbidden"
        )
    if not is_enabled(config):
        return
    attention_impl = getattr(config, "_attn_implementation", None)
    if attention_impl != "eager":
        raise ValueError(
            "fixed-point mode requires eager attention, got "
            f"{attention_impl!r}"
        )
    validate_moe_routing(config)
    if int(getattr(config, "fxp_total_bits", 64)) != 64:
        raise ValueError("fixed-point modes require fxp_total_bits=64")
    if int(getattr(config, "fxp_fractional_bits", 18)) != 18:
        raise ValueError("fixed-point modes require fxp_fractional_bits=18")


@torch.no_grad()
def prepare_model_for_fixed_point(model, chunk_elements: int = 1 << 22):
    config = model.config
    validate_fixed_point_config(config)
    if not is_enabled(config):
        return model
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")

    seen_storage = set()
    for name, parameter in model.named_parameters():
        if not torch.is_floating_point(parameter):
            continue
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"{name}: fixed-point execution requires FP32 parameters, got "
                f"{parameter.dtype}"
            )
        storage_key = (
            parameter.device.type,
            parameter.device.index,
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        if storage_key in seen_storage:
            continue
        seen_storage.add(storage_key)
        if not parameter.data.is_contiguous():
            parameter.data.copy_(
                _round_half_away_from_zero(
                    parameter.data * float(IO_Q18.scale)
                )
                / float(IO_Q18.scale)
            )
            continue
        flat = parameter.data.view(-1)
        for start in range(0, flat.numel(), chunk_elements):
            chunk = flat[start : start + chunk_elements]
            scaled = chunk * float(IO_Q18.scale)
            chunk.copy_(
                _round_half_away_from_zero(scaled) / float(IO_Q18.scale)
            )

    config.fxp_parameters_prepared = True
    return model
