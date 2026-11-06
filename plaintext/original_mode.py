from __future__ import annotations

from typing import Dict

from model_profiles import MODEL_ARCHITECTURE_PROFILES, model_kind

_FIXED_POINT_STATE = {
    "_attn_implementation",
    "moe_base_k",
    "moe_capacity_factor",
}
_NATIVE_EXPERIMENT_STATE = {"top_t_capacity_k"}


def _fixed_point_state_names(config) -> list[str]:
    return sorted(
        name
        for name in vars(config)
        if name.startswith(("fxp_", "_fxp_"))
        or name in _FIXED_POINT_STATE
    )


def _restore_native_float_runtime(config, profile) -> None:
    for name in _fixed_point_state_names(config):
        delattr(config, name)
    config._attn_implementation_internal = None
    config.auto_map = {
        **dict(getattr(config, "auto_map", None) or {}),
        **dict(profile["native_auto_map"]),
    }


def _validate_native_float_model(
    config,
    *,
    expected_active: int,
    expected_capacity_k: int | None,
    mode_name: str,
) -> Dict[str, int | str]:
    profile = MODEL_ARCHITECTURE_PROFILES[model_kind(config)]
    total = int(getattr(config, profile["total_attr"]))
    active = int(getattr(config, "num_experts_per_tok"))
    hidden_act = str(getattr(config, "hidden_act"))
    errors = []
    if total != int(profile["total_experts"]):
        errors.append(
            f"total experts={total}, "
            f"expected {int(profile['total_experts'])}"
        )
    if active != expected_active:
        errors.append(
            f"num_experts_per_tok={active}, expected {expected_active}"
        )
    if hidden_act != str(profile["native_hidden_act"]):
        errors.append(
            f"hidden_act={hidden_act!r}, "
            f"expected {str(profile['native_hidden_act'])!r}"
        )
    attention_override = getattr(
        config, "_attn_implementation_internal", None
    )
    if attention_override is not None:
        errors.append(
            "_attn_implementation_internal="
            f"{attention_override!r}, expected None for default attention"
        )
    auto_map = dict(getattr(config, "auto_map", None) or {})
    for auto_class, expected_target in profile["native_auto_map"].items():
        actual_target = auto_map.get(auto_class)
        if actual_target != expected_target:
            errors.append(
                f"auto_map[{auto_class!r}]={actual_target!r}, "
                f"expected native target {expected_target!r}"
            )
    leaked = _fixed_point_state_names(config)
    if leaked:
        errors.append(
            f"fixed-point state leaked into {mode_name} mode: "
            + ", ".join(leaked)
        )
    actual_capacity_k = getattr(config, "top_t_capacity_k", None)
    if expected_capacity_k is None:
        if actual_capacity_k is not None:
            errors.append(
                "top_t_capacity_k leaked into original mode: "
                f"{actual_capacity_k!r}"
            )
        capacity_k = active
    else:
        if actual_capacity_k != expected_capacity_k:
            errors.append(
                f"top_t_capacity_k={actual_capacity_k!r}, "
                f"expected {expected_capacity_k}"
            )
        capacity_k = expected_capacity_k
    if errors:
        raise ValueError(
            f"{mode_name} model configuration mismatch: "
            + "; ".join(errors)
        )
    return {
        "model": model_kind(config),
        "total_experts": total,
        "active_experts": active,
        "capacity_top_k": capacity_k,
        "hidden_act": hidden_act,
        "implementation": str(
            profile["native_auto_map"]["AutoModelForCausalLM"]
        ),
    }


def configure_original_model(
    config,
    *,
    hidden_act: str | None = None,
    num_experts_per_tok: int | None = None,
):
    """Restore the upstream floating-point routing/activation defaults.

    The checked-in model config files were historically modified by expert
    and activation sweeps.  Original evaluation repairs those values only on
    the in-memory AutoConfig object and removes all fixed-point runtime state;
    checkpoint/config files on disk remain untouched.
    """

    profile = MODEL_ARCHITECTURE_PROFILES[model_kind(config)]
    expected_act = str(profile["native_hidden_act"])
    expected_active = int(profile["native_top_k"])
    if hidden_act not in {None, expected_act}:
        raise ValueError(
            "original mode uses the native activation "
            f"{expected_act!r}; got {hidden_act!r}"
        )
    if num_experts_per_tok not in {None, expected_active}:
        raise ValueError(
            "original mode uses the native top-k "
            f"{expected_active}; got {num_experts_per_tok}"
        )

    config.hidden_act = expected_act
    config.num_experts_per_tok = expected_active
    for name in _NATIVE_EXPERIMENT_STATE:
        if hasattr(config, name):
            delattr(config, name)
    _restore_native_float_runtime(config, profile)
    validate_original_model(config)
    return config


def validate_original_model(config) -> Dict[str, int | str]:
    profile = MODEL_ARCHITECTURE_PROFILES[model_kind(config)]
    return _validate_native_float_model(
        config,
        expected_active=int(profile["native_top_k"]),
        expected_capacity_k=None,
        mode_name="original",
    )


def configure_float_all_experts_model(
    config,
    *,
    hidden_act: str | None = None,
    num_experts_per_tok: int | None = None,
):
    """Use all experts without leaving the checkpoint-native float path.

    Only router top-k changes.  SiLU, parameter dtype, attention selection,
    and model implementation stay native.  The top-t capacity keeps using
    the checkpoint's native top-k so this experiment isolates expert
    activation count from the established capacity budget.
    """

    profile = MODEL_ARCHITECTURE_PROFILES[model_kind(config)]
    expected_act = str(profile["native_hidden_act"])
    expected_active = int(profile["total_experts"])
    capacity_k = int(profile["native_top_k"])
    if hidden_act not in {None, expected_act}:
        raise ValueError(
            "float-all-experts mode uses the native activation "
            f"{expected_act!r}; got {hidden_act!r}"
        )
    if num_experts_per_tok not in {None, expected_active}:
        raise ValueError(
            "float-all-experts mode activates all experts "
            f"({expected_active}); got {num_experts_per_tok}"
        )

    config.hidden_act = expected_act
    config.num_experts_per_tok = expected_active
    config.top_t_capacity_k = capacity_k
    _restore_native_float_runtime(config, profile)
    validate_float_all_experts_model(config)
    return config


def validate_float_all_experts_model(config) -> Dict[str, int | str]:
    profile = MODEL_ARCHITECTURE_PROFILES[model_kind(config)]
    return _validate_native_float_model(
        config,
        expected_active=int(profile["total_experts"]),
        expected_capacity_k=int(profile["native_top_k"]),
        mode_name="float-all-experts",
    )
