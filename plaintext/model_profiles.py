from __future__ import annotations


MODEL_ARCHITECTURE_PROFILES = {
    "deepseek": {
        "total_experts": 64,
        "native_top_k": 6,
        "total_attr": "n_routed_experts",
        "native_hidden_act": "silu",
        "native_auto_map": {
            "AutoModel": "modeling_deepseek_original.DeepseekModel",
            "AutoModelForCausalLM": (
                "modeling_deepseek_original.DeepseekForCausalLM"
            ),
        },
    },
    "qwen": {
        "total_experts": 60,
        "native_top_k": 4,
        "total_attr": "num_experts",
        "native_hidden_act": "silu",
        "native_auto_map": {
            "AutoModel": "modeling_qwen2_moe_original.Qwen2MoeModel",
            "AutoModelForCausalLM": (
                "modeling_qwen2_moe_original.Qwen2MoeForCausalLM"
            ),
        },
    },
    "olmoe": {
        "total_experts": 64,
        "native_top_k": 8,
        "total_attr": "num_experts",
        "native_hidden_act": "silu",
        "native_auto_map": {
            "AutoModel": "modeling_olmoe_original.OlmoeModel",
            "AutoModelForCausalLM": (
                "modeling_olmoe_original.OlmoeForCausalLM"
            ),
        },
    },
}


def model_kind(config) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(
        getattr(config, "architectures", None) or []
    ).lower()
    combined = f"{model_type} {architectures}"
    if "deepseek" in combined:
        return "deepseek"
    if "qwen" in combined:
        return "qwen"
    if "olmoe" in combined:
        return "olmoe"
    raise ValueError(
        f"unsupported MoE model type: model_type={model_type!r}, "
        f"architectures={getattr(config, 'architectures', None)!r}"
    )
