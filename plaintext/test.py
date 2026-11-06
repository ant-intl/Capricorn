from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
import argparse
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from execution_modes import (
    CONTROLLED_MODES,
    FIXED_MODES,
    FLOAT_ALL_EXPERTS_MODE,
    NATIVE_FLOAT_MODES,
    ORIGINAL_MODE,
    EXPERIMENT_MODES,
    validate_mode,
)
from lm_eval.utils import make_table
from original_mode import (
    configure_float_all_experts_model,
    configure_original_model,
    validate_float_all_experts_model,
    validate_original_model,
)


def test_model(dataset_name, model, limit=None):
    results = simple_evaluate(
        model=model, tasks=[dataset_name], write_out=True, limit=limit
    )
    print(make_table(results))


def mode_argument(value):
    try:
        return validate_mode(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_model_load_kwargs(args, config):
    kwargs = {
        "config": config,
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto",
        "torch_dtype": (
            torch.float32 if args.mode in CONTROLLED_MODES else "auto"
        ),
    }
    if args.mode in CONTROLLED_MODES:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        kwargs["attn_implementation"] = "eager"
    return kwargs


def load_model(args):
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.mode in NATIVE_FLOAT_MODES:
        if args.fxp_stats:
            raise ValueError(
                f"{args.mode} mode does not support fixed-point stats"
            )
        if args.routing_trace_out or args.routing_trace_reference:
            raise ValueError(
                f"{args.mode} mode does not support fixed-point routing traces"
            )
        if args.mode == ORIGINAL_MODE:
            configure_original_model(
                config,
                hidden_act=args.hidden_act,
                num_experts_per_tok=args.num_experts_per_tok,
            )
            profile = validate_original_model(config)
        elif args.mode == FLOAT_ALL_EXPERTS_MODE:
            configure_float_all_experts_model(
                config,
                hidden_act=args.hidden_act,
                num_experts_per_tok=args.num_experts_per_tok,
            )
            profile = validate_float_all_experts_model(config)
        else:
            raise AssertionError(f"unhandled native float mode: {args.mode}")
        print(
            json.dumps(
                {
                    "execution_profile": {
                        **profile,
                        "checkpoint_dtype": True,
                        "checkpoint_default_attention": True,
                        "fixed_point": False,
                        "top_t_capacity_factor": 2.0,
                    }
                },
                sort_keys=True,
            )
        )
    else:
        from fixed_point import (
            configure_fixed_point,
            configure_routing_trace,
            validate_moe_routing,
        )

        configure_fixed_point(
            config,
            args.mode,
            stats=args.fxp_stats,
            hidden_act=args.hidden_act,
            num_experts_per_tok=args.num_experts_per_tok,
        )
        validate_moe_routing(config)
        configure_routing_trace(
            config,
            output_path=args.routing_trace_out,
            reference_path=args.routing_trace_reference,
        )

    load_kwargs = build_model_load_kwargs(args, config)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    if args.mode in FIXED_MODES:
        from fixed_point import prepare_model_for_fixed_point

        prepare_model_for_fixed_point(model)

    return HFLM(
        pretrained=model,
        tokenizer=args.model,
        batch_size=args.batch_size,
        trust_remote_code=True,
        parallelize=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Balanced Routing MoE Model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--mode",
        type=mode_argument,
        default="original",
        choices=tuple(sorted(EXPERIMENT_MODES)),
        help="one of the six routing/activation cells in the paper",
    )
    parser.add_argument("--num_experts_per_tok", type=int)
    parser.add_argument("--hidden_act", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fxp_stats", action="store_true")
    parser.add_argument("--routing_trace_out", type=str)
    parser.add_argument("--routing_trace_reference", type=str)

    args = parser.parse_args()

    model = load_model(args)
    test_model(dataset_name=args.dataset, model=model, limit=args.limit)
    if args.fxp_stats:
        print(
            json.dumps(
                getattr(model.model.config, "_fxp_stats_data", {}),
                indent=2,
                sort_keys=True,
            )
        )
    if args.mode in CONTROLLED_MODES:
        from fixed_point import finalize_routing_trace

        routing_summary = finalize_routing_trace(model.model.config)
        if routing_summary["enabled"]:
            print(json.dumps({"routing_trace": routing_summary}, indent=2))


if __name__ == "__main__":
    main()
