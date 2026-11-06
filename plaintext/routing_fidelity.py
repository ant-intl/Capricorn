from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModelForCausalLM

from model_profiles import MODEL_ARCHITECTURE_PROFILES
from original_mode import (
    configure_float_all_experts_model,
    configure_original_model,
    validate_float_all_experts_model,
    validate_original_model,
)


DEFAULT_DATASETS = (
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
)
SUPPORTED_DATASETS = frozenset(DEFAULT_DATASETS)
SPARSE_BLOCK_CLASSES = {
    "deepseek": "DeepseekMoE",
    "qwen": "Qwen2MoeSparseMoeBlock",
    "olmoe": "OlmoeSparseMoeBlock",
}


@dataclass(frozen=True)
class RoutingSnapshot:
    original_assignment: torch.Tensor
    all_experts_assignment: torch.Tensor
    original_candidate_mask: torch.Tensor


def _assignment_from_scores(
    probabilities: torch.Tensor,
    *,
    active_top_k: int,
    capacity_top_k: int,
    capacity_factor: float,
    norm_topk_prob: bool,
    routing_weight_dtype: torch.dtype,
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("router probabilities must have shape [tokens, experts]")
    num_tokens, num_experts = probabilities.shape
    if not 1 <= active_top_k <= num_experts:
        raise ValueError(
            f"active_top_k must be in [1, {num_experts}], got {active_top_k}"
        )
    if capacity_top_k <= 0:
        raise ValueError("capacity_top_k must be positive")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")

    selected_weights, selected_experts = torch.topk(
        probabilities,
        active_top_k,
        dim=-1,
    )
    if norm_topk_prob:
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1,
            keepdim=True,
        )
    # Match Qwen2MoeSparseMoeBlock/OlmoeSparseMoeBlock exactly: token-wise
    # Top-k and optional normalization happen in FP32, then selected weights
    # are cast back to the hidden-state dtype before expert-wise Top-t.
    selected_weights = selected_weights.to(routing_weight_dtype)

    capacity = math.ceil(
        num_tokens * capacity_top_k * capacity_factor / num_experts
    )
    assignment = torch.zeros_like(probabilities, dtype=torch.bool)
    # The native blocks permute one-hot assignments to
    # [experts, rank, tokens]. torch.where therefore enumerates candidates in
    # rank-major order. This order is observable when BF16 weights tie.
    expert_mask = F.one_hot(
        selected_experts,
        num_classes=num_experts,
    ).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        rank_indices, token_indices = torch.where(expert_mask[expert_idx])
        if token_indices.numel() > capacity:
            candidate_weights = selected_weights[token_indices, rank_indices]
            kept = torch.topk(
                candidate_weights,
                capacity,
                dim=0,
                largest=True,
                sorted=True,
            ).indices
            token_indices = token_indices[kept]
        assignment[token_indices, expert_idx] = True
    return assignment


def _deepseek_assignment_from_selection(
    selected_experts: torch.Tensor,
    selected_weights: torch.Tensor,
    *,
    num_experts: int,
    capacity_top_k: int,
    capacity_factor: float,
) -> torch.Tensor:
    if selected_experts.ndim != 2:
        raise ValueError("selected experts must have shape [tokens, top_k]")
    if selected_weights.shape != selected_experts.shape:
        raise ValueError("selected expert indices and weights must align")
    num_tokens, active_top_k = selected_experts.shape
    if not 1 <= active_top_k <= num_experts:
        raise ValueError(
            f"active_top_k must be in [1, {num_experts}], "
            f"got {active_top_k}"
        )
    capacity = math.ceil(
        num_tokens * capacity_top_k * capacity_factor / num_experts
    )

    # Match DeepseekMoE.moe_infer exactly. Subtracting the gate weight from
    # the integer expert id groups entries by expert and ranks each group by
    # descending weight. argsort tie behavior and flattened token indexing
    # are intentionally preserved.
    flat_experts = selected_experts.reshape(-1)
    flat_weights = selected_weights.reshape(-1)
    weight_indices = flat_experts.float() - flat_weights
    sorted_indices = weight_indices.argsort()
    token_indices = sorted_indices // active_top_k
    cumulative = (
        flat_experts.bincount(minlength=num_experts).cpu().cumsum(0)
    )
    assignment = torch.zeros(
        (num_tokens, num_experts),
        dtype=torch.bool,
        device=selected_experts.device,
    )
    start = 0
    for expert_idx, end_tensor in enumerate(cumulative):
        end = int(end_tensor.item())
        kept_end = min(end, start + capacity)
        assignment[
            token_indices[start:kept_end],
            expert_idx,
        ] = True
        start = end
    return assignment


def build_deepseek_routing_snapshot(
    router_logits: torch.Tensor,
    *,
    native_top_k: int,
    capacity_top_k: int,
    capacity_factor: float,
    norm_topk_prob: bool,
    native_selected_experts: torch.Tensor | None = None,
    native_selected_weights: torch.Tensor | None = None,
) -> RoutingSnapshot:
    if router_logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    # DeepSeek intentionally does not request an FP32 softmax. Preserve the
    # checkpoint dtype (BF16 for the evaluated model).
    probabilities = router_logits.detach().softmax(dim=-1)
    num_experts = probabilities.shape[-1]
    if not 1 <= native_top_k <= num_experts:
        raise ValueError(
            f"native_top_k must be in [1, {num_experts}], got {native_top_k}"
        )

    if native_selected_experts is None or native_selected_weights is None:
        native_selected_weights, native_selected_experts = torch.topk(
            probabilities,
            native_top_k,
            dim=-1,
            sorted=False,
        )
        if native_top_k > 1 and norm_topk_prob:
            native_selected_weights = native_selected_weights / (
                native_selected_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
    else:
        native_selected_experts = native_selected_experts.detach()
        native_selected_weights = native_selected_weights.detach()

    all_weights, all_experts = torch.topk(
        probabilities,
        num_experts,
        dim=-1,
        sorted=False,
    )
    if num_experts > 1 and norm_topk_prob:
        all_weights = all_weights / (
            all_weights.sum(dim=-1, keepdim=True) + 1e-20
        )

    original_assignment = _deepseek_assignment_from_selection(
        native_selected_experts,
        native_selected_weights,
        num_experts=num_experts,
        capacity_top_k=capacity_top_k,
        capacity_factor=capacity_factor,
    )
    all_experts_assignment = _deepseek_assignment_from_selection(
        all_experts,
        all_weights,
        num_experts=num_experts,
        capacity_top_k=capacity_top_k,
        capacity_factor=capacity_factor,
    )
    original_candidate_mask = torch.zeros_like(
        probabilities,
        dtype=torch.bool,
    )
    original_candidate_mask.scatter_(1, native_selected_experts, True)
    return RoutingSnapshot(
        original_assignment=original_assignment.cpu().contiguous(),
        all_experts_assignment=all_experts_assignment.cpu().contiguous(),
        original_candidate_mask=original_candidate_mask.cpu().contiguous(),
    )


def build_routing_snapshot(
    router_logits: torch.Tensor,
    *,
    native_top_k: int,
    capacity_top_k: int,
    capacity_factor: float,
    norm_topk_prob: bool,
    routing_weight_dtype: torch.dtype,
) -> RoutingSnapshot:
    if router_logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    probabilities = torch.softmax(
        router_logits.detach().to(torch.float32),
        dim=-1,
    )
    num_experts = probabilities.shape[-1]
    if not 1 <= native_top_k <= num_experts:
        raise ValueError(
            f"native_top_k must be in [1, {num_experts}], got {native_top_k}"
        )

    original_assignment = _assignment_from_scores(
        probabilities,
        active_top_k=native_top_k,
        capacity_top_k=capacity_top_k,
        capacity_factor=capacity_factor,
        norm_topk_prob=norm_topk_prob,
        routing_weight_dtype=routing_weight_dtype,
    )
    all_experts_assignment = _assignment_from_scores(
        probabilities,
        active_top_k=num_experts,
        capacity_top_k=capacity_top_k,
        capacity_factor=capacity_factor,
        norm_topk_prob=norm_topk_prob,
        routing_weight_dtype=routing_weight_dtype,
    )
    native_indices = torch.topk(
        probabilities,
        native_top_k,
        dim=-1,
    ).indices
    original_candidate_mask = torch.zeros_like(
        probabilities,
        dtype=torch.bool,
    )
    original_candidate_mask.scatter_(1, native_indices, True)

    return RoutingSnapshot(
        original_assignment=original_assignment.cpu().contiguous(),
        all_experts_assignment=all_experts_assignment.cpu().contiguous(),
        original_candidate_mask=original_candidate_mask.cpu().contiguous(),
    )


def filter_routing_snapshot(
    snapshot: RoutingSnapshot,
    *,
    batch_size: int,
    sequence_length: int,
    valid_lengths: list[int],
) -> RoutingSnapshot:
    if len(valid_lengths) != batch_size:
        raise ValueError(
            "one valid sequence length is required for every batch row"
        )
    expected_tokens = batch_size * sequence_length

    def select_valid(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2 or mask.shape[0] != expected_tokens:
            raise ValueError(
                "routing mask must have shape "
                f"[{expected_tokens}, experts], got {tuple(mask.shape)}"
            )
        lengths = torch.tensor(valid_lengths, dtype=torch.long)
        if bool(((lengths <= 0) | (lengths > sequence_length)).any()):
            raise ValueError(
                "valid sequence lengths must be in "
                f"[1, {sequence_length}]"
            )
        positions = torch.arange(sequence_length).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        experts = mask.shape[1]
        return (
            mask.reshape(batch_size, sequence_length, experts)[valid]
            .reshape(-1, experts)
            .contiguous()
        )

    return RoutingSnapshot(
        original_assignment=select_valid(snapshot.original_assignment),
        all_experts_assignment=select_valid(snapshot.all_experts_assignment),
        original_candidate_mask=select_valid(
            snapshot.original_candidate_mask
        ),
    )


class FidelityAccumulator:
    def __init__(self) -> None:
        self.dataset = ""
        self.reset("")

    def reset(self, dataset: str) -> None:
        self.dataset = dataset
        self.routing_layers = 0
        self.model_calls = 0
        self.original_assignments = 0
        self.all_experts_assignments = 0
        self.assignment_intersection = 0
        self.outside_original_top_k = 0
        self.logit_difference_squared = 0.0
        self.original_logit_squared = 0.0
        self.top1_matches = 0
        self.logit_positions = 0

    def record_model_call(self) -> None:
        self.model_calls += 1

    def update_routing(self, snapshot: RoutingSnapshot) -> None:
        original = snapshot.original_assignment
        all_experts = snapshot.all_experts_assignment
        candidates = snapshot.original_candidate_mask
        if original.shape != all_experts.shape or original.shape != candidates.shape:
            raise ValueError("routing snapshot masks must have identical shapes")

        self.routing_layers += 1
        self.original_assignments += int(original.sum().item())
        self.all_experts_assignments += int(all_experts.sum().item())
        self.assignment_intersection += int(
            (original & all_experts).sum().item()
        )
        self.outside_original_top_k += int(
            (all_experts & ~candidates).sum().item()
        )

    def update_logits(
        self,
        original_logits: torch.Tensor,
        all_experts_logits: torch.Tensor,
    ) -> None:
        if original_logits.shape != all_experts_logits.shape:
            raise ValueError(
                "paired logits must have identical shapes: "
                f"{tuple(original_logits.shape)} != "
                f"{tuple(all_experts_logits.shape)}"
            )
        if original_logits.ndim != 3:
            raise ValueError("logits must have shape [batch, sequence, vocab]")
        if original_logits.shape[1] == 0:
            return

        reference = original_logits.to(torch.float32)
        comparison = all_experts_logits.to(
            device=reference.device,
            dtype=torch.float32,
        )
        difference = comparison - reference
        self.logit_difference_squared += float(
            difference.square().sum().item()
        )
        self.original_logit_squared += float(reference.square().sum().item())
        reference_top1 = reference.argmax(dim=-1)
        comparison_top1 = comparison.argmax(dim=-1)
        self.top1_matches += int(
            (reference_top1 == comparison_top1).sum().item()
        )
        self.logit_positions += int(reference_top1.numel())

    def finalize(self) -> dict[str, float | int | str]:
        if self.original_assignments <= 0:
            raise RuntimeError(
                f"{self.dataset}: no original routing assignments recorded"
            )
        if self.all_experts_assignments <= 0:
            raise RuntimeError(
                f"{self.dataset}: no all-experts assignments recorded"
            )
        if self.original_logit_squared <= 0:
            raise RuntimeError(
                f"{self.dataset}: no nonzero original logits recorded"
            )
        if self.logit_positions <= 0:
            raise RuntimeError(
                f"{self.dataset}: no teacher-forced logit positions recorded"
            )
        return {
            "dataset": self.dataset,
            "assignment_overlap": (
                self.assignment_intersection / self.original_assignments
            ),
            "outside_original_top_k_rate": (
                self.outside_original_top_k
                / self.all_experts_assignments
            ),
            "logit_relative_l2_error": math.sqrt(
                self.logit_difference_squared
                / self.original_logit_squared
            ),
            "top1_agreement": (
                self.top1_matches / self.logit_positions
            ),
            "counts": {
                "routing_layers": self.routing_layers,
                "model_calls": self.model_calls,
                "original_assignments": self.original_assignments,
                "all_experts_assignments": self.all_experts_assignments,
                "assignment_intersection": self.assignment_intersection,
                "outside_original_top_k": self.outside_original_top_k,
                "logit_positions": self.logit_positions,
                "top1_matches": self.top1_matches,
                "logit_difference_squared": self.logit_difference_squared,
                "original_logit_squared": self.original_logit_squared,
            },
        }


class CounterfactualRoutingRecorder:
    def __init__(
        self,
        model,
        accumulator: FidelityAccumulator,
        *,
        model_kind: str,
        native_top_k: int,
        total_experts: int,
    ) -> None:
        self.accumulator = accumulator
        self.model_kind = model_kind
        self.native_top_k = native_top_k
        self.total_experts = total_experts
        self.active = False
        self.handles = []
        self.current_snapshots: list[RoutingSnapshot] = []
        self.pending_snapshots: list[RoutingSnapshot] | None = None
        expected_class = SPARSE_BLOCK_CLASSES.get(model_kind)
        if expected_class is None:
            raise ValueError(f"unsupported routing recorder model: {model_kind}")
        for name, module in model.named_modules():
            if module.__class__.__name__ != expected_class:
                continue
            if model_kind == "deepseek":
                handle = module.gate.register_forward_hook(
                    self._make_deepseek_hook(name, module)
                )
            else:
                handle = module.register_forward_hook(
                    self._make_sparse_block_hook(name)
                )
            self.handles.append(handle)
        if not self.handles:
            raise ValueError(f"no {expected_class} modules found")

    def _make_sparse_block_hook(self, module_name: str):
        def hook(module, inputs, output):
            if not self.active:
                return
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(
                    f"{module_name}: expected (hidden_states, router_logits)"
                )
            if int(module.top_k) != self.native_top_k:
                raise RuntimeError(
                    f"{module_name}: original top_k={module.top_k}, "
                    f"expected {self.native_top_k}"
                )
            if int(module.num_experts) != self.total_experts:
                raise RuntimeError(
                    f"{module_name}: total experts={module.num_experts}, "
                    f"expected {self.total_experts}"
                )
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(
                    f"{module_name}: hidden-state tensor input is missing"
                )
            capacity_factor = float(module.capacity_factor.detach().item())
            snapshot = build_routing_snapshot(
                output[1],
                native_top_k=self.native_top_k,
                capacity_top_k=int(module.top_t_capacity_k),
                capacity_factor=capacity_factor,
                norm_topk_prob=bool(module.norm_topk_prob),
                routing_weight_dtype=inputs[0].dtype,
            )
            self.current_snapshots.append(snapshot)

        return hook

    def _make_deepseek_hook(self, module_name: str, parent_module):
        def hook(gate, inputs, output):
            if not self.active:
                return
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(
                    f"{module_name}.gate: expected "
                    "(selected_experts, selected_weights, aux_loss)"
                )
            if int(gate.top_k) != self.native_top_k:
                raise RuntimeError(
                    f"{module_name}.gate: original top_k={gate.top_k}, "
                    f"expected {self.native_top_k}"
                )
            if int(gate.n_routed_experts) != self.total_experts:
                raise RuntimeError(
                    f"{module_name}.gate: total experts="
                    f"{gate.n_routed_experts}, expected {self.total_experts}"
                )
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(
                    f"{module_name}.gate: hidden-state tensor input is missing"
                )
            hidden_states = inputs[0]
            flattened = hidden_states.view(-1, hidden_states.shape[-1])
            with torch.no_grad():
                router_logits = F.linear(flattened, gate.weight, None)
            capacity_top_k = int(
                getattr(
                    parent_module.config,
                    "top_t_capacity_k",
                    parent_module.num_experts_per_tok,
                )
            )
            capacity_factor = float(
                parent_module.capacity_factor.detach().item()
            )
            snapshot = build_deepseek_routing_snapshot(
                router_logits,
                native_top_k=self.native_top_k,
                capacity_top_k=capacity_top_k,
                capacity_factor=capacity_factor,
                norm_topk_prob=bool(gate.norm_topk_prob),
                native_selected_experts=output[0],
                native_selected_weights=output[1],
            )
            self.current_snapshots.append(snapshot)

        return hook

    def begin_call(self) -> None:
        if self.active:
            raise RuntimeError("routing recorder call already active")
        if self.pending_snapshots is not None:
            raise RuntimeError(
                "previous routing snapshots were not consumed by lm-eval"
            )
        self.current_snapshots = []
        self.active = True

    def end_call(self) -> None:
        if not self.active:
            raise RuntimeError("routing recorder call is not active")
        self.active = False
        if not self.current_snapshots:
            raise RuntimeError("no routing snapshots recorded for model call")
        if len(self.current_snapshots) != len(self.handles):
            raise RuntimeError(
                "incomplete routing capture: recorded "
                f"{len(self.current_snapshots)} of {len(self.handles)} "
                "sparse layers"
            )
        self.pending_snapshots = self.current_snapshots
        self.current_snapshots = []

    def abort_call(self) -> None:
        self.active = False
        self.current_snapshots = []
        self.pending_snapshots = None

    def commit_call(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        valid_lengths: list[int],
    ) -> None:
        if self.pending_snapshots is None:
            raise RuntimeError("routing snapshots are missing for model call")
        for snapshot in self.pending_snapshots:
            filtered = filter_routing_snapshot(
                snapshot,
                batch_size=batch_size,
                sequence_length=sequence_length,
                valid_lengths=valid_lengths,
            )
            self.accumulator.update_routing(filtered)
        self.pending_snapshots = None

    def close(self) -> None:
        if self.active or self.pending_snapshots is not None:
            raise RuntimeError("unconsumed routing snapshots remain at shutdown")
        self.active = False
        self.current_snapshots = []
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PairedFidelityHFLM(HFLM):
    def __init__(
        self,
        *,
        original_model,
        all_experts_model,
        tokenizer,
        batch_size: int,
        accumulator: FidelityAccumulator,
        model_kind: str,
        native_top_k: int,
        total_experts: int,
    ) -> None:
        _validate_batch_size(batch_size)
        super().__init__(
            pretrained=original_model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            trust_remote_code=True,
            parallelize=False,
        )
        self.all_experts_model = all_experts_model
        self.accumulator = accumulator
        self.routing_recorder = CounterfactualRoutingRecorder(
            self.model,
            accumulator,
            model_kind=model_kind,
            native_top_k=native_top_k,
            total_experts=total_experts,
        )
        embedding = self.all_experts_model.get_input_embeddings()
        self.all_experts_input_device = next(embedding.parameters()).device
        self.pending_logits = None
        self.pending_row_index = 0
        self.pending_valid_lengths: list[int] = []

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inps.ndim != 2 or inps.shape[0] <= 0:
            raise RuntimeError("causal inputs must have shape [batch, sequence]")
        if attn_mask is not None or labels is not None:
            raise ValueError("routing fidelity evaluation supports causal inputs only")

        self.routing_recorder.begin_call()
        try:
            original_logits = super()._model_call(inps)
        except Exception:
            self.routing_recorder.abort_call()
            raise
        self.routing_recorder.end_call()

        comparison_inputs = inps.to(self.all_experts_input_device)
        try:
            with torch.no_grad():
                all_experts_logits = self.all_experts_model(
                    comparison_inputs
                ).logits
        except Exception:
            self.routing_recorder.abort_call()
            raise
        if self.pending_logits is not None:
            raise RuntimeError(
                "previous paired logits were not consumed by lm-eval"
            )
        self.pending_logits = (original_logits, all_experts_logits)
        self.pending_row_index = 0
        self.pending_valid_lengths = []
        self.accumulator.record_model_call()
        return original_logits

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        contlen: int | None = None,
        inplen: int | None = None,
    ) -> torch.Tensor:
        selected = super()._select_cont_toks(
            logits,
            contlen=contlen,
            inplen=inplen,
        )
        if self.pending_logits is None:
            raise RuntimeError("paired logits are missing for continuation slice")
        if not contlen or not inplen:
            raise RuntimeError(
                "causal continuation length and input length are required"
            )
        original_logits, all_experts_logits = self.pending_logits
        batch_size, sequence_length, _ = original_logits.shape
        row_index = self.pending_row_index
        if row_index >= batch_size:
            raise RuntimeError("too many continuation slices for paired batch")
        if all_experts_logits.shape != original_logits.shape:
            raise RuntimeError("paired logits changed shape after model call")
        if inplen > sequence_length:
            raise RuntimeError(
                f"input length {inplen} exceeds padded length {sequence_length}"
            )
        start = inplen - contlen
        stop = inplen
        self.accumulator.update_logits(
            original_logits[row_index : row_index + 1, start:stop, :],
            all_experts_logits[row_index : row_index + 1, start:stop, :],
        )
        self.pending_valid_lengths.append(inplen)
        self.pending_row_index += 1
        if self.pending_row_index == batch_size:
            self.routing_recorder.commit_call(
                batch_size=batch_size,
                sequence_length=sequence_length,
                valid_lengths=self.pending_valid_lengths,
            )
            self.pending_logits = None
            self.pending_row_index = 0
            self.pending_valid_lengths = []
        return selected

    def close_fidelity(self) -> None:
        if self.pending_logits is not None:
            raise RuntimeError("unconsumed paired logits remain at shutdown")
        self.routing_recorder.close()


def _load_pair(model_path: str):
    original_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    configure_original_model(original_config)
    original_profile = validate_original_model(original_config)

    all_experts_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    configure_float_all_experts_model(all_experts_config)
    all_experts_profile = validate_float_all_experts_model(
        all_experts_config
    )

    model_name = str(original_profile["model"])
    if all_experts_profile["model"] != model_name:
        raise ValueError("paired configurations resolved to different models")
    profile = MODEL_ARCHITECTURE_PROFILES[model_name]
    if original_profile["active_experts"] != int(profile["native_top_k"]):
        raise ValueError(f"unexpected {model_name} native top-k")
    if all_experts_profile["active_experts"] != int(
        profile["total_experts"]
    ):
        raise ValueError(f"unexpected {model_name} total expert count")

    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=original_config,
        **common_kwargs,
    )
    original_model.eval()
    all_experts_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=all_experts_config,
        **common_kwargs,
    )
    all_experts_model.eval()
    return (
        original_model,
        all_experts_model,
        original_profile,
        all_experts_profile,
    )


def _validate_datasets(datasets: Iterable[str]) -> tuple[str, ...]:
    result = tuple(datasets)
    if not result:
        raise ValueError("at least one dataset is required")
    unknown = sorted(set(result) - SUPPORTED_DATASETS)
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    if len(set(result)) != len(result):
        raise ValueError("datasets must not contain duplicates")
    return result


def _resolve_limit(limit: int | None, full: bool) -> int | None:
    if full:
        return None
    if limit is None:
        return 4
    if limit <= 0:
        raise ValueError("--limit must be positive")
    return limit


def _validate_batch_size(batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return batch_size


def _aggregate(results: list[dict]) -> dict[str, float | int]:
    successful = [entry for entry in results if entry["status"] == "success"]
    if not successful:
        return {"successful_datasets": 0}
    counts = [entry["metrics"]["counts"] for entry in successful]
    original_assignments = sum(item["original_assignments"] for item in counts)
    all_experts_assignments = sum(
        item["all_experts_assignments"] for item in counts
    )
    intersection = sum(item["assignment_intersection"] for item in counts)
    outside = sum(item["outside_original_top_k"] for item in counts)
    difference_squared = sum(
        item["logit_difference_squared"] for item in counts
    )
    original_squared = sum(
        item["original_logit_squared"] for item in counts
    )
    top1_matches = sum(item["top1_matches"] for item in counts)
    positions = sum(item["logit_positions"] for item in counts)
    return {
        "successful_datasets": len(successful),
        "assignment_overlap": intersection / original_assignments,
        "outside_original_top_k_rate": outside / all_experts_assignments,
        "logit_relative_l2_error": math.sqrt(
            difference_squared / original_squared
        ),
        "top1_agreement": top1_matches / positions,
        "original_assignments": original_assignments,
        "all_experts_assignments": all_experts_assignments,
        "logit_positions": positions,
    }


def _write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_output_path(path: Path, *, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {path}; pass --force to overwrite it"
        )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native and float-all-experts MoE routing fidelity"
        )
    )
    parser.add_argument(
        "--model",
        default="./OLMoE-1B-7B-0125-Instruct",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument("--limit", type=int)
    limit_group.add_argument(
        "--full",
        action="store_true",
        help="evaluate every document in each selected dataset",
    )
    parser.add_argument(
        "--output",
        default="routing_fidelity_results.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output file",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    datasets = _validate_datasets(args.datasets)
    try:
        _validate_batch_size(args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    try:
        evaluation_limit = _resolve_limit(args.limit, args.full)
    except ValueError as error:
        parser.error(str(error))
    try:
        output_path = _validate_output_path(
            Path(args.output),
            force=args.force,
        )
    except FileExistsError as error:
        parser.error(str(error))

    torch.manual_seed(0)
    (
        original_model,
        all_experts_model,
        original_profile,
        all_experts_profile,
    ) = _load_pair(args.model)
    accumulator = FidelityAccumulator()
    paired_model = PairedFidelityHFLM(
        original_model=original_model,
        all_experts_model=all_experts_model,
        tokenizer=args.model,
        batch_size=args.batch_size,
        accumulator=accumulator,
        model_kind=str(original_profile["model"]),
        native_top_k=int(original_profile["active_experts"]),
        total_experts=int(original_profile["total_experts"]),
    )

    payload = {
        "metadata": {
            "model": args.model,
            "model_kind": original_profile["model"],
            "datasets": list(datasets),
            "limit": evaluation_limit,
            "batch_size": args.batch_size,
            "routing_comparison_basis": (
                "same original-run hidden states and router scores"
            ),
            "logit_comparison_basis": (
                "paired complete original and float-all-experts forwards"
            ),
            "smoke_test": evaluation_limit is not None,
            "original_profile": original_profile,
            "float_all_experts_profile": all_experts_profile,
        },
        "results": [],
    }
    try:
        for dataset in datasets:
            print(f"=== routing fidelity: {dataset} ===", flush=True)
            accumulator.reset(dataset)
            try:
                simple_evaluate(
                    model=paired_model,
                    tasks=[dataset],
                    write_out=False,
                    limit=evaluation_limit,
                )
                metrics = accumulator.finalize()
                entry = {
                    "dataset": dataset,
                    "status": "success",
                    "metrics": metrics,
                }
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "assignment_overlap": metrics[
                                "assignment_overlap"
                            ],
                            "outside_original_top_k_rate": metrics[
                                "outside_original_top_k_rate"
                            ],
                            "logit_relative_l2_error": metrics[
                                "logit_relative_l2_error"
                            ],
                            "top1_agreement": metrics["top1_agreement"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as error:
                entry = {
                    "dataset": dataset,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(
                    json.dumps(entry, sort_keys=True),
                    flush=True,
                )
            payload["results"].append(entry)
            payload["aggregate"] = _aggregate(payload["results"])
            _write_results(output_path, payload)
    finally:
        paired_model.close_fidelity()

    failures = [
        entry["dataset"]
        for entry in payload["results"]
        if entry["status"] != "success"
    ]
    if failures:
        raise SystemExit(f"fidelity evaluation failed for: {failures}")


if __name__ == "__main__":
    main()
