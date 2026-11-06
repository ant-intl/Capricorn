# Copyright 2026 Ant International, Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-only JAX frontend for the fixed V-MoE S/32 research model."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from capricorn_models.vmoe_intrinsic import vmoe_block
from capricorn_models.vmoe_intrinsic import vmoe_trace_marker

_TOKEN_COUNT = 50
_MODEL_DIM = 512
_EXPERT_COUNT = 8
_TOP_K = 2
_EXPERT_CAPACITY = 16
_OFFICIAL_EVAL_CAPACITY = 52


def _diagonal_matrix(rows, columns, scale):
    matrix = np.zeros((rows, columns), dtype=np.float32)
    diagonal = np.arange(min(rows, columns))
    matrix[diagonal, diagonal] = np.float32(scale)
    return matrix


def _attention_parameters():
    def projection(scale):
        kernel = _diagonal_matrix(_MODEL_DIM, _MODEL_DIM, scale)
        return {
            "kernel": kernel.reshape(_MODEL_DIM, 8, 64),
            "bias": np.zeros((8, 64), dtype=np.float32),
        }

    output_kernel = _diagonal_matrix(_MODEL_DIM, _MODEL_DIM, 0.10)
    return {
        "query": projection(0.04),
        "key": projection(0.04),
        "value": projection(0.10),
        "out": {
            "kernel": output_kernel.reshape(8, 64, _MODEL_DIM),
            "bias": np.zeros((_MODEL_DIM,), dtype=np.float32),
        },
    }


def _layer_norm_parameters():
    return {
        "scale": np.full((_MODEL_DIM,), 0.25, dtype=np.float32),
        "bias": np.zeros((_MODEL_DIM,), dtype=np.float32),
    }


def _dense_mlp_parameters():
    return {
        "Dense_0": {
            "kernel": _diagonal_matrix(_MODEL_DIM, 2048, 0.20),
            "bias": np.zeros((2048,), dtype=np.float32),
        },
        "Dense_1": {
            "kernel": _diagonal_matrix(2048, _MODEL_DIM, 0.20),
            "bias": np.zeros((_MODEL_DIM,), dtype=np.float32),
        },
    }


def _moe_parameters():
    router = np.zeros((_MODEL_DIM, _EXPERT_COUNT), dtype=np.float32)
    router[0, :] = np.linspace(-0.35, 0.35, _EXPERT_COUNT, dtype=np.float32)
    router[1, :] = np.asarray(
        [0.11, -0.09, 0.07, -0.05, 0.03, -0.01, 0.13, -0.15],
        dtype=np.float32,
    )
    w1 = np.zeros((_EXPERT_COUNT, _MODEL_DIM, 2048), dtype=np.float32)
    w2 = np.zeros((_EXPERT_COUNT, 2048, _MODEL_DIM), dtype=np.float32)
    diagonal = np.arange(_MODEL_DIM)
    for expert in range(_EXPERT_COUNT):
        scale = np.float32(0.25 + expert * 0.01)
        w1[expert, diagonal, diagonal] = scale
        w2[expert, diagonal, diagonal] = scale
    return {
        "Router": {"dense": {"kernel": router}},
        "Mlp": {
            "Dense_0": {
                "kernel": w1,
                "bias": np.zeros((_EXPERT_COUNT, 2048), dtype=np.float32),
            },
            "Dense_1": {
                "kernel": w2,
                "bias": np.zeros((_EXPERT_COUNT, _MODEL_DIM), dtype=np.float32),
            },
        },
    }


def make_synthetic_parameters():
    """Builds deterministic official-shaped inference parameters."""
    patch_kernel = np.zeros((32, 32, 3, _MODEL_DIM), dtype=np.float32)
    for channel in range(3):
        patch_kernel[:, :, channel, channel] = np.float32(0.01)

    position_embedding = np.zeros((1, _TOKEN_COUNT, _MODEL_DIM), dtype=np.float32)
    position_embedding[0, :, 0] = np.linspace(
        -0.05, 0.05, _TOKEN_COUNT, dtype=np.float32
    )
    position_embedding[0, :, 1] = np.linspace(
        0.025, -0.025, _TOKEN_COUNT, dtype=np.float32
    )

    encoder = {"posembed_input": {"pos_embedding": position_embedding}}
    for block_index in range(8):
        block = {
            "LayerNorm_0": _layer_norm_parameters(),
            "SelfAttention": _attention_parameters(),
            "LayerNorm_1": _layer_norm_parameters(),
        }
        if block_index in (5, 7):
            block["Moe"] = _moe_parameters()
        else:
            block["Mlp"] = _dense_mlp_parameters()
        encoder[f"encoderblock_{block_index}"] = block
    encoder["encoder_norm"] = _layer_norm_parameters()

    return {
        "embedding": {
            "kernel": patch_kernel,
            "bias": np.zeros((_MODEL_DIM,), dtype=np.float32),
        },
        "cls": np.zeros((1, 1, _MODEL_DIM), dtype=np.float32),
        "Encoder": encoder,
        "pre_logits": {
            "kernel": _diagonal_matrix(_MODEL_DIM, _MODEL_DIM, 0.20),
            "bias": np.zeros((_MODEL_DIM,), dtype=np.float32),
        },
        "head": {
            "kernel": _diagonal_matrix(_MODEL_DIM, 1000, 0.20),
            "bias": np.zeros((1000,), dtype=np.float32),
        },
    }


def make_synthetic_image():
    """Creates a deterministic bounded NHWC image for inference tests."""
    return np.linspace(-0.10, 0.10, 224 * 224 * 3, dtype=np.float32).reshape(
        1, 224, 224, 3
    )


def token_choice_silu_moe(token_embeddings, router_weights, expert_w1, expert_w2):
    """Runs the current Capricorn token-choice MoE semantics in pure JAX."""
    tokens = jnp.asarray(token_embeddings).reshape(_TOKEN_COUNT, _MODEL_DIM)
    router = jnp.asarray(router_weights)
    w1 = jnp.asarray(expert_w1)
    w2 = jnp.asarray(expert_w2)

    scores = jax.nn.softmax(tokens @ router, axis=-1)
    top_scores, top_indices = jax.lax.top_k(scores, _TOP_K)
    flat_scores = top_scores.reshape(-1)
    flat_indices = top_indices.reshape(-1)

    combined = jnp.zeros_like(tokens)
    for expert in range(_EXPERT_COUNT):
        priority = jnp.where(flat_indices == expert, flat_scores, 0.0)
        selected_scores, selected_positions = jax.lax.top_k(priority, _EXPERT_CAPACITY)
        selected_token_indices = selected_positions // _TOP_K
        selected_tokens = tokens[selected_token_indices]
        hidden = selected_tokens @ w1[expert]
        expert_output = (hidden / (1.0 + jnp.exp(-hidden))) @ w2[expert]
        token_one_hot = jax.nn.one_hot(
            selected_token_indices, _TOKEN_COUNT, dtype=tokens.dtype
        )
        combined += token_one_hot.T @ (expert_output * selected_scores[:, jnp.newaxis])

    return combined.reshape(1, _TOKEN_COUNT, _MODEL_DIM)


def token_choice_gelu_moe(
    token_embeddings, router_weights, expert_w1, expert_b1, expert_w2, expert_b2
):
    """Runs official eval Token-Choice routing with GELU and expert biases."""
    tokens = jnp.asarray(token_embeddings).reshape(_TOKEN_COUNT, _MODEL_DIM)
    router = jnp.asarray(router_weights)
    w1 = jnp.asarray(expert_w1)
    b1 = jnp.asarray(expert_b1)
    w2 = jnp.asarray(expert_w2)
    b2 = jnp.asarray(expert_b2)

    scores = jax.nn.softmax(tokens @ router, axis=-1)
    _, expert_indices = jax.lax.top_k(scores, _TOP_K)
    choice_major_indices = jnp.swapaxes(expert_indices, 0, 1).reshape(-1)
    expert_one_hot = jax.nn.one_hot(
        choice_major_indices, _EXPERT_COUNT, dtype=jnp.int32
    )
    buffer_indices = jnp.cumsum(expert_one_hot, axis=0) * expert_one_hot - 1
    buffer_indices = buffer_indices.reshape(
        _TOP_K, _TOKEN_COUNT, _EXPERT_COUNT
    ).transpose(1, 0, 2)
    buffer_indices = jnp.max(buffer_indices, axis=1)
    dispatch = jax.nn.one_hot(
        buffer_indices, _OFFICIAL_EVAL_CAPACITY, dtype=tokens.dtype
    )
    selected_tokens = jnp.einsum("sec,sd->ecd", dispatch, tokens)
    hidden = jnp.matmul(selected_tokens, w1) + b1[:, jnp.newaxis, :]
    expert_output = jnp.matmul(jax.nn.gelu(hidden), w2) + b2[:, jnp.newaxis, :]
    combine_weights = scores[..., jnp.newaxis] * dispatch
    combined = jnp.einsum("sec,ecd->sd", combine_weights, expert_output)
    return combined.reshape(1, _TOKEN_COUNT, _MODEL_DIM)


def token_choice_silu_bias_moe(
    token_embeddings, router_weights, expert_w1, expert_b1, expert_w2, expert_b2
):
    """Runs eval-capacity Token-Choice with paper SiLU and expert biases."""
    tokens = jnp.asarray(token_embeddings).reshape(_TOKEN_COUNT, _MODEL_DIM)
    router = jnp.asarray(router_weights)
    w1 = jnp.asarray(expert_w1)
    b1 = jnp.asarray(expert_b1)
    w2 = jnp.asarray(expert_w2)
    b2 = jnp.asarray(expert_b2)

    scores = jax.nn.softmax(tokens @ router, axis=-1)
    top_scores, top_indices = jax.lax.top_k(scores, _TOP_K)
    flat_scores = top_scores.reshape(-1)
    flat_indices = top_indices.reshape(-1)
    combined = jnp.zeros_like(tokens)
    for expert in range(_EXPERT_COUNT):
        priority = jnp.where(flat_indices == expert, flat_scores, 0.0)
        selected_scores, selected_positions = jax.lax.top_k(
            priority, _OFFICIAL_EVAL_CAPACITY
        )
        selected_token_indices = selected_positions // _TOP_K
        selected_tokens = tokens[selected_token_indices]
        hidden = selected_tokens @ w1[expert] + b1[expert]
        activated = hidden / (1.0 + jnp.exp(-hidden))
        expert_output = activated @ w2[expert] + b2[expert]
        token_one_hot = jax.nn.one_hot(
            selected_token_indices, _TOKEN_COUNT, dtype=tokens.dtype
        )
        combined += token_one_hot.T @ (
            expert_output * selected_scores[:, jnp.newaxis]
        )
    return combined.reshape(1, _TOKEN_COUNT, _MODEL_DIM)


def expert_choice_silu_moe(token_embeddings, router_weights, expert_w1, expert_w2):
    """Runs Capricorn Expert-Choice top-16 routing in pure JAX."""
    tokens = jnp.asarray(token_embeddings).reshape(_TOKEN_COUNT, _MODEL_DIM)
    router = jnp.asarray(router_weights)
    w1 = jnp.asarray(expert_w1)
    w2 = jnp.asarray(expert_w2)

    scores = jax.nn.softmax(tokens @ router, axis=-1)
    selected_scores, selected_indices = jax.lax.top_k(scores.T, _EXPERT_CAPACITY)
    selected_tokens = tokens[selected_indices]
    hidden = jnp.matmul(selected_tokens, w1)
    expert_output = jnp.matmul(hidden / (1.0 + jnp.exp(-hidden)), w2)
    token_one_hot = jax.nn.one_hot(selected_indices, _TOKEN_COUNT, dtype=tokens.dtype)
    combined = jnp.einsum(
        "etm,etd->md",
        token_one_hot,
        expert_output * selected_scores[..., jnp.newaxis],
    )
    return combined.reshape(1, _TOKEN_COUNT, _MODEL_DIM)


def expert_choice_gelu_moe(
    token_embeddings, router_weights, expert_w1, expert_b1, expert_w2, expert_b2
):
    """Runs the eval-capacity Expert-Choice research comparison in pure JAX."""
    tokens = jnp.asarray(token_embeddings).reshape(_TOKEN_COUNT, _MODEL_DIM)
    scores = jax.nn.softmax(tokens @ jnp.asarray(router_weights), axis=-1)
    selected_scores, selected_indices = jax.lax.top_k(scores.T, 28)
    selected_tokens = tokens[selected_indices]
    hidden = (
        jnp.matmul(selected_tokens, jnp.asarray(expert_w1))
        + jnp.asarray(expert_b1)[:, jnp.newaxis, :]
    )
    expert_output = (
        jnp.matmul(jax.nn.gelu(hidden), jnp.asarray(expert_w2))
        + jnp.asarray(expert_b2)[:, jnp.newaxis, :]
    )
    token_one_hot = jax.nn.one_hot(selected_indices, _TOKEN_COUNT, dtype=tokens.dtype)
    combined = jnp.einsum(
        "etm,etd->md",
        token_one_hot,
        expert_output * selected_scores[..., jnp.newaxis],
    )
    return combined.reshape(1, _TOKEN_COUNT, _MODEL_DIM)


def expert_choice_silu_bias_moe(
    token_embeddings, router_weights, expert_w1, expert_b1, expert_w2, expert_b2
):
    """Runs top-28 Expert-Choice with paper _SiLU semantics and biases."""
    tokens = jnp.asarray(token_embeddings).reshape(_TOKEN_COUNT, _MODEL_DIM)
    scores = jax.nn.softmax(tokens @ jnp.asarray(router_weights), axis=-1)
    selected_scores, selected_indices = jax.lax.top_k(scores.T, 28)
    selected_tokens = tokens[selected_indices]
    hidden = (
        jnp.matmul(selected_tokens, jnp.asarray(expert_w1))
        + jnp.asarray(expert_b1)[:, jnp.newaxis, :]
    )
    activated = hidden / (1.0 + jnp.exp(-hidden))
    expert_output = (
        jnp.matmul(activated, jnp.asarray(expert_w2))
        + jnp.asarray(expert_b2)[:, jnp.newaxis, :]
    )
    token_one_hot = jax.nn.one_hot(selected_indices, _TOKEN_COUNT, dtype=tokens.dtype)
    combined = jnp.einsum(
        "etm,etd->md",
        token_one_hot,
        expert_output * selected_scores[..., jnp.newaxis],
    )
    return combined.reshape(1, _TOKEN_COUNT, _MODEL_DIM)


_PLAINTEXT_MOE_BY_ROUTING_MODE = {
    "token_choice": token_choice_silu_moe,
    "expert_choice": expert_choice_silu_moe,
}

_PLAINTEXT_GELU_MOE_BY_ROUTING_MODE = {
    "token_choice": token_choice_gelu_moe,
    "expert_choice": expert_choice_gelu_moe,
}

_PLAINTEXT_SILU_BIAS_MOE_BY_ROUTING_MODE = {
    "token_choice": token_choice_silu_bias_moe,
    "expert_choice": expert_choice_silu_bias_moe,
}


def _moe_for_routing_mode(routing_mode):
    try:
        return _PLAINTEXT_MOE_BY_ROUTING_MODE[routing_mode]
    except KeyError as error:
        supported = ", ".join(sorted(_PLAINTEXT_MOE_BY_ROUTING_MODE))
        raise ValueError(
            f"unsupported routing_mode {routing_mode!r}; expected one of: "
            f"{supported}"
        ) from error


def _layer_norm(value, parameters, epsilon=1e-6):
    mean = jnp.mean(value, axis=-1, keepdims=True)
    centered = value - mean
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    normalized = (value - mean) * jax.lax.rsqrt(variance + epsilon)
    return normalized * parameters["scale"] + parameters["bias"]


def _spu_batched_matmul(lhs, rhs):
    """Lowers a batch-1 matrix product to SPU's supported DotGeneral form."""
    lhs = jnp.asarray(lhs)
    rhs = jnp.asarray(rhs)
    contracting_size = lhs.shape[-1]
    lhs_3d = lhs.reshape(1, -1, contracting_size)
    rhs_3d = rhs.reshape(1, contracting_size, -1)
    return jax.lax.dot_general(
        lhs_3d,
        rhs_3d,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )


def _attention(value, parameters):
    def project(name):
        projection = parameters[name]
        projected = _spu_batched_matmul(
            value.reshape(_TOKEN_COUNT, _MODEL_DIM),
            projection["kernel"].reshape(_MODEL_DIM, _MODEL_DIM),
        )
        projected += projection["bias"].reshape(_MODEL_DIM)
        return projected.reshape(1, _TOKEN_COUNT, 8, 64)

    query = project("query")
    key = project("key")
    projected_value = project("value")
    attended_heads = []
    for head in range(8):
        attention_logits = _spu_batched_matmul(
            query[0, :, head, :], key[0, :, head, :].T
        ).reshape(_TOKEN_COUNT, _TOKEN_COUNT)
        attention_logits /= jnp.asarray(8.0, dtype=value.dtype)
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        attended_heads.append(
            _spu_batched_matmul(
                attention_weights, projected_value[0, :, head, :]
            ).reshape(_TOKEN_COUNT, 64)
        )
    attended = jnp.stack(attended_heads, axis=1).reshape(_TOKEN_COUNT, _MODEL_DIM)
    output = parameters["out"]
    projected = _spu_batched_matmul(
        attended, output["kernel"].reshape(_MODEL_DIM, _MODEL_DIM)
    ).reshape(_TOKEN_COUNT, _MODEL_DIM)
    return (projected + output["bias"]).reshape(1, _TOKEN_COUNT, _MODEL_DIM)


def _dense_mlp(value, parameters, activation="silu"):
    first = parameters["Dense_0"]
    second = parameters["Dense_1"]
    hidden = _spu_batched_matmul(value, first["kernel"]) + first["bias"]
    if activation == "silu":
        activated = hidden / (1.0 + jnp.exp(-hidden))
    elif activation == "gelu":
        activated = jax.nn.gelu(hidden)
    else:
        raise ValueError(f"unsupported activation {activation!r}")
    return _spu_batched_matmul(activated, second["kernel"]) + second["bias"]


def vmoe_forward(
    image,
    params,
    moe_fn=None,
    routing_mode="token_choice",
    activation="silu",
    dense_activation=None,
    moe_activation=None,
    expert_biases=None,
    stage_marker=None,
):
    """Runs the complete graph with a configurable research routing mode."""
    dense_activation = activation if dense_activation is None else dense_activation
    moe_activation = activation if moe_activation is None else moe_activation
    expert_biases = moe_activation == "gelu" if expert_biases is None else expert_biases
    if moe_fn is None:
        if moe_activation == "gelu":
            try:
                moe_fn = _PLAINTEXT_GELU_MOE_BY_ROUTING_MODE[routing_mode]
            except KeyError as error:
                raise ValueError(
                    f"unsupported routing_mode {routing_mode!r}"
                ) from error
        elif expert_biases:
            try:
                moe_fn = _PLAINTEXT_SILU_BIAS_MOE_BY_ROUTING_MODE[routing_mode]
            except KeyError as error:
                raise ValueError(
                    f"unsupported routing_mode {routing_mode!r}"
                ) from error
        else:
            moe_fn = _moe_for_routing_mode(routing_mode)

    def mark(value, stage):
        return stage_marker(value, stage=stage) if stage_marker else value

    image = mark(image, "patch_embedding")
    embedding = params["embedding"]
    patches = jax.lax.conv_general_dilated(
        jnp.asarray(image),
        embedding["kernel"],
        window_strides=(32, 32),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    patches = patches.reshape(1, 49, _MODEL_DIM) + embedding["bias"]
    patches = mark(patches, "token_position_embedding")
    hidden = jnp.concatenate((params["cls"], patches), axis=1)
    encoder = params["Encoder"]
    hidden += encoder["posembed_input"]["pos_embedding"]

    for block_index in range(8):
        block = encoder[f"encoderblock_{block_index}"]
        hidden = mark(hidden, f"block_{block_index}.attention_norm")
        attention_input = _layer_norm(hidden, block["LayerNorm_0"])
        attention_input = mark(attention_input, f"block_{block_index}.self_attention")
        attention_output = _attention(attention_input, block["SelfAttention"])
        attention_output = mark(
            attention_output, f"block_{block_index}.attention_residual"
        )
        hidden = hidden + attention_output
        hidden = mark(hidden, f"block_{block_index}.mlp_norm")
        mlp_input = _layer_norm(hidden, block["LayerNorm_1"])
        if block_index in (5, 7):
            mlp_input = mark(mlp_input, f"block_{block_index}.moe")
            moe = block["Moe"]
            if expert_biases:
                mlp_output = moe_fn(
                    mlp_input,
                    moe["Router"]["dense"]["kernel"],
                    moe["Mlp"]["Dense_0"]["kernel"],
                    moe["Mlp"]["Dense_0"]["bias"],
                    moe["Mlp"]["Dense_1"]["kernel"],
                    moe["Mlp"]["Dense_1"]["bias"],
                )
            else:
                mlp_output = moe_fn(
                    mlp_input,
                    moe["Router"]["dense"]["kernel"],
                    moe["Mlp"]["Dense_0"]["kernel"],
                    moe["Mlp"]["Dense_1"]["kernel"],
                )
        else:
            mlp_input = mark(mlp_input, f"block_{block_index}.dense_ffn")
            mlp_output = _dense_mlp(mlp_input, block["Mlp"], dense_activation)
        mlp_output = mark(mlp_output, f"block_{block_index}.mlp_residual")
        hidden = hidden + mlp_output

    hidden = mark(hidden, "encoder_norm")
    hidden = _layer_norm(hidden, encoder["encoder_norm"])
    hidden = mark(hidden, "cls_pool")
    pooled = hidden[:, 0]
    pooled = mark(pooled, "pre_logits")
    pre_logits = params["pre_logits"]
    pooled = jnp.tanh(
        _spu_batched_matmul(pooled, pre_logits["kernel"]).reshape(1, _MODEL_DIM)
        + pre_logits["bias"]
    )
    pooled = mark(pooled, "classifier_head")
    head = params["head"]
    logits = _spu_batched_matmul(pooled, head["kernel"]).reshape(1, 1000) + head["bias"]
    return mark(logits, "end")


def vmoe_secure_forward(
    image,
    params,
    routing_mode="token_choice",
    activation="silu",
    dense_activation=None,
    moe_activation=None,
    expert_biases=None,
    trace_stages=False,
):
    """Lowers the graph with the selected Capricorn MoE custom call."""
    _moe_for_routing_mode(routing_mode)  # Validate the public static setting.
    dense_activation = activation if dense_activation is None else dense_activation
    moe_activation = activation if moe_activation is None else moe_activation
    expert_biases = moe_activation == "gelu" if expert_biases is None else expert_biases
    moe_fn = functools.partial(
        vmoe_block, routing_mode=routing_mode, activation=moe_activation
    )
    return vmoe_forward(
        image,
        params,
        moe_fn=moe_fn,
        routing_mode=routing_mode,
        activation=activation,
        dense_activation=dense_activation,
        moe_activation=moe_activation,
        expert_biases=expert_biases,
        stage_marker=vmoe_trace_marker if trace_stages else None,
    )
