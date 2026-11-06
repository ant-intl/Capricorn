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

"""Official V-MoE S/32 inference assets and checkpoint restoration."""

import dataclasses
import hashlib
import math
import pathlib
from typing import Any

import numpy as np
from PIL import Image


OFFICIAL_CHECKPOINT = "vmoe_s32_last2_ilsvrc2012_randaug_light1"
OFFICIAL_CHECKPOINT_SHARD_BYTES = 228_004_029
OFFICIAL_CHECKPOINT_SHARD_MD5 = "7b24c1cd3c6f7033727138eeadbdf259"
OFFICIAL_EVAL_CAPACITY_FACTOR = 4.0
OFFICIAL_EVAL_CAPACITY = 52


@dataclasses.dataclass(frozen=True)
class _Slice:
  start: int | None
  stop: int | None
  step: int | None


@dataclasses.dataclass(frozen=True)
class _IndexInfo:
  shape: tuple[int, ...]
  dtype: np.dtype
  global_slices: tuple[tuple[_Slice, ...], ...]
  shards: tuple[int, ...]


@dataclasses.dataclass
class _ArrayChunks:
  chunks: dict[int, list[tuple[np.ndarray, tuple[_Slice, ...]]]]


def _area_weights(input_size: int, output_size: int) -> np.ndarray:
  """Builds TensorFlow resize_area-compatible downsampling weights."""
  if output_size > input_size:
    raise ValueError(
        "official resize_area compatibility currently requires downsampling; "
        f"got {input_size} -> {output_size}"
    )
  scale = input_size / output_size
  weights = np.zeros((output_size, input_size), dtype=np.float32)
  for output_index in range(output_size):
    left = output_index * scale
    right = (output_index + 1) * scale
    first = int(math.floor(left))
    last = int(math.ceil(right))
    for input_index in range(first, min(last, input_size)):
      overlap = max(0.0, min(right, input_index + 1) - max(left, input_index))
      weights[output_index, input_index] = overlap / scale
  return weights


def preprocess_eval_image(path: pathlib.Path) -> tuple[np.ndarray, dict[str, Any]]:
  """Applies official decode|resize_small(256)|central_crop(224)|[-1,1]."""
  path = pathlib.Path(path)
  encoded = path.read_bytes()
  with Image.open(path) as image:
    image = image.convert("RGB")
    source = np.asarray(image, dtype=np.float32)
    source_height, source_width = source.shape[:2]
  ratio = 256.0 / min(source_height, source_width)
  resized_height = int(np.round(source_height * ratio))
  resized_width = int(np.round(source_width * ratio))
  height_weights = _area_weights(source_height, resized_height)
  width_weights = _area_weights(source_width, resized_width)
  resized_height_first = np.tensordot(height_weights, source, axes=(1, 0))
  resized = np.tensordot(
      resized_height_first, width_weights, axes=(1, 1)
  ).transpose(0, 2, 1)
  top = (resized_height - 224) // 2
  left = (resized_width - 224) // 2
  cropped = resized[top:top + 224, left:left + 224]
  normalized = (cropped / np.float32(255.0)) * np.float32(2.0) - np.float32(1.0)
  metadata = {
      "path": str(path.resolve()),
      "sha256": hashlib.sha256(encoded).hexdigest(),
      "decoded_size": [source_width, source_height],
      "resized_size": [resized_width, resized_height],
      "crop_size": [224, 224],
      "resize_method": "area",
      "value_range": [-1.0, 1.0],
  }
  return normalized[np.newaxis].astype(np.float32), metadata


def _decode_checkpoint_file(path: pathlib.Path):
  """Decodes the legacy V-MoE msgpack format without TensorFlow."""
  import msgpack

  def decode_ndarray(data):
    shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
    dtype = np.dtype(dtype_name.decode())
    return np.frombuffer(buffer, dtype=dtype).reshape(shape, order="C")

  def decode_slice(data):
    start, stop, step = msgpack.unpackb(data, raw=True)
    return _Slice(start, stop, step)

  def decode_slice_nd(data):
    encoded_slices = msgpack.unpackb(data, raw=True)
    return tuple(decode_slice(value) for value in encoded_slices)

  def ext_hook(code, data):
    if code == 1:
      return decode_ndarray(data)
    if code == 3:
      return decode_ndarray(data)[()]
    if code == 4:
      shape, dtype_name, _, _ = msgpack.unpackb(data, raw=True)
      return tuple(shape), np.dtype(dtype_name.decode())
    if code == 5:
      return decode_slice(data)
    if code == 6:
      return decode_slice_nd(data)
    if code == 8:
      encoded_shape, encoded_slices, shards = msgpack.unpackb(data, raw=True)
      shape, dtype = ext_hook(4, encoded_shape)
      return _IndexInfo(
          shape=shape,
          dtype=dtype,
          global_slices=tuple(decode_slice_nd(value) for value in encoded_slices),
          shards=tuple(shards),
      )
    return msgpack.ExtType(code, data)

  magic = "__msgpack_array_chunk_a87ca2__"

  def object_hook(mapping):
    if magic not in mapping:
      return mapping
    chunks = {}
    for array_index, encoded_chunks in mapping.items():
      if array_index == magic:
        continue
      restored = []
      for subchunks, global_slice in encoded_chunks:
        flat = np.concatenate(subchunks)
        shape = tuple(value.stop - value.start for value in global_slice)
        restored.append((flat.reshape(shape), global_slice))
      chunks[int(array_index)] = restored
    return _ArrayChunks(chunks)

  return msgpack.unpackb(
      path.read_bytes(), ext_hook=ext_hook, object_hook=object_hook, raw=False
  )


def _restore_array(info: _IndexInfo, chunks, array_index: int) -> np.ndarray:
  output = np.empty(info.shape, dtype=info.dtype)
  restored_chunks = chunks.chunks.get(array_index, [])
  if not restored_chunks:
    raise ValueError(f"checkpoint data has no chunks for array {array_index}")
  for chunk, global_slice in restored_chunks:
    slices = tuple(slice(value.start, value.stop, value.step) for value in global_slice)
    output[slices] = chunk
  return output


def load_official_checkpoint(prefix: pathlib.Path):
  """Restores the official single-shard checkpoint into a NumPy parameter tree."""
  import jax

  prefix = pathlib.Path(prefix)
  index_path = pathlib.Path(str(prefix) + ".index")
  data_path = pathlib.Path(str(prefix) + ".data-00000-of-00001")
  if not index_path.is_file() or not data_path.is_file():
    raise FileNotFoundError(
        f"expected checkpoint files {index_path.name} and {data_path.name}"
    )
  shard_bytes = data_path.stat().st_size
  shard_md5 = hashlib.md5(data_path.read_bytes()).hexdigest()  # nosec B324
  if shard_bytes != OFFICIAL_CHECKPOINT_SHARD_BYTES:
    raise ValueError(
        f"checkpoint shard has {shard_bytes} bytes; expected "
        f"{OFFICIAL_CHECKPOINT_SHARD_BYTES}"
    )
  if shard_md5 != OFFICIAL_CHECKPOINT_SHARD_MD5:
    raise ValueError(
        f"checkpoint shard MD5 is {shard_md5}; expected "
        f"{OFFICIAL_CHECKPOINT_SHARD_MD5}"
    )
  index_state = _decode_checkpoint_file(index_path)
  if index_state.get("shard_count") != 1:
    raise ValueError(f"expected one checkpoint shard, got {index_state.get('shard_count')}")
  index_tree = index_state["index"]
  index_leaves, tree = jax.tree_util.tree_flatten(index_tree)
  data_chunks = _decode_checkpoint_file(data_path)
  arrays = [
      _restore_array(info, data_chunks, index)
      for index, info in enumerate(index_leaves)
  ]
  params = jax.tree_util.tree_unflatten(tree, arrays)
  report = {
      "checkpoint": OFFICIAL_CHECKPOINT,
      "index_path": str(index_path.resolve()),
      "data_path": str(data_path.resolve()),
      "shard_count": 1,
      "shard_bytes": shard_bytes,
      "shard_md5": shard_md5,
      "restored_arrays": len(arrays),
      "version": index_state.get("version"),
  }
  return params, report


def _expected_parameter_shapes():
  expected = {
      "embedding/bias": (512,),
      "embedding/kernel": (32, 32, 3, 512),
      "cls": (1, 1, 512),
      "Encoder/posembed_input/pos_embedding": (1, 50, 512),
      "Encoder/encoder_norm/bias": (512,),
      "Encoder/encoder_norm/scale": (512,),
      "pre_logits/bias": (512,),
      "pre_logits/kernel": (512, 512),
      "head/bias": (1000,),
      "head/kernel": (512, 1000),
  }
  for block_index in range(8):
    base = f"Encoder/encoderblock_{block_index}"
    for layer_norm in ("LayerNorm_0", "LayerNorm_1"):
      expected[f"{base}/{layer_norm}/bias"] = (512,)
      expected[f"{base}/{layer_norm}/scale"] = (512,)
    for projection in ("key", "query", "value"):
      expected[f"{base}/SelfAttention/{projection}/bias"] = (8, 64)
      expected[f"{base}/SelfAttention/{projection}/kernel"] = (512, 8, 64)
    expected[f"{base}/SelfAttention/out/bias"] = (512,)
    expected[f"{base}/SelfAttention/out/kernel"] = (8, 64, 512)
    mlp = "Moe/Mlp" if block_index in (5, 7) else "Mlp"
    prefix = f"{base}/{mlp}"
    expert_prefix = (8,) if block_index in (5, 7) else ()
    expected[f"{prefix}/Dense_0/bias"] = expert_prefix + (2048,)
    expected[f"{prefix}/Dense_0/kernel"] = expert_prefix + (512, 2048)
    expected[f"{prefix}/Dense_1/bias"] = expert_prefix + (512,)
    expected[f"{prefix}/Dense_1/kernel"] = expert_prefix + (2048, 512)
    if block_index in (5, 7):
      expected[f"{base}/Moe/Router/dense/kernel"] = (512, 8)
  return expected


def _flatten_mapping(tree, prefix=""):
  output = {}
  if isinstance(tree, dict):
    for name, value in tree.items():
      child = f"{prefix}/{name}" if prefix else name
      output.update(_flatten_mapping(value, child))
  else:
    output[prefix] = np.asarray(tree)
  return output


def validate_official_parameters(params):
  """Returns complete key, shape and dtype coverage for the official model."""
  expected = _expected_parameter_shapes()
  actual = _flatten_mapping(params)
  missing = sorted(set(expected) - set(actual))
  unexpected = sorted(set(actual) - set(expected))
  shape_mismatches = sorted(
      f"{name}: expected {expected[name]}, got {actual[name].shape}"
      for name in set(expected) & set(actual)
      if tuple(actual[name].shape) != expected[name]
  )
  dtype_mismatches = sorted(
      f"{name}: expected float32, got {actual[name].dtype}"
      for name in set(expected) & set(actual)
      if actual[name].dtype != np.float32
  )
  valid = len(expected) - len(missing) - len(shape_mismatches) - len(dtype_mismatches)
  return {
      "expected_arrays": len(expected),
      "actual_arrays": len(actual),
      "coverage": valid / len(expected),
      "missing": missing,
      "unexpected": unexpected,
      "shape_mismatches": shape_mismatches,
      "dtype_mismatches": dtype_mismatches,
  }
