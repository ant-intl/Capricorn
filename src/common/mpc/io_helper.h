// Copyright 2026 Ant International, Ant Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "src/common/runtime/link/link_io.h"

namespace capricorn::operator_ {

namespace detail {

inline uint64_t ParallelBitsExtract(uint64_t value, uint64_t mask) {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
  return _pext_u64(value, mask);
#else
  uint64_t result = 0;
  for (uint64_t result_bit = 1; mask != 0; result_bit <<= 1) {
    const uint64_t lowest_mask_bit = mask & -mask;
    if ((value & lowest_mask_bit) != 0) {
      result |= result_bit;
    }
    mask &= mask - 1;
  }
  return result;
#endif
}

inline uint64_t ParallelBitsDeposit(uint64_t value, uint64_t mask) {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
  return _pdep_u64(value, mask);
#else
  uint64_t result = 0;
  for (uint64_t value_bit = 1; mask != 0; value_bit <<= 1) {
    const uint64_t lowest_mask_bit = mask & -mask;
    if ((value & value_bit) != 0) {
      result |= lowest_mask_bit;
    }
    mask &= mask - 1;
  }
  return result;
#endif
}

}  // namespace detail

inline void compress_uint8(const uint8_t* __restrict bits_ptr,
                           uint8_t* __restrict data_ptr, uint32_t round_num,
                           uint32_t bw) {
  assert(1 <= bw && bw <= 7);
  uint64_t mask = 0;
  memset(&mask, (1 << bw) - 1, sizeof(uint64_t));
  uint64_t* bits_u64_ptr = (uint64_t*)bits_ptr;
  for (uint32_t i = 0; i < round_num; i++) {
    uint64_t tmp = detail::ParallelBitsExtract(bits_u64_ptr[i], mask);
    memcpy(data_ptr, &tmp, bw);
    data_ptr += bw;
  }
}

inline void decompress_uint8(const uint8_t* __restrict data_ptr,
                             uint8_t* __restrict bits_ptr, uint32_t round_num,
                             uint32_t bw) {
  assert(1 <= bw && bw <= 7);
  uint64_t mask = 0;
  memset(&mask, (1 << bw) - 1, sizeof(uint64_t));
  uint64_t* bits_u64_ptr = (uint64_t*)bits_ptr;
  const uint8_t* cur_data_ptr = data_ptr;
  for (uint32_t i = 0; i < round_num; i++) {
    uint64_t tmp = 0;
    memcpy(&tmp, cur_data_ptr, bw);
    bits_u64_ptr[i] = detail::ParallelBitsDeposit(tmp, mask);
    cur_data_ptr += bw;
  }
}

/// @brief Compress bits into uint8_t array (8 bits per byte)
inline void CompressBitsIntoUint8(const bool* bits_ptr, uint8_t* data_ptr,
                                  uint32_t round_num) {
  const uint64_t mask = 0x0101010101010101ULL;
  const uint64_t* bits_u64_ptr = reinterpret_cast<const uint64_t*>(bits_ptr);
  for (uint32_t i = 0; i < round_num; i++) {
    data_ptr[i] = detail::ParallelBitsExtract(bits_u64_ptr[i], mask);
  }
}

/// @brief Decompress bits from uint8_t array
inline void DecompressBitsFromUint8(const uint8_t* data_ptr, bool* bits_ptr,
                                    uint32_t round_num) {
  const uint64_t mask = 0x0101010101010101ULL;
  uint64_t* bits_u64_ptr = reinterpret_cast<uint64_t*>(bits_ptr);
  for (uint32_t i = 0; i < round_num; i++) {
    bits_u64_ptr[i] =
        detail::ParallelBitsDeposit(static_cast<uint64_t>(data_ptr[i]), mask);
  }
}

/// @brief Send bool vector with compression (using raw pointers)
inline void SendBoolVector(const bool* x, uint32_t len, link::LinkIO* io) {
  const uint32_t a = len / 8;
  const uint32_t b = len % 8;
  if (a != 0) {
    std::vector<uint8_t> send_data(a);
    CompressBitsIntoUint8(x, send_data.data(), a);
    io->send_data(send_data.data(), static_cast<int>(a));
  }
  if (b != 0) {
    io->send_data(x + a * 8, static_cast<int>(b));
  }
}

/// @brief Receive bool vector with decompression (using raw pointers)
inline void RecvBoolVector(bool* x, uint32_t len, link::LinkIO* io) {
  const uint32_t a = len / 8;
  const uint32_t b = len % 8;
  if (a != 0) {
    std::vector<uint8_t> recv_data(a);
    io->recv_data(recv_data.data(), static_cast<int>(a));
    DecompressBitsFromUint8(recv_data.data(), x, a);
  }
  if (b != 0) {
    io->recv_data(x + a * 8, static_cast<int>(b));
  }
}

/// @brief Send uint64_t vector with bit width compression
inline void SendU64Vector(const uint64_t* x, uint32_t len, uint32_t bw,
                          link::LinkIO* io) {
  uint32_t num_in_bytes = static_cast<uint32_t>(std::ceil(bw / 8.0));
  std::vector<uint8_t> send_data(len * num_in_bytes);
  uint8_t* cur_ptr = send_data.data();
  for (uint32_t i = 0; i < len; i++) {
    std::memcpy(cur_ptr, x + i, num_in_bytes);
    cur_ptr += num_in_bytes;
  }
  io->send_data(send_data.data(), static_cast<int>(len * num_in_bytes));
}

/// @brief Receive uint64_t vector with bit width decompression
inline void RecvU64Vector(uint64_t* x, uint32_t len, uint32_t bw,
                          link::LinkIO* io) {
  uint32_t num_in_bytes = static_cast<uint32_t>(std::ceil(bw / 8.0));
  std::vector<uint8_t> recv_data(len * num_in_bytes);
  io->recv_data(recv_data.data(), static_cast<int>(len * num_in_bytes));
  uint8_t* cur_ptr = recv_data.data();
  for (uint32_t i = 0; i < len; i++) {
    uint64_t tmp = 0;
    std::memcpy(&tmp, cur_ptr, num_in_bytes);
    x[i] = tmp;
    cur_ptr += num_in_bytes;
  }
}

/// @brief Send uint8_t vector with bit width compression
inline void SendUint8Vector(const uint8_t* x, uint32_t len, uint32_t bw,
                            link::LinkIO* io) {
  if (bw == 8) {
    io->send_data(x, static_cast<int>(len));
    return;
  }
  // For bw < 8, use simple packing
  uint32_t round_num = len / 8;
  uint32_t extra_len = len % 8;
  std::vector<uint8_t> send_data(round_num * bw + extra_len);

  compress_uint8(x, send_data.data(), round_num, bw);
  if (extra_len != 0) {
    std::memcpy(send_data.data() + round_num * bw, x + round_num * 8,
                extra_len);
  }
  io->send_data(send_data.data(), static_cast<int>(send_data.size()));
}

/// @brief Receive uint8_t vector with bit width decompression
inline void RecvUint8Vector(uint8_t* x, uint32_t len, uint32_t bw,
                            link::LinkIO* io) {
  if (bw == 8) {
    io->recv_data(x, static_cast<int>(len));
    return;
  }
  uint32_t round_num = len / 8;
  uint32_t extra_len = len % 8;
  uint32_t total_size = round_num * bw + extra_len;
  std::vector<uint8_t> recv_data(total_size);
  io->recv_data(recv_data.data(), static_cast<int>(recv_data.size()));

  decompress_uint8(recv_data.data(), x, round_num, bw);
  if (extra_len != 0) {
    std::memcpy(x + round_num * 8, recv_data.data() + round_num * bw,
                extra_len);
  }
}

}  // namespace capricorn::operator_
