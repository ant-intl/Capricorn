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

#include <emp-tool/emp-tool.h>

#include <cstdint>
#include <cstring>

namespace capricorn::operator_ {

inline void CcrFunctionHInplace(emp::block& x, const emp::AES_KEY _aes) {
  static const emp::block MASK = emp::makeBlock(0xFFFFFFFFFFFFFFFF, 0x00);
  emp::block tmp_block;
  x = tmp_block =
      _mm_shuffle_epi32(x, 78) ^ (x & MASK);  // x = \sigma(x), tmp = \sigma(x)
  emp::AES_ecb_encrypt_blks(&tmp_block, 1,
                            &_aes);  // x = \sigma(x), tmp = AES(\sigma(x))
  x ^= tmp_block;                    // x = \sigma(x) xor AES(\sigma(x))
}

// ============================================================================
// XOR Operations
// ============================================================================

/// @brief Elementwise XOR: z = x ^ y
template <typename T>
inline void ElementwiseXor(uint32_t len, const T* __restrict x,
                           const T* __restrict y, T* __restrict z) {
  for (uint32_t i = 0; i < len; i++) {
    z[i] = x[i] ^ y[i];
  }
}

/// @brief Elementwise XOR with scalar: z = x ^ y
template <typename T>
inline void ElementwiseXor(uint32_t len, const T* __restrict x, T y,
                           T* __restrict z) {
  for (uint32_t i = 0; i < len; i++) {
    z[i] = x[i] ^ y;
  }
}

/// @brief Inplace elementwise XOR with scalar: x ^= y
template <typename T>
inline void ElementwiseXorInplace(uint32_t len, T* __restrict x, T y) {
  for (uint32_t i = 0; i < len; i++) {
    x[i] ^= y;
  }
}

/// @brief Inplace elementwise XOR: x ^= y
template <typename T>
inline void ElementwiseXorInplace(uint32_t len, T* __restrict x,
                                  const T* __restrict y) {
  for (uint32_t i = 0; i < len; i++) {
    x[i] ^= y[i];
  }
}

// ============================================================================
// AND Operations
// ============================================================================

/// @brief Elementwise AND with mask: c = a & mod_mask
template <typename T>
inline void ElementwiseAnd(uint32_t len, const T* __restrict a, T mod_mask,
                           T* __restrict c) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = a[i] & mod_mask;
  }
}

/// @brief Elementwise AND: c = a & b
template <typename T>
inline void ElementwiseAnd(uint32_t len, const T* __restrict a,
                           const T* __restrict b, T* __restrict c) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = a[i] & b[i];
  }
}

/// @brief Inplace elementwise AND with mask: x &= mod_mask
template <typename T>
inline void ElementwiseAndInplace(uint32_t len, T* __restrict x, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    x[i] &= mod_mask;
  }
}

/// @brief Inplace elementwise AND: x &= y
template <typename T>
inline void ElementwiseAndInplace(uint32_t len, T* __restrict x,
                                  const T* __restrict y) {
  for (uint32_t i = 0; i < len; i++) {
    x[i] &= y[i];
  }
}

// ============================================================================
// ADD Operations
// ============================================================================

/// @brief Elementwise add: c = a + b
template <typename T>
inline void ElementwiseAdd(uint32_t len, const T* __restrict a,
                           const T* __restrict b, T* __restrict c) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = a[i] + b[i];
  }
}

/// @brief Inplace elementwise add: a += b
template <typename T>
inline void ElementwiseAddInplace(uint32_t len, T* __restrict a,
                                  const T* __restrict b) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] += b[i];
  }
}

/// @brief Elementwise modular add with scalar: c = (a + b) & mod_mask
template <typename T>
inline void ElementwiseAddmod(uint32_t len, const T* __restrict a, T b,
                              T* __restrict c, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = (a[i] + b) & mod_mask;
  }
}

/// @brief Elementwise modular add: c = (a + b) & mod_mask
template <typename T>
inline void ElementwiseAddmod(uint32_t len, const T* __restrict a,
                              const T* __restrict b, T* __restrict c,
                              T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = (a[i] + b[i]) & mod_mask;
  }
}

/// @brief Inplace elementwise modular add with scalar: a = (a + b) & mod_mask
template <typename T>
inline void ElementwiseAddmodInplace(uint32_t len, T* __restrict a, T b,
                                     T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] + b) & mod_mask;
  }
}

/// @brief Inplace elementwise modular add: a = (a + b) & mod_mask
template <typename T>
inline void ElementwiseAddmodInplace(uint32_t len, T* __restrict a,
                                     const T* __restrict b, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] + b[i]) & mod_mask;
  }
}

// ============================================================================
// SUB Operations
// ============================================================================

/// @brief Elementwise sub: c = a - b
template <typename T>
inline void ElementwiseSub(uint32_t len, const T* __restrict a,
                           const T* __restrict b, T* __restrict c) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = a[i] - b[i];
  }
}

/// @brief Inplace elementwise sub: a -= b
template <typename T>
inline void ElementwiseSubInplace(uint32_t len, T* __restrict a,
                                  const T* __restrict b) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = a[i] - b[i];
  }
}

/// @brief Inplace elementwise sub with scalar: a -= b
template <typename T>
inline void ElementwiseSubInplace(uint32_t len, T* __restrict a, T b) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = a[i] - b;
  }
}

/// @brief Elementwise modular sub with scalar: c = (a - b) & mod_mask
template <typename T>
inline void ElementwiseSubmod(uint32_t len, const T* __restrict a, T b,
                              T* __restrict c, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = (a[i] - b) & mod_mask;
  }
}

/// @brief Elementwise modular sub: c = (a - b) & mod_mask
template <typename T>
inline void ElementwiseSubmod(uint32_t len, const T* __restrict a,
                              const T* __restrict b, T* __restrict c,
                              T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = (a[i] - b[i]) & mod_mask;
  }
}

/// @brief Inplace elementwise modular sub with scalar: a = (a - b) & mod_mask
template <typename T>
inline void ElementwiseSubmodInplace(uint32_t len, T* __restrict a, T b,
                                     T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] - b) & mod_mask;
  }
}

/// @brief Inplace elementwise modular sub: a = (a - b) & mod_mask
template <typename T>
inline void ElementwiseSubmodInplace(uint32_t len, T* __restrict a,
                                     const T* __restrict b, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] - b[i]) & mod_mask;
  }
}

// ============================================================================
// MUL Operations
// ============================================================================

/// @brief Elementwise modular mul with scalar: c = (a * b) & mod_mask
template <typename T>
inline void ElementwiseMulmod(uint32_t len, const T* __restrict a, T b,
                              T* __restrict c, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = (a[i] * b) & mod_mask;
  }
}

/// @brief Elementwise modular mul: c = (a * b) & mod_mask
template <typename T>
inline void ElementwiseMulmod(uint32_t len, const T* __restrict a,
                              const T* __restrict b, T* __restrict c,
                              T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    c[i] = (a[i] * b[i]) & mod_mask;
  }
}

/// @brief Inplace elementwise modular mul with scalar: a = (a * b) & mod_mask
template <typename T>
inline void ElementwiseMulmodInplace(uint32_t len, T* __restrict a, T b,
                                     T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] * b) & mod_mask;
  }
}

/// @brief Inplace elementwise modular mul: a = (a * b) & mod_mask
template <typename T>
inline void ElementwiseMulmodInplace(uint32_t len, T* __restrict a,
                                     const T* __restrict b, T mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] * b[i]) & mod_mask;
  }
}

// ============================================================================
// COPY Operations
// ============================================================================

/// @brief Elementwise copy with type conversion: y = (T2)x
template <typename T1, typename T2>
inline void ElementwiseCopy(uint32_t len, const T1* __restrict x,
                            T2* __restrict y) {
  for (uint32_t i = 0; i < len; i++) {
    y[i] = static_cast<T2>(x[i]);
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Get signed value from unsigned representation
/// @param x Unsigned value
/// @param bw Bit width
/// @return Signed interpretation of x
inline int64_t GetSignedValue(uint64_t x, uint32_t bw) {
  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;
  uint64_t msb_mask = 1ULL << (bw - 1);
  x &= mod_mask;
  if (x & msb_mask) {
    return static_cast<int64_t>((x ^ msb_mask) - msb_mask);
  }
  return static_cast<int64_t>(x);
}

}  // namespace capricorn::operator_