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

#include <cstdint>
#include <memory>

#include "src/common/mpc/rot_context.h"
#include "src/common/runtime/link/link_io.h"

// Forward declarations for SEAL types to avoid heavy header inclusion
namespace seal {
class EncryptionParameters;
class SEALContext;
class PublicKey;
class Evaluator;
class BatchEncoder;
class Decryptor;
class Encryptor;
class Ciphertext;
}  // namespace seal

namespace capricorn::operator_ {

/**
 * BOLE (Batched Oblivious Linear Evaluation) context
 */
class BoleContext {
 public:
  BoleContext(uint32_t party_id, uint32_t thread_num, RotContext* rot_context,
              link::LinkIO* io);
  ~BoleContext() = default;

  // Non-copyable
  BoleContext(const BoleContext&) = delete;
  BoleContext& operator=(const BoleContext&) = delete;

  // Movable
  BoleContext(BoleContext&&) noexcept;
  BoleContext& operator=(BoleContext&&) noexcept;

  void BOLE(uint32_t n, uint32_t bw, const uint64_t* x, const uint64_t* y,
            uint64_t* z);

  void BOLESquare(uint32_t n, uint32_t bw, const uint64_t* x, uint64_t* z);
  void BOLESelf(uint32_t n, uint32_t bw, const uint64_t* x, uint64_t* z);

 private:
  // SEAL parameters
  static constexpr uint32_t kPolynomialDegree = 4096;
  static constexpr uint64_t kP0 = 2061584302081ULL;
  static constexpr uint64_t kP1 = 2748779069441ULL;
  static constexpr uint64_t kP0Inv = 4ULL;
  static constexpr uint64_t kP1Inv = 2061584302078ULL;

  uint32_t party_id_;
  uint32_t thread_num_;
  link::LinkIO* io_;
  RotContext* rot_context_;

  // SEAL context and components
  std::shared_ptr<seal::EncryptionParameters> parms0_;
  std::shared_ptr<seal::EncryptionParameters> parms1_;
  std::shared_ptr<seal::SEALContext> context0_;
  std::shared_ptr<seal::SEALContext> context1_;
  std::shared_ptr<seal::PublicKey> public_key0_;
  std::shared_ptr<seal::PublicKey> public_key1_;
  std::shared_ptr<seal::Evaluator> evaluator0_;
  std::shared_ptr<seal::Evaluator> evaluator1_;
  std::shared_ptr<seal::BatchEncoder> encoder0_;
  std::shared_ptr<seal::BatchEncoder> encoder1_;
  std::shared_ptr<seal::Decryptor> decryptor0_;
  std::shared_ptr<seal::Decryptor> decryptor1_;
  std::shared_ptr<seal::Encryptor> encryptor0_;
  std::shared_ptr<seal::Encryptor> encryptor1_;

  // Helper methods
  void SetupSealContext();
  void SendTwoCiphertextVector(
      uint32_t n, const std::vector<seal::Ciphertext>& cipher_vct0,
      const std::vector<seal::Ciphertext>& cipher_vct1);
  void RecvTwoCiphertextVector(uint32_t num,
                               std::vector<seal::Ciphertext>& cipher_vct0,
                               std::vector<seal::Ciphertext>& cipher_vct1);
  uint64_t MulMod(uint64_t x, uint64_t y, uint64_t mod);
  uint64_t CRT(uint64_t a0, uint64_t a1);
};

}  // namespace capricorn::operator_