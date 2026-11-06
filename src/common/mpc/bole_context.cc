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

#include "src/common/mpc/bole_context.h"

#include <seal/seal.h>
#include <seal/secretkey.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rlwe.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include "src/common/mpc/rot_context.h"

namespace capricorn::operator_ {

namespace {

// Helper functions adapted from Common.h
void ElementwiseCopyAndMul(uint32_t len, uint64_t* dst, const uint64_t* src_a,
                           const uint64_t* src_b, uint64_t mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    dst[i] = (src_a[i] * src_b[i]) & mod_mask;
  }
}

void ElementwiseSubModInplace(uint32_t len, uint64_t* a, const uint64_t* b,
                              uint64_t mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] - b[i]) & mod_mask;
  }
}

void ElementwiseMulModInplace(uint32_t len, uint64_t* a, uint64_t b,
                              uint64_t mod_mask) {
  for (uint32_t i = 0; i < len; i++) {
    a[i] = (a[i] * b) & mod_mask;
  }
}

// SEAL helper functions adapted from FHE_Common.h
void SendPublicKey(const seal::PublicKey& public_key, link::LinkIO& io) {
  std::ostringstream os;
  public_key.save(os);
  std::string data = os.str();
  uint32_t len = static_cast<uint32_t>(data.size());
  io.send_data(&len, sizeof(uint32_t));
  io.send_data(data.data(), len);
  io.flush();
}

void RecvPublicKey(const seal::SEALContext& context,
                   seal::PublicKey& public_key, link::LinkIO& io) {
  uint32_t len;
  io.recv_data(&len, sizeof(uint32_t));
  std::vector<char> buffer(len);
  io.recv_data(buffer.data(), len);
  std::istringstream is(std::string(buffer.data(), len));
  public_key.load(context, is);
}

void SendCiphertext(
    uint32_t cipher_num,
    const std::vector<seal::Serializable<seal::Ciphertext>>& cipher_vct,
    link::LinkIO& io) {
  for (uint32_t i = 0; i < cipher_num; i++) {
    std::ostringstream os;
    cipher_vct[i].save(os);
    std::string data = os.str();
    uint32_t len = static_cast<uint32_t>(data.size());
    io.send_data(&len, sizeof(uint32_t));
    io.send_data(data.data(), len);
  }
  io.flush();
}

void RecvCiphertext(const seal::SEALContext& context,
                    seal::Ciphertext& ciphertext, link::LinkIO& io) {
  uint32_t len;
  io.recv_data(&len, sizeof(uint32_t));
  std::vector<char> buffer(len);
  io.recv_data(buffer.data(), len);
  std::istringstream is(std::string(buffer.data(), len));
  ciphertext.load(context, is);
}

void RecvCiphertext(uint32_t cipher_num, const seal::SEALContext& context,
                    std::vector<seal::Ciphertext>& cipher_vct,
                    link::LinkIO& io) {
  cipher_vct.resize(cipher_num);
  for (uint32_t i = 0; i < cipher_num; i++) {
    RecvCiphertext(context, cipher_vct[i], io);
  }
}

}  // anonymous namespace

BoleContext::BoleContext(uint32_t party_id, uint32_t thread_num,
                         RotContext* rot_context, link::LinkIO* io)
    : party_id_(party_id),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)),
      io_(io),
      rot_context_(rot_context) {
  SetupSealContext();
}

BoleContext::BoleContext(BoleContext&& other) noexcept
    : party_id_(other.party_id_),
      thread_num_(other.thread_num_),
      io_(other.io_),
      parms0_(std::move(other.parms0_)),
      parms1_(std::move(other.parms1_)),
      context0_(std::move(other.context0_)),
      context1_(std::move(other.context1_)),
      public_key0_(std::move(other.public_key0_)),
      public_key1_(std::move(other.public_key1_)),
      evaluator0_(std::move(other.evaluator0_)),
      evaluator1_(std::move(other.evaluator1_)),
      encoder0_(std::move(other.encoder0_)),
      encoder1_(std::move(other.encoder1_)),
      decryptor0_(std::move(other.decryptor0_)),
      decryptor1_(std::move(other.decryptor1_)),
      encryptor0_(std::move(other.encryptor0_)),
      encryptor1_(std::move(other.encryptor1_)) {
  other.io_ = nullptr;
}

BoleContext& BoleContext::operator=(BoleContext&& other) noexcept {
  if (this != &other) {
    party_id_ = other.party_id_;
    thread_num_ = other.thread_num_;
    io_ = other.io_;
    parms0_ = std::move(other.parms0_);
    parms1_ = std::move(other.parms1_);
    context0_ = std::move(other.context0_);
    context1_ = std::move(other.context1_);
    public_key0_ = std::move(other.public_key0_);
    public_key1_ = std::move(other.public_key1_);
    evaluator0_ = std::move(other.evaluator0_);
    evaluator1_ = std::move(other.evaluator1_);
    encoder0_ = std::move(other.encoder0_);
    encoder1_ = std::move(other.encoder1_);
    decryptor0_ = std::move(other.decryptor0_);
    decryptor1_ = std::move(other.decryptor1_);
    encryptor0_ = std::move(other.encryptor0_);
    encryptor1_ = std::move(other.encryptor1_);
    other.io_ = nullptr;
  }
  return *this;
}

void BoleContext::SetupSealContext() {
  using namespace seal;

  // Setup encryption parameters for two CRT moduli
  parms0_ = std::make_shared<EncryptionParameters>(scheme_type::bfv);
  parms1_ = std::make_shared<EncryptionParameters>(scheme_type::bfv);

  parms0_->set_use_special_prime(false);
  parms1_->set_use_special_prime(false);

  parms0_->set_poly_modulus_degree(kPolynomialDegree);
  parms1_->set_poly_modulus_degree(kPolynomialDegree);

  std::vector<int> moduli_bits = {60, 49};
  parms0_->set_coeff_modulus(
      CoeffModulus::Create(kPolynomialDegree, moduli_bits));
  parms1_->set_coeff_modulus(
      CoeffModulus::Create(kPolynomialDegree, moduli_bits));

  parms0_->set_plain_modulus(kP0);
  parms1_->set_plain_modulus(kP1);

  context0_ =
      std::make_shared<SEALContext>(*parms0_, true, sec_level_type::tc128);
  context1_ =
      std::make_shared<SEALContext>(*parms1_, true, sec_level_type::tc128);

  [[maybe_unused]] auto qualifiers0 =
      context0_->first_context_data()->qualifiers();
  [[maybe_unused]] auto qualifiers1 =
      context1_->first_context_data()->qualifiers();
  assert(qualifiers0.using_batching);
  assert(qualifiers1.using_batching);

  evaluator0_ = std::make_shared<Evaluator>(*context0_);
  evaluator1_ = std::make_shared<Evaluator>(*context1_);

  encoder0_ = std::make_shared<BatchEncoder>(*context0_);
  encoder1_ = std::make_shared<BatchEncoder>(*context1_);

  // Generate keys
  KeyGenerator keygen0(*context0_), keygen1(*context1_);
  SecretKey secret_key0 = keygen0.secret_key();
  SecretKey secret_key1 = keygen1.secret_key();
  PublicKey cur_public_key0, cur_public_key1;
  keygen0.create_public_key(cur_public_key0);
  keygen1.create_public_key(cur_public_key1);

  encryptor0_ =
      std::make_shared<Encryptor>(*context0_, cur_public_key0, secret_key0);
  encryptor1_ =
      std::make_shared<Encryptor>(*context1_, cur_public_key1, secret_key1);

  decryptor0_ = std::make_shared<Decryptor>(*context0_, secret_key0);
  decryptor1_ = std::make_shared<Decryptor>(*context1_, secret_key1);

  // Exchange public keys
  PublicKey tmp_public_key0, tmp_public_key1;
  if (party_id_ == kServerId) {
    SendPublicKey(cur_public_key0, *io_);
    SendPublicKey(cur_public_key1, *io_);
    RecvPublicKey(*context0_, tmp_public_key0, *io_);
    RecvPublicKey(*context1_, tmp_public_key1, *io_);
  } else {
    RecvPublicKey(*context0_, tmp_public_key0, *io_);
    RecvPublicKey(*context1_, tmp_public_key1, *io_);
    SendPublicKey(cur_public_key0, *io_);
    SendPublicKey(cur_public_key1, *io_);
  }
  public_key0_ = std::make_shared<PublicKey>(tmp_public_key0);
  public_key1_ = std::make_shared<PublicKey>(tmp_public_key1);
}

uint64_t BoleContext::MulMod(uint64_t x, uint64_t y, uint64_t mod) {
  __uint128_t res = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(y);
  res = res % mod;
  return static_cast<uint64_t>(res);
}

uint64_t BoleContext::CRT(uint64_t a0, uint64_t a1) {
  const static __uint128_t m0 =
      static_cast<__uint128_t>(kP1Inv) * static_cast<__uint128_t>(kP1);
  const static __uint128_t m1 =
      static_cast<__uint128_t>(kP0Inv) * static_cast<__uint128_t>(kP0);
  const static __uint128_t M =
      static_cast<__uint128_t>(kP0) * static_cast<__uint128_t>(kP1);
  __uint128_t res = (a0 * m0 + a1 * m1) % M;
  return static_cast<uint64_t>(res);
}

void BoleContext::SendTwoCiphertextVector(
    uint32_t n, const std::vector<seal::Ciphertext>& cipher_vct0,
    const std::vector<seal::Ciphertext>& cipher_vct1) {
  const uint32_t cipher_size = 2 * kPolynomialDegree * sizeof(uint64_t) + 128;
  const uint32_t len_size = sizeof(uint32_t);

  std::vector<std::ostringstream> os_vct(n * 2);
  std::vector<uint32_t> len_vct(n * 2);

  for (uint32_t i = 0; i < n; i++) {
    cipher_vct0[i].save(os_vct[i]);
    len_vct[i] = static_cast<uint32_t>(os_vct[i].str().length());

    cipher_vct1[i].save(os_vct[i + n]);
    len_vct[i + n] = static_cast<uint32_t>(os_vct[i + n].str().length());
  }

  (void)std::accumulate(len_vct.begin(), len_vct.end(),
                        0u);  // Suppress unused warning
  std::vector<char> buffer(2 * n * (cipher_size + len_size));

  std::memcpy(buffer.data(), len_vct.data(), 2 * n * len_size);
  uint32_t cur = 2 * n * len_size;
  for (uint32_t i = 0; i < 2 * n; i++) {
    std::memcpy(buffer.data() + cur, os_vct[i].str().c_str(), len_vct[i]);
    cur += len_vct[i];
  }

  io_->send_data(buffer.data(), 2 * n * (cipher_size + len_size));
  io_->flush();
}

void BoleContext::RecvTwoCiphertextVector(
    uint32_t num, std::vector<seal::Ciphertext>& cipher_vct0,
    std::vector<seal::Ciphertext>& cipher_vct1) {
  const uint32_t cipher_size = 2 * kPolynomialDegree * sizeof(uint64_t) + 128;
  const uint32_t len_size = sizeof(uint32_t);

  cipher_vct0.resize(num);
  cipher_vct1.resize(num);
  std::vector<std::istringstream> is_vct(num * 2);
  std::vector<uint32_t> len_vct(num * 2);

  std::vector<char> buffer(2 * num * (cipher_size + len_size));
  io_->recv_data(buffer.data(), 2 * num * (cipher_size + len_size));

  std::memcpy(len_vct.data(), buffer.data(), 2 * num * len_size);

  uint32_t cur = 2 * num * len_size;
  for (uint32_t i = 0; i < 2 * num; i++) {
    is_vct[i].str(std::string(buffer.data() + cur, len_vct[i]));
    cur += len_vct[i];
  }

  for (uint32_t i = 0; i < num; i++) {
    cipher_vct0[i].load(*context0_, is_vct[i]);
    cipher_vct1[i].load(*context1_, is_vct[i + num]);
  }
}

void BoleContext::BOLE(uint32_t n, uint32_t bw, const uint64_t* x,
                       const uint64_t* y, uint64_t* z) {
  const uint64_t MOD_MASK = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  const uint32_t cipher_num =
      static_cast<uint32_t>(std::ceil(1.0 * n / kPolynomialDegree));
  const uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * cipher_num / thread_num_));

  // z = (x0 * y0 + x1 * y1) + (x0 * y1 + x1 * y0)
  ElementwiseCopyAndMul(n, z, x, y, MOD_MASK);  // x0 * y0 + x1 * y1

  if (party_id_ == kClientId) {
    // Client side
    seal::Serializable<seal::Ciphertext> dummy0 = encryptor0_->encrypt_zero();
    seal::Serializable<seal::Ciphertext> dummy1 = encryptor1_->encrypt_zero();

    std::vector<seal::Serializable<seal::Ciphertext>> ciphertext_x0(cipher_num,
                                                                    dummy0);
    std::vector<seal::Serializable<seal::Ciphertext>> ciphertext_x1(cipher_num,
                                                                    dummy1);
    std::vector<seal::Serializable<seal::Ciphertext>> ciphertext_y0(cipher_num,
                                                                    dummy0);
    std::vector<seal::Serializable<seal::Ciphertext>> ciphertext_y1(cipher_num,
                                                                    dummy1);

    // Encode and encrypt
    auto encode_then_encrypt = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      std::vector<uint64_t> tmp_vct(kPolynomialDegree);
      seal::Plaintext tmp_plaintext(kPolynomialDegree);

      for (uint32_t i = start; i < end; i++) {
        std::memset(tmp_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_vct.data(), x + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_vct, tmp_plaintext);
        ciphertext_x0[i] = encryptor0_->encrypt_symmetric(tmp_plaintext);

        encoder1_->encode(tmp_vct, tmp_plaintext);
        ciphertext_x1[i] = encryptor1_->encrypt_symmetric(tmp_plaintext);

        std::memcpy(tmp_vct.data(), y + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_vct, tmp_plaintext);
        ciphertext_y0[i] = encryptor0_->encrypt_symmetric(tmp_plaintext);

        encoder1_->encode(tmp_vct, tmp_plaintext);
        ciphertext_y1[i] = encryptor1_->encrypt_symmetric(tmp_plaintext);
      }
    };

    auto* thread_pool = rot_context_->GetThreadPool();
    std::vector<std::future<void>> futures(thread_num_);
    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(encode_then_encrypt, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();

    SendCiphertext(cipher_num, ciphertext_x0, *io_);
    SendCiphertext(cipher_num, ciphertext_x1, *io_);
    SendCiphertext(cipher_num, ciphertext_y0, *io_);
    SendCiphertext(cipher_num, ciphertext_y1, *io_);

    std::vector<seal::Ciphertext> cipher_z0(cipher_num), cipher_z1(cipher_num);
    RecvTwoCiphertextVector(cipher_num, cipher_z0, cipher_z1);

    std::vector<seal::Plaintext> plain_z0(cipher_num), plain_z1(cipher_num);
    auto decrypt_then_decode = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      std::vector<uint64_t> z0(kPolynomialDegree), z1(kPolynomialDegree);
      for (uint32_t i = start; i < end; i++) {
        decryptor0_->decrypt(cipher_z0[i], plain_z0[i]);
        decryptor1_->decrypt(cipher_z1[i], plain_z1[i]);
        encoder0_->decode(plain_z0[i], z0);
        encoder1_->decode(plain_z1[i], z1);
        for (uint32_t j = 0;
             j < std::min(kPolynomialDegree, n - i * kPolynomialDegree); j++) {
          z[i * kPolynomialDegree + j] =
              (z[i * kPolynomialDegree + j] + CRT(z0[j], z1[j])) & MOD_MASK;
        }
      }
    };

    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(decrypt_then_decode, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();
  } else {
    // Server side
    std::vector<uint64_t> c(n);
    // Generate random mask
    for (uint32_t i = 0; i < n; i++) {
      c[i] = static_cast<uint64_t>(rand()) & MOD_MASK;
    }
    ElementwiseSubModInplace(n, z, c.data(), MOD_MASK);

    std::vector<seal::Plaintext> plain_x0(cipher_num), plain_x1(cipher_num);
    std::vector<seal::Plaintext> plain_y0(cipher_num), plain_y1(cipher_num);
    std::vector<seal::Plaintext> plain_c0(cipher_num), plain_c1(cipher_num);

    auto encode_prg = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      std::vector<uint64_t> tmp_x_vct(kPolynomialDegree);
      std::vector<uint64_t> tmp_y_vct(kPolynomialDegree);
      std::vector<uint64_t> tmp_c_vct(kPolynomialDegree);

      for (uint32_t i = start; i < end; i++) {
        std::memset(tmp_x_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_x_vct.data(), x + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_x_vct, plain_x0[i]);
        encoder1_->encode(tmp_x_vct, plain_x1[i]);

        std::memset(tmp_y_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_y_vct.data(), y + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_y_vct, plain_y0[i]);
        encoder1_->encode(tmp_y_vct, plain_y1[i]);

        std::memset(tmp_c_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_c_vct.data(), c.data() + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_c_vct, plain_c0[i]);
        encoder1_->encode(tmp_c_vct, plain_c1[i]);
      }
    };

    auto* thread_pool = rot_context_->GetThreadPool();
    std::vector<std::future<void>> futures(thread_num_);
    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(encode_prg, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();

    std::vector<seal::Ciphertext> cipher_x0(cipher_num), cipher_x1(cipher_num);
    std::vector<seal::Ciphertext> cipher_y0(cipher_num), cipher_y1(cipher_num);
    std::vector<seal::Ciphertext> cipher_z0(cipher_num), cipher_z1(cipher_num);

    RecvCiphertext(cipher_num, *context0_, cipher_x0, *io_);
    RecvCiphertext(cipher_num, *context1_, cipher_x1, *io_);
    RecvCiphertext(cipher_num, *context0_, cipher_y0, *io_);
    RecvCiphertext(cipher_num, *context1_, cipher_y1, *io_);

    auto multiply_then_mask = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      for (uint32_t i = start; i < end; i++) {
        evaluator0_->multiply_plain_inplace(cipher_x0[i], plain_y0[i]);
        evaluator1_->multiply_plain_inplace(cipher_x1[i], plain_y1[i]);

        evaluator0_->multiply_plain_inplace(cipher_y0[i], plain_x0[i]);
        evaluator1_->multiply_plain_inplace(cipher_y1[i], plain_x1[i]);

        evaluator0_->add(cipher_x0[i], cipher_y0[i], cipher_z0[i]);
        evaluator0_->add_plain_inplace(cipher_z0[i], plain_c0[i]);
        evaluator0_->mod_switch_to_inplace(cipher_z0[i],
                                           context0_->last_parms_id());

        evaluator1_->add(cipher_x1[i], cipher_y1[i], cipher_z1[i]);
        evaluator1_->add_plain_inplace(cipher_z1[i], plain_c1[i]);
        evaluator1_->mod_switch_to_inplace(cipher_z1[i],
                                           context1_->last_parms_id());
      }
    };

    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(multiply_then_mask, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();

    SendTwoCiphertextVector(cipher_num, cipher_z0, cipher_z1);
  }
}

void BoleContext::BOLESquare(uint32_t n, uint32_t bw, const uint64_t* x,
                             uint64_t* z) {
  const uint64_t MOD_MASK = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  const uint32_t cipher_num =
      static_cast<uint32_t>(std::ceil(1.0 * n / kPolynomialDegree));
  const uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * cipher_num / thread_num_));

  // x^2 = (x_S * x_S + x_C * x_C) + 2 * (x_S * x_C)
  ElementwiseCopyAndMul(n, z, x, x, MOD_MASK);

  if (party_id_ == kClientId) {
    // Client side
    seal::Serializable<seal::Ciphertext> dummy0 = encryptor0_->encrypt_zero();
    seal::Serializable<seal::Ciphertext> dummy1 = encryptor1_->encrypt_zero();

    std::vector<seal::Serializable<seal::Ciphertext>> ciphertext_x0(cipher_num,
                                                                    dummy0);
    std::vector<seal::Serializable<seal::Ciphertext>> ciphertext_x1(cipher_num,
                                                                    dummy1);

    auto encode_then_encrypt = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      std::vector<uint64_t> tmp_vct(kPolynomialDegree);
      seal::Plaintext tmp_plaintext(kPolynomialDegree);

      for (uint32_t i = start; i < end; i++) {
        std::memset(tmp_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_vct.data(), x + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        ElementwiseMulModInplace(kPolynomialDegree, tmp_vct.data(), 2ULL,
                                 MOD_MASK);
        encoder0_->encode(tmp_vct, tmp_plaintext);
        ciphertext_x0[i] = encryptor0_->encrypt_symmetric(tmp_plaintext);

        encoder1_->encode(tmp_vct, tmp_plaintext);
        ciphertext_x1[i] = encryptor1_->encrypt_symmetric(tmp_plaintext);
      }
    };

    auto* thread_pool = rot_context_->GetThreadPool();
    std::vector<std::future<void>> futures(thread_num_);
    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(encode_then_encrypt, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();

    SendCiphertext(cipher_num, ciphertext_x0, *io_);
    SendCiphertext(cipher_num, ciphertext_x1, *io_);

    std::vector<seal::Ciphertext> cipher_z0(cipher_num), cipher_z1(cipher_num);
    RecvTwoCiphertextVector(cipher_num, cipher_z0, cipher_z1);

    std::vector<seal::Plaintext> plain_z0(cipher_num), plain_z1(cipher_num);
    auto decrypt_then_decode = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      std::vector<uint64_t> z0(kPolynomialDegree), z1(kPolynomialDegree);
      for (uint32_t i = start; i < end; i++) {
        decryptor0_->decrypt(cipher_z0[i], plain_z0[i]);
        decryptor1_->decrypt(cipher_z1[i], plain_z1[i]);
        encoder0_->decode(plain_z0[i], z0);
        encoder1_->decode(plain_z1[i], z1);
        for (uint32_t j = 0;
             j < std::min(kPolynomialDegree, n - i * kPolynomialDegree); j++) {
          z[i * kPolynomialDegree + j] =
              (z[i * kPolynomialDegree + j] + CRT(z0[j], z1[j])) & MOD_MASK;
        }
      }
    };

    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(decrypt_then_decode, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();
  } else {
    // Server side
    std::vector<uint64_t> c(n);
    for (uint32_t i = 0; i < n; i++) {
      c[i] = static_cast<uint64_t>(rand()) & MOD_MASK;
    }
    ElementwiseSubModInplace(n, z, c.data(), MOD_MASK);

    std::vector<seal::Plaintext> plain_x0(cipher_num), plain_x1(cipher_num);
    std::vector<seal::Plaintext> plain_c0(cipher_num), plain_c1(cipher_num);

    auto encode_prg = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      std::vector<uint64_t> tmp_x_vct(kPolynomialDegree);
      std::vector<uint64_t> tmp_c_vct(kPolynomialDegree);

      for (uint32_t i = start; i < end; i++) {
        std::memset(tmp_x_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_x_vct.data(), x + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_x_vct, plain_x0[i]);
        encoder1_->encode(tmp_x_vct, plain_x1[i]);

        std::memset(tmp_c_vct.data(), 0, kPolynomialDegree * sizeof(uint64_t));
        std::memcpy(tmp_c_vct.data(), c.data() + i * kPolynomialDegree,
                    std::min(kPolynomialDegree, n - i * kPolynomialDegree) *
                        sizeof(uint64_t));
        encoder0_->encode(tmp_c_vct, plain_c0[i]);
        encoder1_->encode(tmp_c_vct, plain_c1[i]);
      }
    };

    auto* thread_pool = rot_context_->GetThreadPool();
    std::vector<std::future<void>> futures(thread_num_);
    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(encode_prg, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();

    std::vector<seal::Ciphertext> cipher_x0(cipher_num), cipher_x1(cipher_num);
    std::vector<seal::Ciphertext> cipher_z0(cipher_num), cipher_z1(cipher_num);

    RecvCiphertext(cipher_num, *context0_, cipher_x0, *io_);
    RecvCiphertext(cipher_num, *context1_, cipher_x1, *io_);

    auto multiply_then_mask = [&](uint32_t start, uint32_t end) -> void {
      if (end <= start) return;

      for (uint32_t i = start; i < end; i++) {
        evaluator0_->multiply_plain(cipher_x0[i], plain_x0[i], cipher_z0[i]);
        evaluator1_->multiply_plain(cipher_x1[i], plain_x1[i], cipher_z1[i]);

        evaluator0_->add_plain_inplace(cipher_z0[i], plain_c0[i]);
        evaluator0_->mod_switch_to_inplace(cipher_z0[i],
                                           context0_->last_parms_id());

        evaluator1_->add_plain_inplace(cipher_z1[i], plain_c1[i]);
        evaluator1_->mod_switch_to_inplace(cipher_z1[i],
                                           context1_->last_parms_id());
      }
    };

    for (uint32_t i = 0; i < thread_num_; i++) {
      futures[i] =
          thread_pool->Submit(multiply_then_mask, i * block_size,
                              std::min(cipher_num, (i + 1) * block_size));
    }
    for (auto& f : futures) f.wait();

    SendTwoCiphertextVector(cipher_num, cipher_z0, cipher_z1);
  }
}

void BoleContext::BOLESelf(uint32_t n, uint32_t bw, const uint64_t* x,
                           uint64_t* z) {
  // Self multiplication: z = x * x (local)
  const uint64_t MOD_MASK = (bw == 64) ? -1ULL : (1ULL << bw) - 1;
  for (uint32_t i = 0; i < n; i++) {
    z[i] = (x[i] * x[i]) & MOD_MASK;
  }
}

}  // namespace capricorn::operator_