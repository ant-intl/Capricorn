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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/and_protocol.h"
#include "src/common/mpc/b2a.h"
#include "src/common/mpc/bole_context.h"
#include "src/common/mpc/compare.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/lookup_table.h"
#include "src/common/mpc/mux.h"
#include "src/common/mpc/preprocess.h"
#include "src/common/mpc/ring_extension.h"
#include "src/common/mpc/rot_context.h"
#include "src/common/mpc/truncate.h"
#include "src/common/runtime/link/communicator.h"
#include "src/common/runtime/link/link_io.h"
#include "src/common/runtime/link/mock_link_io.h"

#include "yacl/link/context.h"
// #include "yacl/utils/thread_pool.h"

namespace capricorn::silu {

namespace {
// SiLU piecewise linear approximation parameters
// These are the coefficients for 16 segments of the SiLU function

// [-6, 2]
// const std::vector<double> kSlope = {
//     -0.0150650647, -0.0221113302, -0.0318989865, -0.0449510142,
//     -0.0612609833, -0.0794050470, -0.0949306637, -0.0980040953,
//     -0.0713476390, 0.0081698885,  0.1591725200,  0.3770438731,
//     0.6229562163,  0.8408275247,  0.9918303490,  1.0713477135};

// const std::vector<double> kIntercept = {
//     -0.1049897894, -0.1436634958, -0.1925002635, -0.2511205673,
//     -0.3162630200, -0.3797549903, -0.4265414476, -0.4348786175,
//     -0.3829230964, -0.2657985985, -0.1173049808, -0.0101515148,
//     -0.0101515157, -0.1173049957, -0.2657987177, -0.3829231858};

// [-8, 8]
const std::vector<double> kSlope = {
    -0.0036530436, -0.0083718523, -0.0184673574, -0.0382546298,
    -0.0703148022, -0.0974453092, -0.0348083526, 0.2654413581,
    0.7345586419,  1.0348082781,  1.0974453688,  1.0703147650,
    1.0382546186,  1.0184673071,  1.0083717108,  1.0036530495};

const std::vector<double> kIntercept = {
    -0.0316713527, -0.0644570142, -0.1245436966, -0.2226767540,
    -0.3502154350, -0.4335395098, -0.3192403316, -0.0378005207,
    -0.0378005020, -0.3192403913, -0.4335394502, -0.3502155542,
    -0.2226768285, -0.1245449260, -0.0644548759, -0.0316730142};
}  // namespace

/// SiLU configuration parameters
struct SiluConfig {
  uint32_t io_bw = 64;             // Input/Output bit width
  uint32_t io_scale_bw = 18;       // Input/Output scaled bit width
  uint32_t input_bw = 40;          // SiLU input fixed-point bit width
  uint32_t input_int_bw = 8;       // Internal representation width, not domain
  uint32_t input_scale_bw = 18;    // SiLU scaled bit width
  uint32_t table_a_bw = 14;        // 2+12
  uint32_t table_d_bw = 20;        // 2+18
  uint32_t table_a_scale_bw = 12;  // Fixed-point scale (fractional bits)
  uint32_t table_d_scale_bw = 18;  // Fixed-point scale (fractional bits)
  uint32_t thread_num = 16;        // Number of threads
};

// Round up to a non-empty, byte-aligned slice for each LinkIO worker.
size_t PaddedInputSize(size_t input_size,
                       const SiluConfig& config = SiluConfig{});

/**
 * @brief SecureSiLU base class
 *
 * Implements the secure SiLU (Sigmoid Linear Unit) activation function
 * using piecewise linear approximation and secure multi-party computation.
 * Input values are supported in the fixed-point interval [-8, 8).
 */
class SecureSiluBase {
 public:
  // Constructor that takes ownership of SpawnedLinkIOs
  SecureSiluBase(std::vector<std::shared_ptr<link::LinkIO>> spawned_link_ios,
                 const uint32_t party_id,
                 const SiluConfig& config = SiluConfig{});

  ~SecureSiluBase() {
    if (!rot_context_) {
      return;
    }
    auto all_rot_number =
        rot_context_->SendRotUsed() + rot_context_->RecvRotUsed();
    SPDLOG_INFO("all_rot_number: {}", all_rot_number);
  }

  // Non-copyable
  SecureSiluBase(const SecureSiluBase&) = delete;
  SecureSiluBase& operator=(const SecureSiluBase&) = delete;

  void Offline(uint32_t n);
  void RefillOffline(uint32_t n);
  std::vector<uint64_t> Online(const std::vector<uint64_t>& x);
  uint64_t GetGenerationCount() const noexcept {
    return rot_context_ ? rot_context_->GetGenerationCount() : 0;
  }

 private:
  void InitParams();
  void ConvertToFixedPoint();

  // SiLU computation steps
  void ComputeMSB(uint32_t n, const uint64_t* input_x,
                  std::vector<uint8_t>& msb_res);
  void ComputeLUTIndex(uint32_t n, const uint64_t* input_x,
                       std::vector<uint8_t>& lut_index);
  void LookupAndExtend(uint32_t n, const std::vector<uint8_t>& lut_index,
                       std::vector<uint64_t>& splut_x,
                       std::vector<uint64_t>& splut_y);
  void ComputeBole(uint32_t n, const uint64_t* input_x,
                   std::vector<uint64_t>& splut_x,
                   std::vector<uint64_t>& splut_y,
                   std::vector<uint64_t>& bole_res);
  void FinalMux(uint32_t n, const uint64_t* input_x,
                const std::vector<uint8_t>& msb_res,
                const std::vector<uint64_t>& truncate_output, uint64_t* res_x);

  void Reveal64(const string& s, uint32_t i, uint64_t x, uint64_t MOD_MASK,
                uint32_t BW) {
    auto io_ = link_ios_[i];
    if (party_id_ == operator_::kServerId) {
      uint64_t x0;
      io_->recv_data(&x0, sizeof(uint64_t));
      std::cout << "[" << i << "]" << s
                << operator_::GetSignedValue((x + x0) & MOD_MASK, BW) << endl;
    } else {
      io_->send_data(&x, sizeof(uint64_t));
      io_->flush();
    }
  }

  std::vector<std::shared_ptr<link::LinkIO>> link_ios_;

  // Configuration
  SiluConfig config_;
  uint64_t io_modmask_;
  uint64_t input_modmask_;
  uint64_t input_scale_up_;
  uint64_t table_a_scale_up_;
  uint64_t table_d_scale_up_;
  uint32_t table_index_bw_;

  // Lookup table data (piecewise linear approximation coefficients)
  std::vector<uint64_t> scaled_table_a_;  // Slopes
  std::vector<uint64_t> scaled_table_d_;  // Intercepts

  uint32_t party_id_;
  uint32_t offline_input_size_{0};
  uint64_t online_elements_consumed_{0};
  // Protocol contexts
  std::shared_ptr<operator_::RotContext> rot_context_;
  std::unique_ptr<operator_::BoleContext> bole_context_;
  std::unique_ptr<operator_::TruncateContext> truncate_context_;
  std::unique_ptr<operator_::RingExtensionContext> ring_ext_context_;
  std::unique_ptr<operator_::MuxContext> mux_context_;
  std::unique_ptr<operator_::LookupTableContext> lut_context_;
  std::unique_ptr<operator_::CompareContext> compare_context_;

  // yacl::ThreadPool thread_pool_;
};

// Owns SecureSiLU preprocessing in one of two mutually exclusive modes:
// - Evaluate keeps per-size Ferret state and refills fresh ROTs on each call.
// - Prepare/Online precomputes a fixed query budget for benchmark timing.
// An exception escaping protocol evaluation poisons the session; its
// correlations cannot be retried.
class SecureSiluSession {
 public:
  SecureSiluSession(std::shared_ptr<yacl::link::Context> lctx,
                    uint32_t party_id, const SiluConfig& config = SiluConfig{},
                    std::string link_tag_prefix = "silu-worker");

  SecureSiluSession(const SecureSiluSession&) = delete;
  SecureSiluSession& operator=(const SecureSiluSession&) = delete;

  std::vector<uint64_t> Evaluate(const std::vector<uint64_t>& input_x);

  void Prepare(size_t logical_input_size, size_t query_count);
  std::vector<uint64_t> Online(size_t logical_input_size,
                               const std::vector<uint64_t>& padded_input);

  uint64_t GetInitializationCount() const;
  uint64_t GetGenerationCount() const;
  bool is_prepared() const;
  size_t logical_input_size() const;
  size_t padded_input_size() const;
  size_t query_count() const;
  size_t queries_used() const;
  const SiluConfig& config() const { return config_; }

 private:
  enum class Mode { kUnused, kRefill, kPrepared };

  struct SizeState {
    std::shared_ptr<link::LinkIO> timer_io;
    std::unique_ptr<SecureSiluBase> silu;
  };

  // Call only while holding mutex_.
  uint64_t GenerationCountLocked() const;

  std::shared_ptr<yacl::link::Context> lctx_;
  uint32_t party_id_;
  SiluConfig config_;
  std::string link_tag_prefix_;
  Mode mode_{Mode::kUnused};
  bool poisoned_{false};
  mutable std::mutex mutex_;
  uint64_t initialization_count_{0};

  // Refill mode retains independent state for each public input size.
  std::unordered_map<uint32_t, SizeState> states_;

  // Prepared mode consumes disjoint prefixes of one preprocessed ROT batch.
  std::unique_ptr<SecureSiluBase> prepared_base_;
  size_t logical_input_size_{0};
  size_t padded_input_size_{0};
  size_t query_count_{0};
  size_t queries_used_{0};
};

/**
 * @brief Convenience function to execute SecureSiLU
 * @note Input values outside the fixed-point interval [-8, 8) are unsupported.
 * @param lctx Link context for communication
 * @param input_x Secret-shared input values
 * @param party_id Server ID (1) or Client ID (2)
 * @return Secret-shared SiLU result
 */
std::vector<uint64_t> SecureSilu(std::shared_ptr<yacl::link::Context> lctx,
                                 const std::vector<uint64_t>& input_x,
                                 uint32_t party_id,
                                 const SiluConfig& config = SiluConfig{});

std::vector<uint64_t> SecureSiluWitLinks(
    std::vector<std::shared_ptr<link::LinkIO>> spawned_link_ios,
    const std::vector<uint64_t>& input_x, uint32_t party_id,
    const SiluConfig& config = SiluConfig{});

}  // namespace capricorn::silu
