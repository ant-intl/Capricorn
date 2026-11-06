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

#include "src/capricorn/activation/secure_silu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include "fmt/format.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/runtime/macro.h"

namespace capricorn::silu {

namespace {
constexpr uint64_t kSupportedInputBound = 8;
constexpr uint32_t kRotsPerElement = 62;
constexpr uint32_t kMaxOfflineInputSize =
    std::numeric_limits<uint32_t>::max() / kRotsPerElement;
}  // namespace

size_t PaddedInputSize(size_t input_size, const SiluConfig& config) {
  YACL_ENFORCE_GT(input_size, 0U);
  YACL_ENFORCE_GT(config.thread_num, 0U);
  constexpr size_t kElementsPerWorkerBlock = 8;
  YACL_ENFORCE_LE(config.thread_num,
                  std::numeric_limits<size_t>::max() / kElementsPerWorkerBlock);
  const size_t worker_block =
      static_cast<size_t>(config.thread_num) * kElementsPerWorkerBlock;
  YACL_ENFORCE_LE(input_size,
                  std::numeric_limits<size_t>::max() - (worker_block - 1));
  return ((input_size + worker_block - 1) / worker_block) * worker_block;
}

SecureSiluBase::SecureSiluBase(
    std::vector<std::shared_ptr<link::LinkIO>> spawned_link_ios,
    const uint32_t party_id, const SiluConfig& config)
    : link_ios_(std::move(spawned_link_ios)),
      config_(config),
      party_id_(party_id) {
  InitParams();
  ConvertToFixedPoint();
}

void SecureSiluBase::InitParams() {
  io_modmask_ = config_.io_bw == 64 ? -1 : ((1ULL << config_.io_bw) - 1);
  input_modmask_ =
      config_.input_bw == 64 ? -1 : ((1ULL << config_.input_bw) - 1);
  input_scale_up_ = 1ULL << config_.input_scale_bw;
  table_a_scale_up_ = 1ULL << config_.table_a_scale_bw;
  table_d_scale_up_ = 1ULL << config_.table_d_scale_bw;
  table_index_bw_ = static_cast<uint32_t>(std::log2(kSlope.size()));
}

void SecureSiluBase::ConvertToFixedPoint() {
  const uint32_t array_size = static_cast<uint32_t>(kSlope.size());
  scaled_table_a_.resize(array_size);
  scaled_table_d_.resize(array_size);

  for (uint32_t i = 0; i < array_size; i++) {
    double scaled_a =
        (kSlope[i] + 0.5) * static_cast<double>(table_a_scale_up_);
    scaled_table_a_[i] = static_cast<uint64_t>(std::round(scaled_a));

    double scaled_d =
        -1.0 * kIntercept[i] * static_cast<double>(table_d_scale_up_);
    scaled_table_d_[i] = static_cast<uint64_t>(std::round(scaled_d));
  }
}

void SecureSiluBase::Offline(uint32_t n) {
  if (rot_context_) {
    throw std::logic_error("SecureSilu offline state is already initialized");
  }
  YACL_ENFORCE_GT(n, 0U);
  YACL_ENFORCE_LE(n, kMaxOfflineInputSize,
                  "SiLU preprocessing size overflows the ROT count");
  const uint32_t rot_number = kRotsPerElement * n;

  // Initialize preprocessing (ROT, BOLE, common AES key)
  operator_::PreprocessResult preprocess_result = operator_::Preprocess(
      link_ios_, party_id_, config_.thread_num, rot_number);

  // Move contexts from preprocessing result
  rot_context_ = preprocess_result.rot_context;
  bole_context_ = std::move(preprocess_result.bole_context);

  // Initialize other contexts
  mux_context_ = std::make_unique<operator_::MuxContext>(
      party_id_, rot_context_.get(), config_.thread_num);
  compare_context_ = std::make_unique<operator_::CompareContext>(
      party_id_, rot_context_.get(), config_.thread_num);
  truncate_context_ = std::make_unique<operator_::TruncateContext>(
      party_id_, rot_context_.get(), config_.thread_num);
  ring_ext_context_ = std::make_unique<operator_::RingExtensionContext>(
      party_id_, rot_context_.get(), config_.thread_num);
  lut_context_ = std::make_unique<operator_::LookupTableContext>(
      party_id_, rot_context_.get(), config_.thread_num);
  offline_input_size_ = n;
  online_elements_consumed_ = 0;
}

void SecureSiluBase::RefillOffline(uint32_t n) {
  if (!rot_context_) {
    throw std::logic_error("SecureSilu offline state is not initialized");
  }
  if (n != offline_input_size_) {
    throw std::invalid_argument(
        fmt::format("SecureSilu refill input size must remain {}, got {}",
                    offline_input_size_, n));
  }
  rot_context_->Refill();
  online_elements_consumed_ = 0;
}

void SecureSiluBase::ComputeMSB(uint32_t n, const uint64_t* input_x,
                                std::vector<uint8_t>& msb_res) {
  // Copy input to three positions for computing:
  // msb_res[0..n-1]: x < -8
  // msb_res[n..2n-1]: x < 8
  std::vector<uint64_t> msb_input(2 * n);
  std::memcpy(msb_input.data(), input_x, n * sizeof(uint64_t));
  std::memcpy(msb_input.data() + n, input_x, n * sizeof(uint64_t));

  // Server adds threshold offsets
  if (party_id_ == operator_::kServerId) {
    operator_::ElementwiseAddmodInplace(
        n, msb_input.data(),
        static_cast<uint64_t>(4ULL << config_.input_scale_bw), input_modmask_);
    operator_::ElementwiseSubmodInplace(
        n, msb_input.data() + n,
        static_cast<uint64_t>(4ULL << config_.input_scale_bw), input_modmask_);
  } else {
    operator_::ElementwiseAddmodInplace(
        n, msb_input.data(),
        static_cast<uint64_t>(4ULL << config_.input_scale_bw), input_modmask_);
    operator_::ElementwiseSubmodInplace(
        n, msb_input.data() + n,
        static_cast<uint64_t>(4ULL << config_.input_scale_bw), input_modmask_);
  }

  // Compute MSB: input_scale_bw + input_int_bw
  compare_context_->MSB(2 * n, msb_input.data(), msb_res.data(),
                        config_.input_int_bw + config_.input_scale_bw);
}

void SecureSiluBase::ComputeLUTIndex(uint32_t n, const uint64_t* input_x,
                                     std::vector<uint8_t>& lut_index) {
  // Compute x + 8, so each + 4
  std::vector<uint64_t> x_vct(n);
  std::memcpy(x_vct.data(), input_x, n * sizeof(uint64_t));
  operator_::ElementwiseAddmodInplace(n, x_vct.data(),
                                      (uint64_t)4ULL << config_.input_scale_bw,
                                      input_modmask_);

  // Compute 1 * (x + 8) for index calculation
  std::vector<uint64_t> truncate_output(n);

  // // Truncate to get LUT index
  // truncate_context_->TruncateApproximate(n, x_vct.data(),
  // truncate_output.data(),
  //                                     config_.input_bw,
  //                                     config_.input_scale_bw - 5);
  truncate_context_->TruncateAccurate(n, x_vct.data(), truncate_output.data(),
                                      config_.input_bw, config_.input_scale_bw);

  // Convert to uint8 index
  operator_::ElementwiseCopy(n, truncate_output.data(), lut_index.data());
}

void SecureSiluBase::LookupAndExtend(uint32_t n,
                                     const std::vector<uint8_t>& lut_index,
                                     std::vector<uint64_t>& splut_x,
                                     std::vector<uint64_t>& splut_y) {
  // Lookup from two tables (slope and intercept)
  lut_context_->LookupTable2(n, lut_index.data(), splut_x.data(),
                             splut_y.data(), table_index_bw_,
                             config_.table_a_bw, config_.table_d_bw,
                             scaled_table_a_.data(), scaled_table_d_.data());

  if (config_.table_a_bw == config_.table_d_bw) {
    // Case 1: a and d are the same, if not equal to input_bw, extend both
    // together Case 3: if also equal to input_bw, no extension needed
    if (config_.table_a_bw != config_.input_bw) {
      std::vector<uint64_t> bit_ext_input(2 * n);
      std::vector<uint64_t> bit_ext_output(2 * n);
      std::memcpy(bit_ext_input.data(), splut_x.data(), n * sizeof(uint64_t));
      std::memcpy(bit_ext_input.data() + n, splut_y.data(),
                  n * sizeof(uint64_t));

      ring_ext_context_->RingExtension(2 * n, bit_ext_input.data(),
                                       bit_ext_output.data(),
                                       config_.table_a_bw, config_.input_bw);
      std::memcpy(splut_x.data(), bit_ext_output.data(), n * sizeof(uint64_t));
      std::memcpy(splut_y.data(), bit_ext_output.data() + n,
                  n * sizeof(uint64_t));
    }
  } else {
    // Case 2: a and d are different, extend each separately if needed
    if (config_.table_a_bw != config_.input_bw) {
      ring_ext_context_->RingExtension(n, splut_x.data(), splut_x.data(),
                                       config_.table_a_bw, config_.input_bw);
    }
    if (config_.table_d_bw != config_.input_bw) {
      ring_ext_context_->RingExtension(n, splut_y.data(), splut_y.data(),
                                       config_.table_d_bw, config_.input_bw);
    }
  }
}

void SecureSiluBase::ComputeBole(uint32_t n, const uint64_t* input_x,
                                 std::vector<uint64_t>& splut_x,
                                 std::vector<uint64_t>& splut_y,
                                 std::vector<uint64_t>& bole_res) {
  // Adjust splut_x for BOLE computation
  operator_::ElementwiseSubmodInplace(
      n, splut_x.data(), (uint64_t)1ULL << (config_.table_a_scale_bw - 2),
      input_modmask_);  // for a - 0.5

  // BOLE: compute a * x
  bole_context_->BOLE(n, config_.input_bw, splut_x.data(), input_x,
                      bole_res.data());

  // result = a * x - d
  auto left_shift = config_.table_a_scale_bw + config_.input_scale_bw -
                    config_.table_d_scale_bw;
  for (uint32_t i = 0; i < n; i++) {
    bole_res[i] = (bole_res[i] - (splut_y[i] << left_shift)) & input_modmask_;
  }
  // operator_::ElementwiseSubmodInplace(n, bole_res.data(), splut_y.data(),
  // input_modmask_);
}

void SecureSiluBase::FinalMux(uint32_t n, const uint64_t* input_x,
                              const std::vector<uint8_t>& msb_res,
                              const std::vector<uint64_t>& truncate_output,
                              uint64_t* res_x) {
  // Compute MUX choice bits:
  // mux_choice[0..n-1]: (-8 <= x < 8) = (x < -8) XOR (x < 8)
  // mux_choice[n..2n-1]: (x >= 8) = (x < 8) XOR 1 (server only)
  std::vector<uint8_t> mux_choice(2 * n);
  operator_::ElementwiseXor(n, msb_res.data(), msb_res.data() + n,
                            mux_choice.data());

  if (party_id_ == operator_::kServerId) {
    operator_::ElementwiseXor(n, msb_res.data() + n, static_cast<uint8_t>(1),
                              mux_choice.data() + n);
  } else {
    operator_::ElementwiseCopy(n, msb_res.data() + n, mux_choice.data() + n);
  }

  // Prepare candidate results
  std::vector<uint64_t> candidate_res(2 * n);
  std::memcpy(candidate_res.data(), truncate_output.data(),
              n * sizeof(uint64_t));
  std::memcpy(candidate_res.data() + n, input_x, n * sizeof(uint64_t));

  // MUX selection
  std::vector<uint64_t> mux_output(2 * n);
  mux_context_->Mux(2 * n, mux_choice.data(), candidate_res.data(),
                    mux_output.data(), config_.input_bw);

  // Final addition
  operator_::ElementwiseAddmod(n, mux_output.data(), mux_output.data() + n,
                               res_x, input_modmask_);
}

std::vector<uint64_t> SecureSiluBase::Online(const std::vector<uint64_t>& x) {
  YACL_ENFORCE(rot_context_ != nullptr,
               "SecureSiluBase::Offline must precede Online");
  YACL_ENFORCE_GT(x.size(), 0U);
  YACL_ENFORCE_LE(x.size(), std::numeric_limits<uint32_t>::max());
  YACL_ENFORCE_LE(x.size(), offline_input_size_ - online_elements_consumed_,
                  "SiLU session exhausted preprocessed correlations");
  // Reserve before entering a protocol that may consume only a partial prefix.
  online_elements_consumed_ += x.size();
  const auto n = static_cast<uint32_t>(x.size());
  SPDLOG_INFO("Start Online computation");

  // Reduce Ring io_bw->input_bw: like 64 -> 32
  std::vector<uint64_t> input_x(n);
  ring_ext_context_->RingExtension(n, x.data(), input_x.data(), config_.io_bw,
                                   config_.input_bw);
  SPDLOG_INFO("[RingReduce] Done");

  // if input_scale_bw > io_scale_bw, need truncate
  if (config_.io_scale_bw > config_.input_scale_bw) {
    SPDLOG_INFO("Truncate input");
    truncate_context_->TruncateAccurate(
        n, input_x.data(), input_x.data(), config_.input_bw,
        config_.io_scale_bw - config_.input_scale_bw);
  }

  // Step 1: Compute MSB for threshold comparisons
  std::vector<uint8_t> msb_res(2 * n);
  ComputeMSB(n, input_x.data(), msb_res);
  SPDLOG_INFO("[MSB] Done");

  // Step 2: Compute LUT index
  std::vector<uint8_t> lut_index(n);
  ComputeLUTIndex(n, input_x.data(), lut_index);
  SPDLOG_INFO("[LUT index] Done");

  // Step 3: Lookup and extend
  std::vector<uint64_t> splut_x(n);
  std::vector<uint64_t> splut_y(n);
  LookupAndExtend(n, lut_index, splut_x, splut_y);
  SPDLOG_INFO("[LUT and Ring extension] Done");

  // Step 4: BOLE computation
  std::vector<uint64_t> bole_res(n);
  ComputeBole(n, input_x.data(), splut_x, splut_y, bole_res);
  SPDLOG_INFO("[BOLE] Done");

  // Step 5: Truncate
  std::vector<uint64_t> truncate_res(n);
  truncate_context_->TruncateApproximate(
      n, bole_res.data(), truncate_res.data(), config_.input_bw,
      config_.table_a_scale_bw + config_.input_scale_bw -
          config_.io_scale_bw);  // not table_scale_bw, because need extend to
                                 // io_scale_bw

  // Step 6: Final MUX
  std::vector<uint64_t> mux_res(n);
  FinalMux(n, input_x.data(), msb_res, truncate_res, mux_res.data());
  SPDLOG_INFO("[Final MUX] Done");

  // On the supported input interval [-8, 8), the SiLU output also lies in
  // [-8, 8). Shift it positive so ring extension cannot miss a lower-bit
  // carry at the signed boundary, then remove the shift in the output ring.
  std::vector<uint64_t> output(n);
  const uint64_t output_offset = kSupportedInputBound << config_.io_scale_bw;
  ring_ext_context_->RingExtensionWithOffset(n, mux_res.data(), output.data(),
                                             config_.input_bw, config_.io_bw,
                                             output_offset);

  return output;
}

SecureSiluSession::SecureSiluSession(std::shared_ptr<yacl::link::Context> lctx,
                                     uint32_t party_id,
                                     const SiluConfig& config,
                                     std::string link_tag_prefix)
    : lctx_(std::move(lctx)),
      party_id_(party_id),
      config_(config),
      link_tag_prefix_(std::move(link_tag_prefix)) {
  if (!lctx_) {
    throw std::invalid_argument("SecureSiluSession link context is null");
  }
  YACL_ENFORCE(party_id_ == operator_::kServerId ||
               party_id_ == operator_::kClientId);
  YACL_ENFORCE_GT(config_.thread_num, 0U);
  YACL_ENFORCE(!link_tag_prefix_.empty());
}

uint64_t SecureSiluSession::GenerationCountLocked() const {
  uint64_t total = prepared_base_ ? prepared_base_->GetGenerationCount() : 0;
  for (const auto& [input_size, state] : states_) {
    static_cast<void>(input_size);
    if (state.silu) {
      total += state.silu->GetGenerationCount();
    }
  }
  return total;
}

uint64_t SecureSiluSession::GetInitializationCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialization_count_;
}

uint64_t SecureSiluSession::GetGenerationCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return GenerationCountLocked();
}

std::vector<uint64_t> SecureSiluSession::Evaluate(
    const std::vector<uint64_t>& input_x) {
  std::lock_guard<std::mutex> lock(mutex_);
  YACL_ENFORCE(!poisoned_, "SecureSiluSession is poisoned");
  YACL_ENFORCE(mode_ != Mode::kPrepared,
               "Evaluate cannot be mixed with Prepare/Online");
  if (input_x.empty() || input_x.size() > kMaxOfflineInputSize) {
    throw std::invalid_argument("SecureSiluSession input size is unsupported");
  }

  mode_ = Mode::kRefill;
  try {
    const auto input_size = static_cast<uint32_t>(input_x.size());
    const std::string party_name =
        party_id_ == operator_::kServerId ? "server" : "client";
    auto [state_it, inserted] = states_.try_emplace(input_size);
    auto& state = state_it->second;
    if (inserted) {
      auto spawned_link_ios =
          link::SpawnLinkIOs(lctx_, config_.thread_num, link_tag_prefix_);
      state.timer_io = spawned_link_ios[0];
      state.silu = std::make_unique<SecureSiluBase>(std::move(spawned_link_ios),
                                                    party_id_, config_);
      {
        ScopedCommTimer<link::LinkIO> timer(
            state.timer_io, fmt::format("[{}_SILU_INITIALIZE]", party_name));
        state.silu->Offline(input_size);
      }
      ++initialization_count_;
    } else {
      ScopedCommTimer<link::LinkIO> timer(
          state.timer_io, fmt::format("[{}_SILU_REFILL]", party_name));
      state.silu->RefillOffline(input_size);
    }
    SPDLOG_INFO(
        "SecureSiluSession initialization_count={}, generation_count={}",
        initialization_count_, GenerationCountLocked());
    ScopedCommTimer<link::LinkIO> timer(
        state.timer_io, fmt::format("[{}_SILU_ONLINE]", party_name));
    return state.silu->Online(input_x);
  } catch (...) {
    poisoned_ = true;
    throw;
  }
}

void SecureSiluSession::Prepare(size_t logical_input_size, size_t query_count) {
  std::lock_guard<std::mutex> lock(mutex_);
  YACL_ENFORCE(!poisoned_, "SecureSiluSession is poisoned");
  YACL_ENFORCE(mode_ == Mode::kUnused,
               "Prepare requires an unused session and cannot follow Evaluate");
  YACL_ENFORCE_GT(query_count, 0U);
  const size_t padded_size = PaddedInputSize(logical_input_size, config_);
  YACL_ENFORCE_LE(query_count, kMaxOfflineInputSize / padded_size,
                  "SiLU query preprocessing size overflows the ROT count");
  const size_t hardware_threads = std::thread::hardware_concurrency();
  YACL_ENFORCE_GT(
      hardware_threads, config_.thread_num,
      "SecureSilu requires more hardware threads than configured workers");

  mode_ = Mode::kPrepared;
  logical_input_size_ = logical_input_size;
  padded_input_size_ = padded_size;
  query_count_ = query_count;
  try {
    auto spawned_link_ios =
        link::SpawnLinkIOs(lctx_, config_.thread_num, link_tag_prefix_);
    prepared_base_ = std::make_unique<SecureSiluBase>(
        std::move(spawned_link_ios), party_id_, config_);
    prepared_base_->Offline(
        static_cast<uint32_t>(padded_input_size_ * query_count_));
    ++initialization_count_;
  } catch (...) {
    poisoned_ = true;
    throw;
  }
}

std::vector<uint64_t> SecureSiluSession::Online(
    size_t logical_input_size, const std::vector<uint64_t>& padded_input) {
  std::lock_guard<std::mutex> lock(mutex_);
  YACL_ENFORCE(mode_ == Mode::kPrepared && !poisoned_,
               "SecureSiluSession is not prepared or is poisoned");
  YACL_ENFORCE_EQ(logical_input_size, logical_input_size_,
                  "SiLU logical input shape changed within a session");
  YACL_ENFORCE_EQ(padded_input.size(), padded_input_size_,
                  "SiLU padded input shape changed within a session");
  YACL_ENFORCE_LT(queries_used_, query_count_,
                  "SiLU session exhausted its prepared query count");

  // Reserve before entering the protocol. On failure, never retry ROTs.
  ++queries_used_;
  try {
    return prepared_base_->Online(padded_input);
  } catch (...) {
    poisoned_ = true;
    throw;
  }
}

bool SecureSiluSession::is_prepared() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_ == Mode::kPrepared;
}

size_t SecureSiluSession::logical_input_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return logical_input_size_;
}

size_t SecureSiluSession::padded_input_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return padded_input_size_;
}

size_t SecureSiluSession::query_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return query_count_;
}

size_t SecureSiluSession::queries_used() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queries_used_;
}

std::vector<uint64_t> SecureSilu(std::shared_ptr<yacl::link::Context> lctx,
                                 const std::vector<uint64_t>& input_x,
                                 uint32_t party_id, const SiluConfig& config) {
  SecureSiluSession session(std::move(lctx), party_id, config);
  return session.Evaluate(input_x);
}

// ony for test
std::vector<uint64_t> SecureSiluWitLinks(
    std::vector<std::shared_ptr<link::LinkIO>> spawned_link_ios,
    const std::vector<uint64_t>& input_x, uint32_t party_id,
    const SiluConfig& config) {
  std::string party_name =
      party_id == operator_::kServerId ? "server" : "client";

  ScopedCommTimer<link::LinkIO> timer(spawned_link_ios[0],
                                      fmt::format("[{}_SILU]", party_name));

  auto silu =
      std::make_unique<SecureSiluBase>(spawned_link_ios, party_id, config);
  {
    ScopedCommTimer<link::LinkIO> timer_offline(
        spawned_link_ios[0], fmt::format("[{}_SILU_OFFLINE]", party_name));
    silu->Offline(input_x.size());
  }

  ScopedCommTimer<link::LinkIO> timer_online(
      spawned_link_ios[0], fmt::format("[{}_SILU_ONLINE]", party_name));
  auto result = silu->Online(input_x);
  return result;
}

}  // namespace capricorn::silu
