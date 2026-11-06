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
#include <stdexcept>
#include <vector>

#include "emp-ot/emp-ot.h"
#include "emp-tool/emp-tool.h"
#include "src/common/runtime/link/link_io.h"

#include "yacl/utils/thread_pool.h"

namespace capricorn::operator_ {

// Constants
constexpr uint32_t kRcotBufferSize = 10'000'000;
constexpr uint32_t kOtBatchSize = 256;  // ot_bsize
constexpr uint32_t kMinRotNumber = 1;   // Minimum ROT number

// Party IDs
constexpr uint32_t kServerId = 1;
constexpr uint32_t kClientId = 2;

class RotContextError : public std::runtime_error {
 public:
  explicit RotContextError(const std::string& msg) : std::runtime_error(msg) {}
};
class RotContext {
 public:
  explicit RotContext(uint32_t party_id, uint32_t thread_number,
                      uint32_t rot_number = 0);

  ~RotContext() = default;

  // Non-copyable
  RotContext(const RotContext&) = delete;
  RotContext& operator=(const RotContext&) = delete;

  void Init(const std::vector<std::shared_ptr<link::LinkIO>>& io_array);

  // Generate a fresh ROT batch from the existing Ferret COT state. The caller
  // must ensure that no protocol operation is still using the current batch.
  void Refill();

  // Getters for IO resources
  link::LinkIO* GetIO() const noexcept { return io_; }
  const std::vector<link::LinkIO*>& GetIOArray() const noexcept {
    return io_array_;
  }
  uint32_t GetPartyId() const noexcept { return party_id_; }
  uint32_t GetRotNumber() const noexcept { return rot_number_; }
  uint64_t GetGenerationCount() const noexcept { return generation_count_; }

  // ROT data access (for sender)
  emp::block* GetSenderMsg0() noexcept { return sender_msg0_.data(); }
  emp::block* GetSenderMsg1() noexcept { return sender_msg1_.data(); }

  // ROT data access (for receiver)
  emp::block* GetRecvMsg() noexcept { return recv_msg_.data(); }
  uint8_t* GetRecvChoice() noexcept { return recv_choice_.data(); }

  // ROT counters
  uint32_t& SendRotUsed() noexcept { return send_rot_used_; }
  uint32_t& RecvRotUsed() noexcept { return recv_rot_used_; }
  const uint32_t& SendRotUsed() const noexcept { return send_rot_used_; }
  const uint32_t& RecvRotUsed() const noexcept { return recv_rot_used_; }

  // Atomically validate and reserve a contiguous range of preprocessed ROTs.
  uint32_t ReserveSenderRot(uint64_t count);
  uint32_t ReserveReceiverRot(uint64_t count);

  // RCOT counters
  uint32_t& SendRcotUsed() noexcept { return send_rcot_used_; }
  uint32_t& RecvRcotUsed() noexcept { return recv_rcot_used_; }
  const uint32_t& SendRcotUsed() const noexcept { return send_rcot_used_; }
  const uint32_t& RecvRcotUsed() const noexcept { return recv_rcot_used_; }

  // Common AES key for SPLUT protocols
  const emp::block& GetCommonAesKey() const noexcept { return common_aes_key_; }

  // Size getters
  size_t GetSenderMsgSize() const noexcept { return sender_msg0_.size(); }
  yacl::ThreadPool* GetThreadPool() noexcept { return &thread_pool_; }

 private:
  uint32_t party_id_;
  uint32_t thread_number_;
  uint32_t rot_number_;
  uint32_t rcot_buffer_size_;

  // IO resources (not owned)
  std::vector<link::LinkIO*> io_array_;
  link::LinkIO* io_;

  // FerretCOT instances (owned)
  std::vector<std::shared_ptr<emp::FerretCOT<link::LinkIO>>> rcot_send_array_;
  std::vector<std::shared_ptr<emp::FerretCOT<link::LinkIO>>> rcot_recv_array_;

  // Delta values
  std::vector<emp::block> delta_;
  emp::block rcot_delta_;

  // ROT preprocessing data
  std::vector<emp::block> sender_msg0_;
  std::vector<emp::block> sender_msg1_;
  std::vector<emp::block> recv_msg_;
  std::vector<uint8_t> recv_choice_;
  uint32_t send_rot_used_{0};
  uint32_t recv_rot_used_{0};

  // RCOT preprocessing data
  // std::vector<emp::block> rcot_sender_msg_;
  // std::vector<emp::block> rcot_recv_msg_;
  // std::vector<uint8_t> rcot_recv_choice_;
  uint32_t send_rcot_used_{0};
  uint32_t recv_rcot_used_{0};
  uint64_t generation_count_{0};

  // Common AES key for SPLUT protocols
  emp::block common_aes_key_;

  yacl::ThreadPool thread_pool_;
  uint32_t rcot_blocks_;

  // Initialization methods
  void InitFerretCOT();
  void PreprocessingROT();

  // Validation helper
  void ValidateParameters() const;
  uint32_t ReserveRot(uint64_t count, uint32_t& used, const char* direction);
};

}  // namespace capricorn::operator_
