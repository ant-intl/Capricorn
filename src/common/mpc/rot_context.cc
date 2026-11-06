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

#include "src/common/mpc/rot_context.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <thread>

#include "src/common/runtime/macro.h"

namespace {
std::string GetCotDataDir() {
  const char* writable_dir = std::getenv("COT_DATA_DIR");
  if (writable_dir && writable_dir[0] != '\0') {
    return writable_dir;
  }
  throw capricorn::operator_::RotContextError(
      "COT_DATA_DIR must point to a writable persistent cache directory");
}

// Ensure the data directory exists (for writing)
void EnsureDataDirExists() {
  std::filesystem::create_directories(GetCotDataDir());
}

// Get full path for COT data file
std::string GetCotFilePath(const std::string& filename) {
  return GetCotDataDir() + "/" + filename;
}
}  // namespace

namespace capricorn::operator_ {

RotContext::RotContext(uint32_t party_id, uint32_t thread_number,
                       uint32_t rot_number)
    : party_id_(party_id),
      thread_number_(thread_number),
      io_(nullptr),
      thread_pool_(thread_number_) {
  if (rot_number == 0) {
    rot_number_ = thread_number_ * kRcotBufferSize;
  } else {
    rot_number_ =
        ((rot_number + thread_number_ - 1) / thread_number_) * thread_number_;
  }

  // One RCOT worker is still required for a single-thread context. Use ceiling
  // division so an odd ROT count is fully covered by the worker ranges.
  rcot_blocks_ = std::max(1u, thread_number_ / 2);
  rcot_buffer_size_ = (rot_number_ + rcot_blocks_ - 1) / rcot_blocks_;

  ValidateParameters();

  sender_msg0_.resize(rot_number_);
  sender_msg1_.resize(rot_number_);
  recv_msg_.resize(rot_number_);
  recv_choice_.resize(rot_number_);
  rcot_send_array_.resize(rcot_blocks_);
  rcot_recv_array_.resize(rcot_blocks_);
  delta_.resize(rcot_blocks_);
}

void RotContext::ValidateParameters() const {
  if (party_id_ != kServerId && party_id_ != kClientId) {
    throw RotContextError("Invalid party_id: must be " +
                          std::to_string(kServerId) + " (server) or " +
                          std::to_string(kClientId) +
                          " (client), got: " + std::to_string(party_id_));
  }
  if (thread_number_ == 0) {
    throw RotContextError("thread_number must be greater than 0");
  }
  if (rot_number_ < kMinRotNumber) {
    throw RotContextError("rot_number must be at least " +
                          std::to_string(kMinRotNumber) +
                          ", got: " + std::to_string(rot_number_));
  }
}

uint32_t RotContext::ReserveRot(uint64_t count, uint32_t& used,
                                const char* direction) {
  if (used > rot_number_ || count > static_cast<uint64_t>(rot_number_ - used)) {
    throw RotContextError(
        std::string("Insufficient ") + direction + " ROTs: requested " +
        std::to_string(count) + ", remaining " +
        std::to_string(used <= rot_number_ ? rot_number_ - used : 0));
  }

  const uint32_t offset = used;
  used += static_cast<uint32_t>(count);
  return offset;
}

uint32_t RotContext::ReserveSenderRot(uint64_t count) {
  return ReserveRot(count, send_rot_used_, "sender");
}

uint32_t RotContext::ReserveReceiverRot(uint64_t count) {
  return ReserveRot(count, recv_rot_used_, "receiver");
}

void RotContext::Init(
    const std::vector<std::shared_ptr<link::LinkIO>>& io_array) {
  // Validate io_array
  if (io_array.empty()) {
    throw RotContextError("io_array cannot be empty");
  }
  if (io_array.size() != thread_number_) {
    throw RotContextError("io_array size (" + std::to_string(io_array.size()) +
                          ") must match thread_number (" +
                          std::to_string(thread_number_) + ")");
  }

  // Validate all pointers are non-null
  io_array_.resize(io_array.size());
  for (size_t i = 0; i < io_array.size(); ++i) {
    if (io_array[i].get() == nullptr) {
      throw RotContextError("io_array[" + std::to_string(i) + "] is null");
    }
    io_array_[i] = io_array[i].get();
  }

  io_ = io_array_[0];

  InitFerretCOT();
  PreprocessingROT();
}

void RotContext::Refill() {
  if (rcot_send_array_.empty() || rcot_recv_array_.empty() || io_ == nullptr) {
    throw RotContextError("RotContext must be initialized before refill");
  }

  PreprocessingROT();
  send_rot_used_ = 0;
  recv_rot_used_ = 0;
}

void RotContext::InitFerretCOT() {
  EnsureDataDirExists();

  auto rcot_init = [&](uint32_t i) -> void {
    if (party_id_ == kServerId) {
      // Server logic
      rcot_send_array_[i] = std::make_shared<emp::FerretCOT<link::LinkIO>>(
          emp::ALICE, 1, (link::LinkIO**)(&io_array_[i]), false, true,
          emp::ferret_b13,
          GetCotFilePath("server_send" + std::to_string(i) + ".txt"));
      rcot_recv_array_[i] = std::make_shared<emp::FerretCOT<link::LinkIO>>(
          emp::BOB, 1, (link::LinkIO**)(&io_array_[i]), false, true,
          emp::ferret_b13,
          GetCotFilePath("server_receive" + std::to_string(i) + ".txt"));
    } else {
      // Client logic
      rcot_recv_array_[i] = std::make_shared<emp::FerretCOT<link::LinkIO>>(
          emp::BOB, 1, (link::LinkIO**)(&io_array_[i]), false, true,
          emp::ferret_b13,
          GetCotFilePath("client_receive" + std::to_string(i) + ".txt"));
      rcot_send_array_[i] = std::make_shared<emp::FerretCOT<link::LinkIO>>(
          emp::ALICE, 1, (link::LinkIO**)(&io_array_[i]), false, true,
          emp::ferret_b13,
          GetCotFilePath("client_send" + std::to_string(i) + ".txt"));
    }
    delta_[i] = rcot_send_array_[i]->Delta;
  };

  std::vector<std::future<void>> futures(rcot_blocks_);
  for (uint32_t i = 0; i < rcot_blocks_; i++) {
    futures[i] = thread_pool_.Submit(rcot_init, i);
  }
  for (auto& f : futures) {
    f.get();
  }

  rcot_delta_ = rcot_send_array_[0]->Delta;
}

void RotContext::PreprocessingROT() {
  // Generate and exchange hash seeds
  std::vector<emp::block> hash_seed(thread_number_);
  if (party_id_ == kServerId) {
    emp::PRG prg;
    prg.random_block(hash_seed.data(), static_cast<int>(thread_number_));
    io_->send_block(hash_seed.data(), static_cast<int>(thread_number_));
    io_->flush();
  } else {
    io_->recv_block(hash_seed.data(), static_cast<int>(thread_number_));
  }

  // Prepare pointers for parallel processing
  // Note: sender_msg0_.data() already returns emp::block*, no reinterpret_cast
  // needed
  emp::block* rot_data0 = sender_msg0_.data();
  emp::block* rot_data1 = sender_msg1_.data();
  emp::block* rot_data = recv_msg_.data();

  // Lambda for parallel RCOT processing
  auto rcot_prg = [&](uint32_t t_id, uint32_t start, uint32_t end) -> void {
    if (party_id_ == kServerId) {
      rcot_send_array_[t_id]->rcot(rot_data0 + start,
                                   static_cast<int>(end - start));
      rcot_recv_array_[t_id]->rcot(rot_data + start,
                                   static_cast<int>(end - start));
    } else {
      rcot_recv_array_[t_id]->rcot(rot_data + start,
                                   static_cast<int>(end - start));
      rcot_send_array_[t_id]->rcot(rot_data0 + start,
                                   static_cast<int>(end - start));
    }
    io_array_[t_id]->sync();

    // Hash: RCOT -> ROT
    emp::MITCCRH<kOtBatchSize> mitccrh;
    mitccrh.setS(hash_seed[t_id]);
    std::array<emp::block, kOtBatchSize * 2> pad;

    for (uint32_t i = start; i < end; i++) {
      recv_choice_[i] = emp::getLSB(rot_data[i]);
    }

    if (party_id_ == kServerId) {
      // HASH server -> client
      for (uint32_t i = start; i < end; i += kOtBatchSize) {
        const uint32_t batch_end = std::min(i + kOtBatchSize, end);
        for (uint32_t j = i; j < batch_end; ++j) {
          pad[2 * (j - i)] = rot_data0[j];
          pad[2 * (j - i) + 1] = rot_data0[j] ^ delta_[t_id];
        }
        mitccrh.hash<kOtBatchSize, 2>(pad.data());
        for (uint32_t j = i; j < batch_end; ++j) {
          rot_data0[j] = pad[2 * (j - i)];
          rot_data1[j] = pad[2 * (j - i) + 1];
        }
      }

      // HASH server <- client
      for (uint32_t i = start; i < end; i += kOtBatchSize) {
        const uint32_t chunk_size = std::min(kOtBatchSize, end - i);
        std::memcpy(pad.data(), rot_data + i, chunk_size * sizeof(emp::block));
        mitccrh.hash<kOtBatchSize, 1>(pad.data());
        std::memcpy(rot_data + i, pad.data(), chunk_size * sizeof(emp::block));
      }
    } else {
      // HASH server -> client
      for (uint32_t i = start; i < end; i += kOtBatchSize) {
        const uint32_t chunk_size = std::min(kOtBatchSize, end - i);
        std::memcpy(pad.data(), rot_data + i, chunk_size * sizeof(emp::block));
        mitccrh.hash<kOtBatchSize, 1>(pad.data());
        std::memcpy(rot_data + i, pad.data(), chunk_size * sizeof(emp::block));
      }

      // HASH server <- client
      for (uint32_t i = start; i < end; i += kOtBatchSize) {
        const uint32_t batch_end = std::min(i + kOtBatchSize, end);
        for (uint32_t j = i; j < batch_end; ++j) {
          pad[2 * (j - i)] = rot_data0[j];
          pad[2 * (j - i) + 1] = rot_data0[j] ^ delta_[t_id];
        }
        mitccrh.hash<kOtBatchSize, 2>(pad.data());
        for (uint32_t j = i; j < batch_end; ++j) {
          rot_data0[j] = pad[2 * (j - i)];
          rot_data1[j] = pad[2 * (j - i) + 1];
        }
      }
    }
  };

  // Execute in parallel
  std::vector<std::future<void>> futures(rcot_blocks_);
  for (uint32_t i = 0; i < rcot_blocks_; i++) {
    uint32_t start = i * rcot_buffer_size_;
    uint32_t end = std::min(rot_number_, (i + 1) * rcot_buffer_size_);
    futures[i] = thread_pool_.Submit(rcot_prg, i, start, end);
  }
  for (auto& f : futures) {
    f.get();
  }

  // Exchange common AES key for SPLUT protocols
  if (party_id_ == kServerId) {
    emp::PRG prg;
    prg.random_block(&common_aes_key_, 1);
    io_->send_block(&common_aes_key_, 1);
    io_->flush();
  } else {
    io_->recv_block(&common_aes_key_, 1);
  }

  ++generation_count_;
}

}  // namespace capricorn::operator_
