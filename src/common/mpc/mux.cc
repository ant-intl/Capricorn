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

#include "src/common/mpc/mux.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/io_helper.h"

namespace capricorn::operator_ {

MuxContext::MuxContext(const uint32_t party_id, RotContext* rot_context,
                       const uint32_t thread_num)
    : party_id_(party_id),
      rot_context_(rot_context),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {}

void MuxContext::Mux(uint32_t n, const uint8_t* choice, uint64_t value,
                     uint64_t* output, uint32_t bw) {
  if (n == 0) {
    return;
  }

  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8)) * 8;

  std::vector<std::future<void>> futures(thread_num_);
  auto* thread_pool = rot_context_->GetThreadPool();
  std::vector<uint64_t> value_vct(n, value);
  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n) ? std::min(block_size, n - i * block_size) : 0;

    // Alternate sender/receiver roles across threads
    bool is_sender = (party_id_ == kServerId) ^ (i & 1);

    if (is_sender) {
      uint32_t rot_offset = rot_context_->ReserveSenderRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &MuxContext::RotSender, this, rot_context_->GetIOArray()[i],
          cur_block_size, choice + i * block_size,
          value_vct.data() + i * block_size, output + i * block_size, bw,
          rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &MuxContext::RotReceiver, this, rot_context_->GetIOArray()[i],
          cur_block_size, choice + i * block_size, output + i * block_size, bw,
          rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void MuxContext::Mux(uint32_t n, const uint8_t* choice, const uint64_t* x,
                     uint64_t* output, uint32_t bw) {
  if (n == 0) {
    return;
  }

  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8)) * 8;

  std::vector<std::future<void>> futures(thread_num_);
  auto* thread_pool = rot_context_->GetThreadPool();

  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n ? min(block_size, n - i * block_size) : 0);
    uint32_t send_offset = rot_context_->ReserveSenderRot(cur_block_size);
    uint32_t recv_offset = rot_context_->ReserveReceiverRot(cur_block_size);
    futures[i] = thread_pool->Submit(
        &MuxContext::MuxProtocol, this, rot_context_->GetIOArray()[i],
        (i & 1 ? 3 - party_id_ : party_id_), cur_block_size,
        choice + i * block_size, x + i * block_size, output + i * block_size,
        bw, rot_context_->GetSenderMsg0() + send_offset,
        rot_context_->GetSenderMsg1() + send_offset,
        rot_context_->GetRecvChoice() + recv_offset,
        rot_context_->GetRecvMsg() + recv_offset);
  }
  for (auto& f : futures) f.wait();
}

void MuxContext::Mux(uint32_t n, const uint8_t* choice, const uint64_t* x,
                     const uint64_t* y, uint64_t* output, uint32_t bw) {
  if (n == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  // Compute y - x
  std::vector<uint64_t> y_sub_x(n);
  ElementwiseSubmod(n, y, x, y_sub_x.data(), mod_mask);

  // MUX: c * (y - x)
  std::vector<uint64_t> mux_res(n);
  Mux(n, choice, y_sub_x.data(), mux_res.data(), bw);

  // x + c * (y - x)
  // c = 0 -> x
  // c = 1 -> y
  ElementwiseAddmod(n, x, mux_res.data(), output, mod_mask);
}

void MuxContext::MuxProtocol(link::LinkIO* io, uint32_t cur_party_id,
                             uint32_t len, const uint8_t* choice,
                             const uint64_t* shared_value, uint64_t* output,
                             uint32_t bw, const emp::block* send_msg0,
                             const emp::block* send_msg1,
                             const uint8_t* recv_choice,
                             const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  // Compute masked choice
  std::vector<uint8_t> send_masked_choice(len);
  for (uint32_t i = 0; i < len; i++) {
    send_masked_choice[i] = recv_choice[i] ^ choice[i];
  }

  // Exchange masked choices using raw bool arrays
  auto bool_send_masked_choice = std::make_unique<bool[]>(len);
  auto bool_recv_masked_choice = std::make_unique<bool[]>(len);
  for (uint32_t i = 0; i < len; i++) {
    bool_send_masked_choice[i] = (send_masked_choice[i] != 0);
  }

  if (cur_party_id == kServerId) {
    SendBoolVector(bool_send_masked_choice.get(), len, io);
    RecvBoolVector(bool_recv_masked_choice.get(), len, io);
  } else {
    RecvBoolVector(bool_recv_masked_choice.get(), len, io);
    SendBoolVector(bool_send_masked_choice.get(), len, io);
    io->flush();
  }

  // Extract messages from ROT
  std::vector<uint64_t> msg0(len);
  std::vector<uint64_t> msg1(len);
  for (uint32_t i = 0; i < len; i++) {
    msg0[i] = send_msg0[i][0] & mod_mask;
    msg1[i] = send_msg1[i][0] & mod_mask;
    if (bool_recv_masked_choice[i]) {
      std::swap(msg0[i], msg1[i]);
    }
  }

  std::vector<uint64_t> msg(len);
  for (uint32_t i = 0; i < len; i++) {
    msg[i] = recv_msg[i][0] & mod_mask;
  }

  // Compute send delta
  std::vector<uint64_t> send_delta(len);
  for (uint32_t i = 0; i < len; i++) {
    if (choice[i] == 0) {
      send_delta[i] = (msg0[i] - msg1[i] + shared_value[i]) & mod_mask;
    } else {
      send_delta[i] = (msg0[i] - msg1[i] - shared_value[i]) & mod_mask;
    }
  }

  // Exchange deltas
  std::vector<uint64_t> recv_delta(len);
  if (cur_party_id == kServerId) {
    SendU64Vector(send_delta.data(), len, bw, io);
    RecvU64Vector(recv_delta.data(), len, bw, io);
  } else {
    RecvU64Vector(recv_delta.data(), len, bw, io);
    SendU64Vector(send_delta.data(), len, bw, io);
    io->flush();
  }

  // Compute output
  for (uint32_t i = 0; i < len; i++) {
    if (choice[i] == 0) {
      output[i] = (msg[i] - msg0[i]) & mod_mask;
    } else {
      output[i] =
          (recv_delta[i] + msg[i] - msg0[i] + shared_value[i]) & mod_mask;
    }
  }
}

void MuxContext::RotSender(link::LinkIO* io, uint32_t len,
                           const uint8_t* choice, const uint64_t* value,
                           uint64_t* output, uint32_t bw,
                           const emp::block* send_msg0,
                           const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  // Receive OT choice adjustment from receiver
  auto bool_ot_choice_adjust = std::make_unique<bool[]>(len);
  RecvBoolVector(bool_ot_choice_adjust.get(), len, io);

  // Extract messages from ROT
  std::vector<uint64_t> msg0(len);
  std::vector<uint64_t> msg1(len);
  for (uint32_t i = 0; i < len; i++) {
    msg0[i] = send_msg0[i][0] & mod_mask;
    msg1[i] = send_msg1[i][0] & mod_mask;
    if (bool_ot_choice_adjust[i]) {
      std::swap(msg0[i], msg1[i]);
    }
  }

  // Compute delta
  std::vector<uint64_t> delta_vct(len);
  for (uint32_t i = 0; i < len; i++) {
    if (choice[i] == 0) {
      delta_vct[i] = (msg0[i] - msg1[i] + value[i]) & mod_mask;
    } else {
      delta_vct[i] = (msg0[i] - msg1[i] - value[i]) & mod_mask;
    }
  }

  SendU64Vector(delta_vct.data(), len, bw, io);
  io->flush();

  // Compute output
  for (uint32_t i = 0; i < len; i++) {
    if (choice[i] == 0) {
      output[i] = (static_cast<uint64_t>(0) - msg0[i]) & mod_mask;
    } else {
      output[i] = (static_cast<uint64_t>(0) - msg0[i] + value[i]) & mod_mask;
    }
  }
}

void MuxContext::RotReceiver(link::LinkIO* io, uint32_t len,
                             const uint8_t* choice, uint64_t* output,
                             uint32_t bw, const uint8_t* recv_choice,
                             const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  // Compute and send OT choice
  auto bool_ot_choice = std::make_unique<bool[]>(len);
  for (uint32_t i = 0; i < len; i++) {
    bool_ot_choice[i] = ((choice[i] ^ recv_choice[i]) != 0);
  }

  SendBoolVector(bool_ot_choice.get(), len, io);
  io->flush();

  // Extract message from ROT
  std::vector<uint64_t> msg(len);
  for (uint32_t i = 0; i < len; i++) {
    msg[i] = recv_msg[i][0] & mod_mask;
  }

  // Receive delta from sender
  std::vector<uint64_t> delta_vct(len);
  RecvU64Vector(delta_vct.data(), len, bw, io);

  // Compute output
  for (uint32_t i = 0; i < len; i++) {
    if (choice[i] == 0) {
      output[i] = msg[i];
    } else {
      output[i] = (delta_vct[i] + msg[i]) & mod_mask;
    }
  }
}

}  // namespace capricorn::operator_
