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

#include "src/common/mpc/and_protocol.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/io_helper.h"
#include "src/common/mpc/rot_context.h"

namespace capricorn::operator_ {

AndContext::AndContext(const uint32_t party_id, RotContext* rot_context,
                       const uint32_t thread_num)
    : party_id_(party_id),
      rot_context_(rot_context),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {}

void AndContext::And(uint32_t n, const uint8_t* input_x, uint8_t* output_z) {
  if (n == 0) {
    return;
  }

  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8)) * 8;

  std::vector<std::future<void>> futures(thread_num_);
  const auto& io_array = rot_context_->GetIOArray();
  auto* thread_pool = rot_context_->GetThreadPool();

  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n) ? std::min(block_size, n - i * block_size) : 0;

    // Alternate sender/receiver roles across threads
    bool is_sender = (party_id_ == kServerId) ^ (i & 1);

    if (is_sender) {
      uint32_t rot_offset = rot_context_->ReserveSenderRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &AndContext::AndSender, this, io_array[i], cur_block_size,
          input_x + i * block_size, output_z + i * block_size,
          rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &AndContext::AndReceiver, this, io_array[i], cur_block_size,
          input_x + i * block_size, output_z + i * block_size,
          rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& d : futures) {
    d.wait();
  }
}

void AndContext::And(uint32_t n, const uint8_t* input_x, const uint8_t* input_y,
                     uint8_t* output_z) {
  if (n == 0) {
    return;
  }

  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8)) * 8;

  std::vector<std::future<void>> futures(thread_num_);
  const auto& io_array = rot_context_->GetIOArray();
  auto* thread_pool = rot_context_->GetThreadPool();

  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n) ? std::min(block_size, n - i * block_size) : 0;

    // For two-input AND, both parties use both send and receive ROT
    uint32_t send_offset = rot_context_->ReserveSenderRot(cur_block_size);
    uint32_t recv_offset = rot_context_->ReserveReceiverRot(cur_block_size);

    // Determine current party id for this thread (alternating)
    uint32_t cur_party_id = ((i & 1) != 0) ? (3 - party_id_) : party_id_;

    futures[i] = thread_pool->Submit(
        &AndContext::AndProtocol, this, io_array[i], cur_party_id,
        cur_block_size, input_x + i * block_size, input_y + i * block_size,
        output_z + i * block_size, rot_context_->GetSenderMsg0() + send_offset,
        rot_context_->GetSenderMsg1() + send_offset,
        rot_context_->GetRecvChoice() + recv_offset,
        rot_context_->GetRecvMsg() + recv_offset);
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void AndContext::Or(uint32_t n, const uint8_t* input_x, uint8_t* output_z) {
  std::vector<uint8_t> and_input_x(n);
  std::vector<uint8_t> and_output_z(n);

  // De Morgan's law: x OR x = NOT(NOT x AND NOT x)
  // For XOR secret sharing: NOT x = (NOT server_x) XOR (NOT client_x)
  // So BOTH parties need to flip their shares
  ElementwiseXor(n, input_x, static_cast<uint8_t>(1), and_input_x.data());

  And(n, and_input_x.data(), and_output_z.data());

  // For the result: NOT z = (NOT server_z) XOR client_z
  // Only server flips its share of the result
  if (party_id_ == kServerId) {
    ElementwiseXor(n, and_output_z.data(), static_cast<uint8_t>(1), output_z);
  } else {
    std::memcpy(output_z, and_output_z.data(), n);
  }
}

void AndContext::Or(uint32_t n, const uint8_t* input_x, const uint8_t* input_y,
                    uint8_t* output_z) {
  std::vector<uint8_t> and_input_x(n);
  std::vector<uint8_t> and_input_y(n);

  // De Morgan's law: x OR y = NOT(NOT x AND NOT y)
  if (party_id_ == kServerId) {
    ElementwiseXor(n, input_x, static_cast<uint8_t>(1), and_input_x.data());
    ElementwiseXor(n, input_y, static_cast<uint8_t>(1), and_input_y.data());
  } else {
    std::memcpy(and_input_x.data(), input_x, n);
    std::memcpy(and_input_y.data(), input_y, n);
  }

  And(n, and_input_x.data(), and_input_y.data(), output_z);

  if (party_id_ == kServerId) {
    ElementwiseXorInplace(n, output_z, static_cast<uint8_t>(1));
  }
}

void AndContext::AndProtocol(link::LinkIO* io, uint32_t cur_party_id,
                             uint32_t len, const uint8_t* input_x,
                             const uint8_t* input_y, uint8_t* output_z,
                             const emp::block* send_msg0,
                             const emp::block* send_msg1,
                             const uint8_t* recv_choice,
                             const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  // Generate multiplication triple from ROT
  std::vector<uint8_t> mt_x(len);
  std::vector<uint8_t> mt_y(len);
  std::vector<uint8_t> mt_z(len);
  for (uint32_t i = 0; i < len; i++) {
    mt_x[i] = recv_choice[i];
    mt_y[i] = emp::getLSB(send_msg0[i]) ^ emp::getLSB(send_msg1[i]);
    mt_z[i] = (mt_x[i] & mt_y[i]) ^ emp::getLSB(recv_msg[i]) ^
              emp::getLSB(send_msg0[i]);
  }

  // Compute local masks
  std::vector<uint8_t> d(len);
  std::vector<uint8_t> e(len);
  ElementwiseXor(len, input_x, mt_x.data(), d.data());
  ElementwiseXor(len, input_y, mt_y.data(), e.data());

  // Exchange masks using raw bool arrays
  // Allocate memory for bool arrays (use unique_ptr for exception safety)
  auto bool_d = std::make_unique<bool[]>(len);
  auto bool_e = std::make_unique<bool[]>(len);
  auto bool_d0 = std::make_unique<bool[]>(len);
  auto bool_e0 = std::make_unique<bool[]>(len);

  for (uint32_t i = 0; i < len; i++) {
    bool_d[i] = (d[i] != 0);
    bool_e[i] = (e[i] != 0);
  }

  if (cur_party_id == kServerId) {
    SendBoolVector(bool_d.get(), len, io);
    SendBoolVector(bool_e.get(), len, io);
    RecvBoolVector(bool_d0.get(), len, io);
    RecvBoolVector(bool_e0.get(), len, io);
  } else {
    RecvBoolVector(bool_d0.get(), len, io);
    RecvBoolVector(bool_e0.get(), len, io);
    SendBoolVector(bool_d.get(), len, io);
    SendBoolVector(bool_e.get(), len, io);
    io->flush();
  }

  // Convert bool back to uint8_t and XOR with local masks
  std::vector<uint8_t> d0(len);
  std::vector<uint8_t> e0(len);
  for (uint32_t i = 0; i < len; i++) {
    d0[i] = static_cast<uint8_t>(bool_d0[i]) ^ d[i];
    e0[i] = static_cast<uint8_t>(bool_e0[i]) ^ e[i];
  }

  // Compute final result
  if (cur_party_id == kClientId) {
    for (uint32_t i = 0; i < len; i++) {
      output_z[i] =
          (d0[i] & e0[i]) ^ (d0[i] & mt_y[i]) ^ (e0[i] & mt_x[i]) ^ mt_z[i];
    }
  } else {
    for (uint32_t i = 0; i < len; i++) {
      output_z[i] = (d0[i] & mt_y[i]) ^ (e0[i] & mt_x[i]) ^ mt_z[i];
    }
  }
}

void AndContext::AndSender(link::LinkIO* io, uint32_t len,
                           const uint8_t* input_x, uint8_t* output_z,
                           const emp::block* send_msg0,
                           const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  // Receive OT choice adjustment from receiver
  auto bool_ot_choice_adjust = std::make_unique<bool[]>(len);
  RecvBoolVector(bool_ot_choice_adjust.get(), len, io);

  // Extract LSB from ROT messages
  std::vector<uint8_t> msg0(len);
  std::vector<uint8_t> msg1(len);
  for (uint32_t i = 0; i < len; i++) {
    msg0[i] = emp::getLSB(send_msg0[i]);
    msg1[i] = emp::getLSB(send_msg1[i]);
    if (bool_ot_choice_adjust[i]) {
      std::swap(msg0[i], msg1[i]);
    }
  }

  // Compute delta
  auto bool_delta = std::make_unique<bool[]>(len);
  for (uint32_t i = 0; i < len; i++) {
    bool_delta[i] = (msg0[i] ^ msg1[i] ^ input_x[i]);
  }

  SendBoolVector(bool_delta.get(), len, io);
  io->flush();

  // Output is msg0
  std::memcpy(output_z, msg0.data(), len);
}

void AndContext::AndReceiver(link::LinkIO* io, uint32_t len,
                             const uint8_t* input_x, uint8_t* output_z,
                             const uint8_t* recv_choice,
                             const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  // Compute and send OT choice
  vector<uint8_t> bool_ot_choice(len);
  ElementwiseXor(len, input_x, recv_choice, bool_ot_choice.data());
  SendBoolVector((bool*)bool_ot_choice.data(), len, io);
  io->flush();

  // Extract LSB from received ROT message
  std::vector<uint8_t> msg(len);
  for (uint32_t i = 0; i < len; i++) {
    msg[i] = emp::getLSB(recv_msg[i]);
  }

  // Receive delta from sender
  vector<uint8_t> bool_delta(len);
  RecvBoolVector((bool*)bool_delta.data(), len, io);

  ElementwiseAndInplace(len, bool_delta.data(), input_x);
  ElementwiseXor(len, bool_delta.data(), msg.data(), output_z);
}

}  // namespace capricorn::operator_
