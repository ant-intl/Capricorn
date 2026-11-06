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

#include "src/common/mpc/ring_extension.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/io_helper.h"
#include "src/common/mpc/rot_context.h"

namespace capricorn::operator_ {

RingExtensionContext::RingExtensionContext(const uint32_t party_id,
                                           RotContext* rot_context,
                                           const uint32_t thread_num)
    : party_id_(party_id),
      rot_context_(rot_context),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {}

void RingExtensionContext::RingExtension(uint32_t n, const uint64_t* input_x,
                                         uint64_t* output_y, uint32_t in_bw,
                                         uint32_t out_bw) {
  // If output bit width is not larger, just mask and copy
  if (in_bw == out_bw) {
    std::memcpy(output_y, input_x, n * sizeof(uint64_t));
    return;
  } else if (in_bw > out_bw) {
    uint64_t out_mask = (out_bw == 64) ? -1ULL : ((1ULL << out_bw) - 1);
    ElementwiseAnd(n, input_x, out_mask, output_y);
    return;
  }

  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8)) * 8;

  auto* thread_pool = rot_context_->GetThreadPool();
  std::vector<std::future<void>> futures(thread_num_);

  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n) ? std::min(block_size, n - i * block_size) : 0;

    bool is_sender = (party_id_ == kServerId) ^ (i & 1);

    if (is_sender) {
      uint32_t rot_offset = rot_context_->ReserveSenderRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &RingExtensionContext::RingExtensionSender, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, output_y + i * block_size, in_bw, out_bw,
          rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &RingExtensionContext::RingExtensionReceiver, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, output_y + i * block_size, in_bw, out_bw,
          rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void RingExtensionContext::RingExtensionWithOffset(
    uint32_t n, const uint64_t* input_x, uint64_t* output_y, uint32_t in_bw,
    uint32_t out_bw, uint64_t offset) {
  if (n == 0 || in_bw >= out_bw) {
    RingExtension(n, input_x, output_y, in_bw, out_bw);
    return;
  }

  const uint64_t input_mask = (1ULL << in_bw) - 1;
  const uint64_t output_mask = (out_bw == 64) ? -1ULL : ((1ULL << out_bw) - 1);
  offset &= input_mask;

  std::vector<uint64_t> adjusted_input(input_x, input_x + n);
  if (party_id_ == kServerId) {
    ElementwiseAddmodInplace(n, adjusted_input.data(), offset, input_mask);
  }

  RingExtension(n, adjusted_input.data(), output_y, in_bw, out_bw);

  if (party_id_ == kServerId) {
    ElementwiseSubmodInplace(n, output_y, offset, output_mask);
  }
}

void RingExtensionContext::RingExtensionSender(link::LinkIO* io, uint32_t len,
                                               const uint64_t* x, uint64_t* res,
                                               uint32_t in_bw, uint32_t out_bw,
                                               const emp::block* send_msg0,
                                               const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  uint64_t a_msb_mask = 1ULL << (in_bw - 1);
  uint64_t b_mod_mask = (out_bw == 64) ? -1ULL : ((1ULL << out_bw) - 1);

  if (in_bw >= out_bw) {
    std::transform(x, x + len, res,
                   [b_mod_mask](uint64_t p) { return p & b_mod_mask; });
    return;
  }

  int32_t shift_bw = static_cast<int32_t>(out_bw) - static_cast<int32_t>(in_bw);
  uint64_t shift_mod_mask = (shift_bw == 64) ? -1ULL : ((1ULL << shift_bw) - 1);

  // Extract MSB for each element
  std::vector<uint8_t> msb(len);
  for (uint32_t i = 0; i < len; i++) {
    // msb[i] = static_cast<uint8_t>(x[i] & a_msb_mask);
    msb[i] = (bool)(x[i] & a_msb_mask);
  }

  // Receive OT choice adjustment
  auto ot_choice_adjust = std::make_unique<bool[]>(len);
  RecvBoolVector(ot_choice_adjust.get(), len, io);

  std::vector<uint64_t> msg0(len);
  std::vector<uint64_t> msg1(len);
  for (uint32_t i = 0; i < len; i++) {
    msg0[i] = send_msg0[i][0] & shift_mod_mask;
    msg1[i] = send_msg1[i][0] & shift_mod_mask;
    if (ot_choice_adjust[i]) {
      std::swap(msg0[i], msg1[i]);
    }
  }

  // Compute delta
  std::vector<uint64_t> delta_vct(len);
  for (uint32_t i = 0; i < len; i++) {
    if (msb[i]) {
      delta_vct[i] = msg0[i] - msg1[i];
    } else {
      delta_vct[i] = msg0[i] - msg1[i] + 1;
    }
  }

  ElementwiseAndInplace(len, delta_vct.data(), shift_mod_mask);
  SendU64Vector(delta_vct.data(), len, static_cast<uint32_t>(shift_bw), io);
  io->flush();

  // Compute result
  std::vector<uint64_t> msb_or_res(len);
  for (uint32_t i = 0; i < len; i++) {
    if (msb[i]) {
      msb_or_res[i] = static_cast<uint64_t>(0) - msg0[i] + 1;
    } else {
      msb_or_res[i] = static_cast<uint64_t>(0) - msg0[i];
    }
  }

  ElementwiseAndInplace(len, msb_or_res.data(), shift_mod_mask);

  for (uint32_t i = 0; i < len; i++) {
    res[i] = (x[i] - (msb_or_res[i] << in_bw)) & b_mod_mask;
  }
}

void RingExtensionContext::RingExtensionReceiver(link::LinkIO* io, uint32_t len,
                                                 const uint64_t* x,
                                                 uint64_t* res, uint32_t in_bw,
                                                 uint32_t out_bw,
                                                 const uint8_t* recv_choice,
                                                 const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint64_t a_msb_mask = 1ULL << (in_bw - 1);
  uint64_t b_mod_mask = (out_bw == 64) ? -1ULL : ((1ULL << out_bw) - 1);

  if (in_bw >= out_bw) {
    std::transform(x, x + len, res,
                   [b_mod_mask](uint64_t p) { return p & b_mod_mask; });
    return;
  }

  int32_t shift_bw = static_cast<int32_t>(out_bw) - static_cast<int32_t>(in_bw);
  uint64_t shift_mod_mask = (shift_bw >= 64) ? -1ULL : ((1ULL << shift_bw) - 1);

  // Extract MSB for each element
  std::vector<uint8_t> msb(len);
  for (uint32_t i = 0; i < len; i++) {
    msb[i] = (bool)(x[i] & a_msb_mask);
  }

  // Compute and send OT choice
  auto ot_choice = std::make_unique<bool[]>(len);
  for (uint32_t i = 0; i < len; i++) {
    ot_choice[i] = recv_choice[i] ^ msb[i];
  }

  SendBoolVector(ot_choice.get(), len, io);

  // Extract message from ROT
  std::vector<uint64_t> msg(len);
  for (uint32_t i = 0; i < len; i++) {
    msg[i] = recv_msg[i][0] & shift_mod_mask;
  }

  // Receive delta from sender
  std::vector<uint64_t> delta_vct(len);
  RecvU64Vector(delta_vct.data(), len, static_cast<uint32_t>(shift_bw), io);

  // Compute result
  std::vector<uint64_t> msb_or_res(len);
  for (uint32_t i = 0; i < len; i++) {
    if (msb[i]) {
      msb_or_res[i] = delta_vct[i] + msg[i];
    } else {
      msb_or_res[i] = msg[i];
    }
  }

  ElementwiseAndInplace(len, msb_or_res.data(), shift_mod_mask);
  for (uint32_t i = 0; i < len; i++) {
    res[i] = (x[i] - (msb_or_res[i] << in_bw)) & b_mod_mask;
  }
}

}  // namespace capricorn::operator_
