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

#include "src/common/mpc/truncate.h"

#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/io_helper.h"

namespace capricorn::operator_ {

TruncateContext::TruncateContext(const uint32_t party_id,
                                 RotContext* rot_context,
                                 const uint32_t thread_num)
    : party_id_(party_id),
      rot_context_(rot_context),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {
  b2a_context_ =
      std::make_unique<B2AContext>(party_id, rot_context, thread_num);
  compare_context_ =
      std::make_unique<CompareContext>(party_id, rot_context, thread_num);
}

void TruncateContext::TruncateApproximate(uint32_t n, const uint64_t* input_x,
                                          uint64_t* output_y, uint32_t bw,
                                          uint32_t shift) {
  if (shift == 0) {
    std::memcpy(output_y, input_x, n * sizeof(uint64_t));
    return;
  }

  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8)) * 8;

  auto* thread_pool = rot_context_->GetThreadPool();
  std::vector<std::future<void>> futures(thread_num_);

  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n) ? std::min(block_size, n - i * block_size) : 0;

    // Alternate sender/receiver roles across threads
    bool is_sender = (party_id_ == kServerId) ^ (i & 1);

    if (is_sender) {
      uint32_t rot_offset = rot_context_->ReserveSenderRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &TruncateContext::TruncateApproximateSender, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, output_y + i * block_size, bw, shift,
          rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &TruncateContext::TruncateApproximateReceiver, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, output_y + i * block_size, bw, shift,
          rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void TruncateContext::TruncatePositive(uint32_t n, const uint64_t* input_x,
                                       uint64_t* output_y, uint32_t bw,
                                       uint32_t shift) {
  if (shift == 0) {
    std::memcpy(output_y, input_x, n * sizeof(uint64_t));
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
          &TruncateContext::TruncatePositiveSender, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, output_y + i * block_size, bw, shift,
          rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(cur_block_size);
      futures[i] = thread_pool->Submit(
          &TruncateContext::TruncatePositiveReceiver, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, output_y + i * block_size, bw, shift,
          rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void TruncateContext::TruncateAccurate(uint32_t n, const uint64_t* input_x,
                                       uint64_t* output_y, uint32_t bw,
                                       uint32_t shift) {
  if (shift == 0) {
    std::memcpy(output_y, input_x, n * sizeof(uint64_t));
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : ((1ULL << bw) - 1);
  uint64_t truncated_mask = (1ULL << shift) - 1;

  // First do positive truncation
  TruncatePositive(n, input_x, output_y, bw, shift);

  // Get truncated lower bits
  std::vector<uint64_t> truncated_value(n);
  ElementwiseAnd(n, input_x, truncated_mask, truncated_value.data());

  // Compute wrap on lower bits
  std::vector<uint8_t> wrap_res(n);
  compare_context_->Wrap(n, truncated_value.data(), wrap_res.data(), shift);

  // Convert wrap from boolean to arithmetic
  std::vector<uint64_t> b2a_res(n);
  b2a_context_->B2A(n, wrap_res.data(), b2a_res.data(), bw);

  // Add correction
  ElementwiseAddmodInplace(n, output_y, b2a_res.data(), mod_mask);
}

void TruncateContext::TruncateAndReduce(uint32_t n, const uint64_t* input_x,
                                        uint64_t* output_y, uint32_t bw,
                                        uint32_t shift, uint8_t* wrap_res) {
  if (shift == 0) {
    std::memcpy(output_y, input_x, n * sizeof(uint64_t));
    return;
  }

  uint64_t mask_bw = (bw == 64) ? -1ULL : ((1ULL << bw) - 1);
  uint64_t mask_shift = (shift == 64) ? -1ULL : ((1ULL << shift) - 1);
  uint64_t mask_out =
      ((bw - shift) == 64) ? -1ULL : ((1ULL << (bw - shift)) - 1);

  std::vector<uint64_t> in_x_lower(n);
  std::vector<uint8_t> wrap_res_local(n);
  ElementwiseAnd(n, input_x, mask_shift, in_x_lower.data());

  compare_context_->Wrap(n, in_x_lower.data(), wrap_res_local.data(), shift);

  if (wrap_res != nullptr) {
    std::memcpy(wrap_res, wrap_res_local.data(), n);
  }

  std::vector<uint64_t> b2a_res(n);
  b2a_context_->B2A(n, wrap_res_local.data(), b2a_res.data(), bw - shift);

  for (uint32_t i = 0; i < n; i++) {
    output_y[i] = (((input_x[i] & mask_bw) >> shift) + b2a_res[i]) & mask_out;
  }
}

void TruncateContext::TruncateApproximateSender(link::LinkIO* io, uint32_t len,
                                                const uint64_t* x,
                                                uint64_t* res, uint32_t bw,
                                                uint32_t shift,
                                                const emp::block* send_msg0,
                                                const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : ((1ULL << bw) - 1);
  uint64_t msb_mask = 1ULL << (bw - 1);
  uint64_t big_number = 1ULL << (bw - 2);
  uint64_t wrap_error = 1ULL << (bw - shift);

  std::vector<uint64_t> adjust(len);
  ElementwiseAddmod(len, x, big_number, adjust.data(), mod_mask);

  // Receive OT choice adjustment
  auto ot_choice_adjust = std::make_unique<bool[]>(len);
  RecvBoolVector(ot_choice_adjust.get(), len, io);

  std::vector<uint64_t> msg0(len);
  std::vector<uint64_t> msg1(len);
  for (uint32_t i = 0; i < len; i++) {
    msg0[i] = send_msg0[i][0] & mod_mask;
    msg1[i] = send_msg1[i][0] & mod_mask;
    if (ot_choice_adjust[i]) {
      std::swap(msg0[i], msg1[i]);
    }
  }

  std::vector<uint64_t> delta_vct(len);
  for (uint32_t i = 0; i < len; i++) {
    if (adjust[i] & msb_mask) {
      delta_vct[i] = msg0[i] - msg1[i];
    } else {
      delta_vct[i] = msg0[i] - msg1[i] - wrap_error;
    }
  }

  ElementwiseAndInplace(len, delta_vct.data(), mod_mask);
  SendU64Vector(delta_vct.data(), len, bw, io);
  io->flush();

  for (uint32_t i = 0; i < len; i++) {
    if (adjust[i] & msb_mask) {
      res[i] = ((adjust[i] - big_number) >> shift) - msg0[i] - wrap_error;
    } else {
      res[i] = ((adjust[i] - big_number) >> shift) - msg0[i];
    }
  }
  ElementwiseAndInplace(len, res, mod_mask);
}

void TruncateContext::TruncateApproximateReceiver(
    link::LinkIO* io, uint32_t len, const uint64_t* x, uint64_t* res,
    uint32_t bw, uint32_t shift, const uint8_t* recv_choice,
    const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;
  uint64_t msb_mask = 1ULL << (bw - 1);

  auto ot_choice = std::make_unique<bool[]>(len);
  for (uint32_t i = 0; i < len; i++) {
    ot_choice[i] = recv_choice[i] ^ bool((x[i] & msb_mask));
  }

  SendBoolVector(ot_choice.get(), len, io);
  std::vector<uint64_t> msg(len);
  for (uint32_t i = 0; i < len; i++) {
    msg[i] = recv_msg[i][0] & mod_mask;
  }

  std::vector<uint64_t> delta_vct(len);
  RecvU64Vector(delta_vct.data(), len, bw, io);

  for (uint32_t i = 0; i < len; i++) {
    if (x[i] & msb_mask) {
      res[i] = (x[i] >> shift) + delta_vct[i] + msg[i] + 1;
    } else {
      res[i] = (x[i] >> shift) + msg[i] + 1;
    }
  }

  ElementwiseAndInplace(len, res, mod_mask);
}

void TruncateContext::TruncatePositiveSender(link::LinkIO* io, uint32_t len,
                                             const uint64_t* x, uint64_t* res,
                                             uint32_t bw, uint32_t shift,
                                             const emp::block* send_msg0,
                                             const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : ((1ULL << bw) - 1);
  uint64_t msb_mask = 1ULL << (bw - 1);
  uint64_t wrap_error = 1ULL << (bw - shift);

  auto ot_choice_adjust = std::make_unique<bool[]>(len);
  RecvBoolVector(ot_choice_adjust.get(), len, io);

  std::vector<uint64_t> msg0(len);
  std::vector<uint64_t> msg1(len);
  for (uint32_t i = 0; i < len; i++) {
    msg0[i] = send_msg0[i][0] & mod_mask;
    msg1[i] = send_msg1[i][0] & mod_mask;
    if (ot_choice_adjust[i]) {
      std::swap(msg0[i], msg1[i]);
    }
  }

  std::vector<uint64_t> delta_vct(len);
  for (uint32_t i = 0; i < len; i++) {
    if (x[i] & msb_mask) {
      delta_vct[i] = msg0[i] - msg1[i];
    } else {
      delta_vct[i] = msg0[i] - msg1[i] - wrap_error;
    }
  }

  ElementwiseAndInplace(len, delta_vct.data(), mod_mask);
  SendU64Vector(delta_vct.data(), len, bw, io);
  io->flush();

  for (uint32_t i = 0; i < len; i++) {
    if (x[i] & msb_mask) {
      res[i] = (x[i] >> shift) - msg0[i] - wrap_error;
    } else {
      res[i] = (x[i] >> shift) - msg0[i];
    }
  }

  ElementwiseAndInplace(len, res, mod_mask);
}

void TruncateContext::TruncatePositiveReceiver(link::LinkIO* io, uint32_t len,
                                               const uint64_t* x, uint64_t* res,
                                               uint32_t bw, uint32_t shift,
                                               const uint8_t* recv_choice,
                                               const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : ((1ULL << bw) - 1);
  uint64_t msb_mask = 1ULL << (bw - 1);

  auto ot_choice = std::make_unique<bool[]>(len);
  for (uint32_t i = 0; i < len; i++) {
    ot_choice[i] = recv_choice[i] ^ bool(x[i] & msb_mask);
  }

  SendBoolVector(ot_choice.get(), len, io);
  io->flush();

  std::vector<uint64_t> msg(len);
  for (uint32_t i = 0; i < len; i++) {
    msg[i] = recv_msg[i][0] & mod_mask;
  }

  std::vector<uint64_t> delta_vct(len);
  RecvU64Vector(delta_vct.data(), len, bw, io);

  for (uint32_t i = 0; i < len; i++) {
    if (x[i] & msb_mask) {
      res[i] = (x[i] >> shift) + delta_vct[i] + msg[i];
    } else {
      res[i] = (x[i] >> shift) + msg[i];
    }
  }

  ElementwiseAndInplace(len, res, mod_mask);
}

}  // namespace capricorn::operator_
