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

#include "src/common/mpc/lookup_table.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/io_helper.h"

namespace capricorn::operator_ {

LookupTableContext::LookupTableContext(uint32_t party_id,
                                       RotContext* rot_context,
                                       const uint32_t thread_num)
    : party_id_(party_id),
      rot_context_(rot_context),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {
  common_aes_key_ = rot_context_->GetCommonAesKey();
}

void LookupTableContext::LookupTable(uint32_t n, const uint8_t* input_x,
                                     uint64_t* res, uint32_t input_bw,
                                     uint32_t output_bw,
                                     const uint64_t* public_table) {
  if (n == 0) {
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
      uint32_t rot_offset = rot_context_->ReserveSenderRot(
          static_cast<uint64_t>(cur_block_size) * input_bw);
      futures[i] = thread_pool->Submit(
          &LookupTableContext::LookupTableSender1, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, res + i * block_size, input_bw, output_bw,
          public_table, rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(
          static_cast<uint64_t>(cur_block_size) * input_bw);
      futures[i] = thread_pool->Submit(
          &LookupTableContext::LookupTableReceiver1, this,
          rot_context_->GetIOArray()[i], cur_block_size,
          input_x + i * block_size, res + i * block_size, input_bw, output_bw,
          public_table, rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void LookupTableContext::LookupTable2(uint32_t n, const uint8_t* input_x,
                                      uint64_t* res_x, uint64_t* res_y,
                                      uint32_t input_bw, uint32_t output_bw_x,
                                      uint32_t output_bw_y,
                                      const uint64_t* public_table_x,
                                      const uint64_t* public_table_y) {
  if (n == 0) {
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
      uint32_t rot_offset = rot_context_->ReserveSenderRot(
          static_cast<uint64_t>(cur_block_size) * input_bw);
      futures[i] =
          thread_pool->Submit(&LookupTableContext::LookupTableSender2, this,
                              rot_context_->GetIOArray()[i], cur_block_size,
                              input_x + i * block_size, res_x + i * block_size,
                              res_y + i * block_size, input_bw, output_bw_x,
                              output_bw_y, public_table_x, public_table_y,
                              rot_context_->GetSenderMsg0() + rot_offset,
                              rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(
          static_cast<uint64_t>(cur_block_size) * input_bw);
      futures[i] =
          thread_pool->Submit(&LookupTableContext::LookupTableReceiver2, this,
                              rot_context_->GetIOArray()[i], cur_block_size,
                              input_x + i * block_size, res_x + i * block_size,
                              res_y + i * block_size, input_bw, output_bw_x,
                              output_bw_y, public_table_x, public_table_y,
                              rot_context_->GetRecvChoice() + rot_offset,
                              rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void LookupTableContext::LookupTableSender1(
    link::LinkIO* io, uint32_t len, const uint8_t* x, uint64_t* res,
    uint32_t input_bw, uint32_t output_bw, const uint64_t* public_table,
    const emp::block* send_msg0, const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  uint32_t table_size = 1 << input_bw;
  uint8_t index_mod_mask = static_cast<uint8_t>(table_size - 1);
  uint64_t mod_mask = (output_bw == 64) ? -1ULL : ((1ULL << output_bw) - 1);

  emp::AES_KEY common_aes;
  emp::AES_set_encrypt_key(common_aes_key_, &common_aes);

  // Generate random result shares
  emp::PRG prg;
  prg.random_data(res, len * sizeof(uint64_t));
  ElementwiseAndInplace(len, res, mod_mask);

  // Get input indices
  std::vector<uint8_t> input_id(len);
  ElementwiseAnd(len, x, index_mod_mask, input_id.data());

  // Prepare masked table
  std::vector<uint64_t> masked_value(len * table_size, 0);

  for (uint32_t i = 0; i < len; i++) {
    // Rotate public table by input_id[i]
    std::memcpy(masked_value.data() + i * table_size,
                public_table + input_id[i],
                (table_size - input_id[i]) * sizeof(uint64_t));
    std::memcpy(
        masked_value.data() + i * table_size + (table_size - input_id[i]),
        public_table, input_id[i] * sizeof(uint64_t));
    // Subtract result share
    ElementwiseSubmodInplace(table_size, masked_value.data() + i * table_size,
                             res[i], mod_mask);
  }

  // Generate 1-out-of-N OT from 1-out-of-2 OT
  std::vector<emp::block> sender_mask(len * table_size, emp::zero_block);
  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < input_bw; j++) {
      if (j != 0) {
        for (uint32_t k = 0; k < (1U << j); k++) {
          sender_mask[i * table_size + (1U << j) + k] =
              sender_mask[i * table_size + k];
        }
      }
      for (uint32_t k = 0; k < (1U << j); k++) {
        sender_mask[i * table_size + k] ^= send_msg0[i * input_bw + j];
      }
      for (uint32_t k = 0; k < (1U << j); k++) {
        sender_mask[i * table_size + (1U << j) + k] ^=
            send_msg1[i * input_bw + j];
      }
    }
    for (uint32_t j = 0; j < table_size; j++) {
      CcrFunctionHInplace(sender_mask[i * table_size + j], common_aes);
    }
  }

  // Receive masked index from receiver
  std::vector<uint8_t> masked_index(len);
  RecvUint8Vector(masked_index.data(), len, input_bw, io);

  // XOR masked table with sender mask
  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < (table_size - masked_index[i]); j++) {
      masked_value[i * table_size + j] ^=
          sender_mask[i * table_size + masked_index[i] + j][0];
    }
    for (uint32_t j = 0; j < masked_index[i]; j++) {
      masked_value[i * table_size + (table_size - masked_index[i]) + j] ^=
          sender_mask[i * table_size + j][0];
    }
  }

  SendU64Vector(masked_value.data(), len * table_size, output_bw, io);
  io->flush();
}

void LookupTableContext::LookupTableReceiver1(
    link::LinkIO* io, uint32_t len, const uint8_t* x, uint64_t* res,
    uint32_t input_bw, uint32_t output_bw, const uint64_t* public_table,
    const uint8_t* recv_choice, const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint32_t table_size = 1 << input_bw;
  uint8_t index_mod_mask = (input_bw == 8)
                               ? static_cast<uint8_t>(-1)
                               : static_cast<uint8_t>((1U << input_bw) - 1);
  uint64_t mod_mask = (output_bw == 64) ? -1ULL : ((1ULL << output_bw) - 1);

  emp::AES_KEY common_aes;
  emp::AES_set_encrypt_key(common_aes_key_, &common_aes);

  std::vector<uint8_t> input_id(len);
  ElementwiseAnd(len, x, index_mod_mask, input_id.data());

  std::vector<uint8_t> masked_index(len, 0);
  std::vector<emp::block> receiver_mask(len, emp::zero_block);

  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < input_bw; j++) {
      receiver_mask[i] ^= recv_msg[i * input_bw + j];
      masked_index[i] =
          static_cast<uint8_t>((masked_index[i] << 1) |
                               recv_choice[i * input_bw + (input_bw - j - 1)]);
    }
    CcrFunctionHInplace(receiver_mask[i], common_aes);
  }

  ElementwiseSubmodInplace(len, masked_index.data(), input_id.data(),
                           index_mod_mask);
  SendUint8Vector(masked_index.data(), len, input_bw, io);
  io->flush();

  std::vector<uint64_t> masked_value(len * table_size);
  RecvU64Vector(masked_value.data(), len * table_size, output_bw, io);

  for (uint32_t i = 0; i < len; i++) {
    res[i] = masked_value[i * table_size + input_id[i]] ^ receiver_mask[i][0];
  }
  ElementwiseAndInplace(len, res, mod_mask);
}

void LookupTableContext::LookupTableSender2(
    link::LinkIO* io, uint32_t len, const uint8_t* x, uint64_t* res_x,
    uint64_t* res_y, uint32_t input_bw, uint32_t output_bw_x,
    uint32_t output_bw_y, const uint64_t* public_table_x,
    const uint64_t* public_table_y, const emp::block* send_msg0,
    const emp::block* send_msg1) {
  if (len == 0) {
    return;
  }

  uint32_t table_size = 1 << input_bw;
  uint8_t index_mod_mask = static_cast<uint8_t>(table_size - 1);
  uint64_t x_mod_mask =
      (output_bw_x == 64) ? -1ULL : ((1ULL << output_bw_x) - 1);
  uint64_t y_mod_mask =
      (output_bw_y == 64) ? -1ULL : ((1ULL << output_bw_y) - 1);

  emp::AES_KEY common_aes;
  emp::AES_set_encrypt_key(common_aes_key_, &common_aes);

  emp::PRG prg;
  prg.random_data(res_x, len * sizeof(uint64_t));
  ElementwiseAndInplace(len, res_x, x_mod_mask);

  prg.random_data(res_y, len * sizeof(uint64_t));
  ElementwiseAndInplace(len, res_y, y_mod_mask);

  std::vector<uint8_t> input_id(len);
  ElementwiseAnd(len, x, index_mod_mask, input_id.data());

  std::vector<uint64_t> masked_value_x(len * table_size, 0);
  std::vector<uint64_t> masked_value_y(len * table_size, 0);

  for (uint32_t i = 0; i < len; i++) {
    std::memcpy(masked_value_x.data() + i * table_size,
                public_table_x + input_id[i],
                (table_size - input_id[i]) * sizeof(uint64_t));
    std::memcpy(
        masked_value_x.data() + i * table_size + (table_size - input_id[i]),
        public_table_x, input_id[i] * sizeof(uint64_t));
    ElementwiseSubmodInplace(table_size, masked_value_x.data() + i * table_size,
                             res_x[i], x_mod_mask);

    std::memcpy(masked_value_y.data() + i * table_size,
                public_table_y + input_id[i],
                (table_size - input_id[i]) * sizeof(uint64_t));
    std::memcpy(
        masked_value_y.data() + i * table_size + (table_size - input_id[i]),
        public_table_y, input_id[i] * sizeof(uint64_t));
    ElementwiseSubmodInplace(table_size, masked_value_y.data() + i * table_size,
                             res_y[i], y_mod_mask);
  }

  std::vector<emp::block> sender_mask(len * table_size, emp::zero_block);
  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < input_bw; j++) {
      if (j != 0) {
        for (uint32_t k = 0; k < (1U << j); k++) {
          sender_mask[i * table_size + (1U << j) + k] =
              sender_mask[i * table_size + k];
        }
      }
      for (uint32_t k = 0; k < (1U << j); k++) {
        sender_mask[i * table_size + k] ^= send_msg0[i * input_bw + j];
      }
      for (uint32_t k = 0; k < (1U << j); k++) {
        sender_mask[i * table_size + (1U << j) + k] ^=
            send_msg1[i * input_bw + j];
      }
    }
    for (uint32_t j = 0; j < table_size; j++) {
      CcrFunctionHInplace(sender_mask[i * table_size + j], common_aes);
    }
  }

  std::vector<uint8_t> masked_index(len);
  RecvUint8Vector(masked_index.data(), len, input_bw, io);

  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < (table_size - masked_index[i]); j++) {
      masked_value_x[i * table_size + j] ^=
          sender_mask[i * table_size + masked_index[i] + j][0];
      masked_value_y[i * table_size + j] ^=
          sender_mask[i * table_size + masked_index[i] + j][1];
    }
    for (uint32_t j = 0; j < masked_index[i]; j++) {
      masked_value_x[i * table_size + (table_size - masked_index[i]) + j] ^=
          sender_mask[i * table_size + j][0];
      masked_value_y[i * table_size + (table_size - masked_index[i]) + j] ^=
          sender_mask[i * table_size + j][1];
    }
  }

  if (output_bw_x == output_bw_y) {
    std::vector<uint64_t> masked_value_vct(2 * len * table_size);
    std::memcpy(masked_value_vct.data(), masked_value_x.data(),
                len * table_size * sizeof(uint64_t));
    std::memcpy(masked_value_vct.data() + len * table_size,
                masked_value_y.data(), len * table_size * sizeof(uint64_t));
    SendU64Vector(masked_value_vct.data(), 2 * len * table_size, output_bw_x,
                  io);
  } else {
    SendU64Vector(masked_value_x.data(), len * table_size, output_bw_x, io);
    SendU64Vector(masked_value_y.data(), len * table_size, output_bw_y, io);
  }
  io->flush();
}

void LookupTableContext::LookupTableReceiver2(
    link::LinkIO* io, uint32_t len, const uint8_t* x, uint64_t* res_x,
    uint64_t* res_y, uint32_t input_bw, uint32_t output_bw_x,
    uint32_t output_bw_y, const uint64_t* public_table_x,
    const uint64_t* public_table_y, const uint8_t* recv_choice,
    const emp::block* recv_msg) {
  if (len == 0) {
    return;
  }

  uint32_t table_size = 1 << input_bw;
  uint8_t index_mod_mask = (input_bw == 8)
                               ? static_cast<uint8_t>(-1)
                               : static_cast<uint8_t>((1U << input_bw) - 1);
  uint64_t x_mod_mask =
      (output_bw_x == 64) ? -1ULL : ((1ULL << output_bw_x) - 1);
  uint64_t y_mod_mask =
      (output_bw_y == 64) ? -1ULL : ((1ULL << output_bw_y) - 1);

  emp::AES_KEY common_aes;
  emp::AES_set_encrypt_key(common_aes_key_, &common_aes);

  std::vector<uint8_t> input_id(len);
  ElementwiseAnd(len, x, index_mod_mask, input_id.data());

  std::vector<uint8_t> masked_index(len, 0);
  std::vector<emp::block> receiver_mask(len, emp::zero_block);

  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < input_bw; j++) {
      receiver_mask[i] ^= recv_msg[i * input_bw + j];
      masked_index[i] =
          static_cast<uint8_t>((masked_index[i] << 1) |
                               recv_choice[i * input_bw + (input_bw - j - 1)]);
    }
    CcrFunctionHInplace(receiver_mask[i], common_aes);
  }

  ElementwiseSubmodInplace(len, masked_index.data(), input_id.data(),
                           index_mod_mask);
  SendUint8Vector(masked_index.data(), len, input_bw, io);
  io->flush();

  std::vector<uint64_t> masked_value_vct;
  std::vector<uint64_t> masked_value_x_vec;
  std::vector<uint64_t> masked_value_y_vec;
  uint64_t* masked_value_x;
  uint64_t* masked_value_y;

  if (output_bw_x == output_bw_y) {
    masked_value_vct.resize(2 * len * table_size);
    RecvU64Vector(masked_value_vct.data(), 2 * len * table_size, output_bw_x,
                  io);
    masked_value_x = masked_value_vct.data();
    masked_value_y = masked_value_vct.data() + len * table_size;
  } else {
    masked_value_x_vec.resize(len * table_size);
    masked_value_y_vec.resize(len * table_size);
    RecvU64Vector(masked_value_x_vec.data(), len * table_size, output_bw_x, io);
    RecvU64Vector(masked_value_y_vec.data(), len * table_size, output_bw_y, io);
    masked_value_x = masked_value_x_vec.data();
    masked_value_y = masked_value_y_vec.data();
  }

  for (uint32_t i = 0; i < len; i++) {
    res_x[i] =
        masked_value_x[i * table_size + input_id[i]] ^ receiver_mask[i][0];
    res_y[i] =
        masked_value_y[i * table_size + input_id[i]] ^ receiver_mask[i][1];
  }
  ElementwiseAndInplace(len, res_x, x_mod_mask);
  ElementwiseAndInplace(len, res_y, y_mod_mask);
}

}  // namespace capricorn::operator_
