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

#include "src/common/mpc/compare.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/io_helper.h"

namespace capricorn::operator_ {

// Maximum sub-block size for comparison
constexpr uint32_t kMaxSubBlockSize = 5;

CompareContext::CompareContext(const uint32_t party_id, RotContext* rot_context,
                               const uint32_t thread_num)
    : party_id_(party_id),
      rot_context_(rot_context),
      thread_num_(
          std::min(thread_num, std::thread::hardware_concurrency() - 1)) {
  and_context_ =
      std::make_unique<AndContext>(party_id, rot_context, thread_num_);
}

void CompareContext::MSB(uint32_t n, const uint64_t* input_x, uint8_t* msb_res,
                         uint32_t bw) {
  if (bw == 1) {
    // Edge case: bw=1 means the value itself is the MSB
    for (uint32_t i = 0; i < n; i++) {
      msb_res[i] = static_cast<uint8_t>(input_x[i] & 1);
    }
    return;
  }

  uint64_t wrap_mask = (1ULL << (bw - 1)) - 1;
  uint64_t msb_mask = 1ULL << (bw - 1);

  // msb(x) = msb(x0) ^ msb(x1) ^ wrap(x0, x1)
  // Extract local MSB
  for (uint32_t i = 0; i < n; i++) {
    msb_res[i] = static_cast<uint8_t>((input_x[i] & msb_mask) != 0);
  }

  // Compute wrap on lower bits
  std::vector<uint64_t> wrap_input_x(n);
  std::vector<uint8_t> wrap_res(n);

  ElementwiseAnd(n, input_x, wrap_mask, wrap_input_x.data());
  Wrap(n, wrap_input_x.data(), wrap_res.data(), bw - 1);

  // XOR with wrap result
  ElementwiseXorInplace(n, msb_res, wrap_res.data());
}

void CompareContext::Wrap(uint32_t n, const uint64_t* input_x,
                          uint8_t* wrap_res, uint32_t bw) {
  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  std::vector<uint64_t> compare_input_x(n);
  if (party_id_ == kServerId) {
    std::memcpy(compare_input_x.data(), input_x, n * sizeof(uint64_t));
  } else {
    // Client XORs with mod_mask to get complement
    ElementwiseXor(n, input_x, mod_mask, compare_input_x.data());
  }

  UnsignedCompare(n, compare_input_x.data(), wrap_res, bw, false);
}

void CompareContext::Compare(uint32_t n, const uint64_t* input_x,
                             uint64_t value, uint8_t* cmp_res, uint32_t bw,
                             bool _less, bool _equal) {
  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  std::vector<uint64_t> msb_input_x(n);
  if (party_id_ == kServerId) {
    if (_less) {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = input_x[i];
      }
    } else {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (mod_mask - input_x[i] + 1) & mod_mask;  // -input_x[i]
      }
    }
  } else {
    if (_less) {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (input_x[i] - value) & mod_mask;
      }
    } else {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (value - input_x[i]) & mod_mask;
      }
    }
    if (_equal) {
      ElementwiseSubInplace(n, msb_input_x.data(), static_cast<uint64_t>(1));
    }
  }

  ElementwiseAndInplace(n, msb_input_x.data(), mod_mask);
  MSB(n, msb_input_x.data(), cmp_res, bw);
}

void CompareContext::Compare(uint32_t n, const uint64_t* input_x,
                             const uint64_t* input_y, uint8_t* cmp_res,
                             uint32_t bw, bool _less, bool _equal) {
  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  std::vector<uint64_t> msb_input_x(n);
  if (party_id_ == kServerId) {
    if (_less) {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (input_x[i] - input_y[i]) & mod_mask;
      }
    } else {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (input_y[i] - input_x[i]) & mod_mask;
      }
    }
  } else {
    if (_less) {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (input_x[i] - input_y[i]) & mod_mask;
      }
    } else {
      for (uint32_t i = 0; i < n; i++) {
        msb_input_x[i] = (input_y[i] - input_x[i]) & mod_mask;
      }
    }
    if (_equal) {
      ElementwiseSubInplace(n, msb_input_x.data(), static_cast<uint64_t>(1));
    }
  }

  ElementwiseAndInplace(n, msb_input_x.data(), mod_mask);
  MSB(n, msb_input_x.data(), cmp_res, bw);
}

void CompareContext::UnsignedCompare(uint32_t n, const uint64_t* input_x,
                                     uint8_t* cmp_res, uint32_t bw,
                                     bool _equal) {
  if (_equal) {
    // TODO: Implement equality comparison
    // For now, we only support greater-than comparison
  }

  if (bw == 1) {
    // Edge case: single bit comparison
    std::vector<uint8_t> and_input(n);
    if (party_id_ == kServerId) {
      for (uint32_t i = 0; i < n; i++) {
        and_input[i] = static_cast<uint8_t>(input_x[i] & 1);
      }
    } else {
      for (uint32_t i = 0; i < n; i++) {
        and_input[i] = static_cast<uint8_t>(1 ^ (input_x[i] & 1));
      }
    }
    and_context_->And(n, and_input.data(), cmp_res);
    return;
  }

  uint64_t mod_mask = (bw == 64) ? -1ULL : (1ULL << bw) - 1;

  // Decompose into sub-blocks
  uint32_t m = static_cast<uint32_t>(std::ceil(1.0 * bw / kMaxSubBlockSize));
  uint32_t cur_sub_block_size = static_cast<uint32_t>(std::ceil(1.0 * bw / m));
  uint64_t cur_sub_block_modmask = (1ULL << cur_sub_block_size) - 1;

  std::vector<uint8_t> sub_block_vct(n * m);
  std::vector<uint8_t> mill_res(n * m);
  std::vector<uint8_t> equal_res(n * m);

  // Decompose input_x into sub-blocks (from MSB to LSB)
  const uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8.0) * 8);
  auto decomposite_prg = [&](uint32_t beg, uint32_t end) -> void {
    if (end <= beg) return;
    for (uint32_t i = beg; i < end; i++) {
      uint64_t cur_x = input_x[i] & mod_mask;
      for (int j = m - 1; j >= 0; j--) {
        sub_block_vct[i * m + j] = cur_x & cur_sub_block_modmask;
        cur_x >>= cur_sub_block_size;
      }
    }
  };

  SPDLOG_INFO("start decomposite_prg");
  auto* thread_pool = rot_context_->GetThreadPool();
  std::vector<std::future<void>> futures(thread_num_);
  for (uint32_t i = 0; i < thread_num_; i++)
    futures[i] = thread_pool->Submit(decomposite_prg, i * block_size,
                                     min(n, (i + 1) * block_size));
  for (auto& f : futures) f.wait();

  SPDLOG_INFO("start SubBlockCompare, {}", n * m);
  // Perform sub-block comparison
  SubBlockCompare(n * m, sub_block_vct.data(), mill_res.data(),
                  equal_res.data(), cur_sub_block_size);
  SPDLOG_INFO("End SubBlockCompare");

  // Aggregate comparison results
  GetCompareResFromSubBlock(n, mill_res.data(), equal_res.data(), cmp_res, m);
}

void CompareContext::GetCompareResFromSubBlock(uint32_t n, const uint8_t* gt,
                                               const uint8_t* eq, uint8_t* res,
                                               uint32_t m) {
  if (m == 1) {
    std::memcpy(res, gt, n);
    return;
  }

  uint32_t round_number = static_cast<uint32_t>(std::ceil(std::log2(m)));
  std::vector<uint8_t> pre_gt(n * m);
  std::vector<uint8_t> pre_eq(n * m);
  std::memcpy(pre_gt.data(), gt, n * m);
  std::memcpy(pre_eq.data(), eq, n * m);

  std::vector<uint8_t> cur_gt(n * m);
  std::vector<uint8_t> cur_eq(n * m);
  uint32_t pre_block_size = m, cur_block_size;

  for (uint32_t round_id = 0; round_id < round_number; round_id++) {
    bool last_round = (round_id + 1 == round_number);
    cur_block_size = (pre_block_size + 1) / 2;

    if (!last_round) {
      std::vector<uint8_t> and_input1(2 * n * (pre_block_size / 2));
      std::vector<uint8_t> and_input2(2 * n * (pre_block_size / 2));
      std::vector<uint8_t> and_out(2 * n * (pre_block_size / 2));

      for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < (pre_block_size / 2); j++) {
          and_input1[i * (pre_block_size / 2) + j] =
              pre_eq[i * pre_block_size + 2 * j];
          and_input2[i * (pre_block_size / 2) + j] =
              pre_gt[i * pre_block_size + 2 * j + 1];
          and_input1[(n + i) * (pre_block_size / 2) + j] =
              pre_eq[i * pre_block_size + 2 * j];
          and_input2[(n + i) * (pre_block_size / 2) + j] =
              pre_eq[i * pre_block_size + 2 * j + 1];
        }
      }

      and_context_->And(2 * n * (pre_block_size / 2), and_input1.data(),
                        and_input2.data(), and_out.data());

      for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < (pre_block_size / 2); j++) {
          and_out[i * (pre_block_size / 2) + j] ^=
              pre_gt[i * pre_block_size + 2 * j];
        }
      }

      if (pre_block_size & 1) {
        for (uint32_t i = 0; i < n; i++) {
          std::memcpy(cur_gt.data() + i * cur_block_size,
                      and_out.data() + i * (pre_block_size / 2),
                      pre_block_size / 2);
          std::memcpy(cur_eq.data() + i * cur_block_size,
                      and_out.data() + (n + i) * (pre_block_size / 2),
                      pre_block_size / 2);
          cur_gt[i * cur_block_size + cur_block_size - 1] =
              pre_gt[i * pre_block_size + pre_block_size - 1];
          cur_eq[i * cur_block_size + cur_block_size - 1] =
              pre_eq[i * pre_block_size + pre_block_size - 1];
        }
      } else {
        std::memcpy(cur_gt.data(), and_out.data(), n * cur_block_size);
        std::memcpy(cur_eq.data(), and_out.data() + n * cur_block_size,
                    n * cur_block_size);
      }

      std::swap(pre_gt, cur_gt);
      std::swap(pre_eq, cur_eq);
      pre_block_size = cur_block_size;
    } else {
      // Last round: pre_block_size = 2, cur_block_size = 1
      std::vector<uint8_t> and_input1(n);
      std::vector<uint8_t> and_input2(n);
      std::vector<uint8_t> and_out(n);

      for (uint32_t i = 0; i < n; i++) {
        and_input1[i] = pre_eq[i * pre_block_size];
        and_input2[i] = pre_gt[i * pre_block_size + 1];
      }

      and_context_->And(n, and_input1.data(), and_input2.data(),
                        and_out.data());

      for (uint32_t i = 0; i < n; i++) {
        and_out[i] ^= pre_gt[i * 2];
      }

      std::memcpy(res, and_out.data(), n);
    }
  }
}

void CompareContext::SubBlockCompare(uint32_t n, const uint8_t* input_x,
                                     uint8_t* cmp_res, uint8_t* equal_res,
                                     uint32_t bw) {
  emp::block common_aes_key = rot_context_->GetCommonAesKey();

  // Calculate block size for threading
  uint32_t block_size =
      static_cast<uint32_t>(std::ceil(1.0 * n / thread_num_ / 8.0)) * 8;
  if (block_size * thread_num_ < n) {
    // todo: add exception handling
    block_size = (n + thread_num_ - 1) / thread_num_;
  }

  // Run mill_and_equal protocol in parallel
  std::vector<std::future<void>> futures(thread_num_);
  auto thread_pool = rot_context_->GetThreadPool();

  for (uint32_t i = 0; i < thread_num_; i++) {
    uint32_t cur_block_size =
        (i * block_size < n) ? std::min(block_size, n - i * block_size) : 0;

    if ((party_id_ == kServerId) ^ (i & 1)) {
      // Sender role
      uint32_t rot_offset = rot_context_->ReserveSenderRot(
          static_cast<uint64_t>(cur_block_size) * bw);
      futures[i] = thread_pool->Submit(
          &CompareContext::MillAndEqualSender, this,
          rot_context_->GetIOArray()[i], common_aes_key, cur_block_size,
          input_x + i * block_size, cmp_res + i * block_size,
          equal_res + i * block_size, bw,
          rot_context_->GetSenderMsg0() + rot_offset,
          rot_context_->GetSenderMsg1() + rot_offset);
    } else {
      // Receiver role
      uint32_t rot_offset = rot_context_->ReserveReceiverRot(
          static_cast<uint64_t>(cur_block_size) * bw);
      futures[i] = thread_pool->Submit(
          &CompareContext::MillAndEqualReceiver, this,
          rot_context_->GetIOArray()[i], common_aes_key, cur_block_size,
          input_x + i * block_size, cmp_res + i * block_size,
          equal_res + i * block_size, bw,
          rot_context_->GetRecvChoice() + rot_offset,
          rot_context_->GetRecvMsg() + rot_offset);
    }
  }

  for (auto& f : futures) {
    f.wait();
  }
}

void CompareContext::MillAndEqualSender(
    link::LinkIO* cur_IO, const emp::block common_aes_key, uint32_t len,
    const uint8_t* x, uint8_t* mill_res, uint8_t* equal_res, uint32_t bw,
    const emp::block* send_msg0, const emp::block* send_msg1) {
  if (len == 0) return;

  uint32_t table_size = 1 << bw;
  uint8_t index_mod_mask = static_cast<uint8_t>(table_size - 1);

  std::vector<uint8_t> input_id(len);
  ElementwiseAnd(len, x, index_mod_mask, input_id.data());

  emp::AES_KEY common_AES;
  emp::AES_set_encrypt_key(common_aes_key, &common_AES);

  emp::PRG prg;
  prg.random_bool(reinterpret_cast<bool*>(mill_res), len);
  prg.random_bool(reinterpret_cast<bool*>(equal_res), len);

  std::vector<uint8_t> masked_mill(len * table_size, 0);
  std::vector<uint8_t> masked_equal(len * table_size, 0);
  std::vector<emp::block> sender_mask(len * table_size, emp::zero_block);

  // Build masked tables based on party role
  // For SERVER: mill table has 1s before input_id (prefix)
  // For CLIENT: mill table has 1s after input_id (suffix)
  for (uint32_t i = 0; i < len; i++) {
    if (party_id_ == kServerId) {
      // SERVER: prefix of 1s
      for (uint32_t j = 0; j < input_id[i]; j++) {
        masked_mill[i * table_size + j] = 1;
      }
    } else {
      // CLIENT: suffix of 1s
      for (uint32_t j = input_id[i] + 1; j < table_size; j++) {
        masked_mill[i * table_size + j] = 1;
      }
    }

    // Equal table: 1 at position input_id
    masked_equal[i * table_size + input_id[i]] = 1;

    ElementwiseXorInplace(table_size, masked_mill.data() + i * table_size,
                          mill_res[i]);
    ElementwiseXorInplace(table_size, masked_equal.data() + i * table_size,
                          equal_res[i]);
  }

  // Generate 1-T ROT from 1-2 ROT
  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < bw; j++) {
      if (j != 0) {
        for (uint32_t k = 0; k < static_cast<uint32_t>(1 << j); k++) {
          sender_mask[i * table_size + (1 << j) + k] =
              sender_mask[i * table_size + k];
        }
      }
      for (uint32_t k = 0; k < static_cast<uint32_t>(1 << j); k++) {
        sender_mask[i * table_size + k] ^= send_msg0[i * bw + j];
      }
      for (uint32_t k = 0; k < static_cast<uint32_t>(1 << j); k++) {
        sender_mask[i * table_size + (1 << j) + k] ^= send_msg1[i * bw + j];
      }
    }
    // Hash all masks
    for (uint32_t j = 0; j < table_size; j++) {
      CcrFunctionHInplace(sender_mask[i * table_size + j], common_AES);
    }
  }

  // Receive masked_index from receiver
  std::vector<uint8_t> masked_index(len);
  RecvUint8Vector(masked_index.data(), len, bw, cur_IO);

  // Mask the tables with sender_mask
  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < table_size - masked_index[i]; j++) {
      masked_mill[i * table_size + j] ^=
          (sender_mask[i * table_size + masked_index[i] + j][0] & 1);
    }
    for (uint32_t j = 0; j < masked_index[i]; j++) {
      masked_mill[i * table_size + (table_size - masked_index[i]) + j] ^=
          (sender_mask[i * table_size + j][0] & 1);
    }

    for (uint32_t j = 0; j < table_size - masked_index[i]; j++) {
      masked_equal[i * table_size + j] ^=
          (sender_mask[i * table_size + masked_index[i] + j][1] & 1);
    }
    for (uint32_t j = 0; j < masked_index[i]; j++) {
      masked_equal[i * table_size + (table_size - masked_index[i]) + j] ^=
          (sender_mask[i * table_size + j][1] & 1);
    }
  }

  // Send masked tables
  std::vector<uint8_t> masked_data(2 * len * table_size);
  std::memcpy(masked_data.data(), masked_mill.data(), len * table_size);
  std::memcpy(masked_data.data() + len * table_size, masked_equal.data(),
              len * table_size);
  SendBoolVector(reinterpret_cast<const bool*>(masked_data.data()),
                 2 * len * table_size, cur_IO);
  cur_IO->flush();
}

void CompareContext::MillAndEqualReceiver(
    link::LinkIO* cur_IO, const emp::block common_aes_key, uint32_t len,
    const uint8_t* x, uint8_t* mill_res, uint8_t* equal_res, uint32_t bw,
    const uint8_t* recv_choice, const emp::block* recv_msg) {
  if (len == 0) return;

  uint32_t table_size = 1 << bw;
  uint8_t index_mask = (bw == 8) ? static_cast<uint8_t>(-1)
                                 : static_cast<uint8_t>((1 << bw) - 1);

  emp::AES_KEY common_AES;
  emp::AES_set_encrypt_key(common_aes_key, &common_AES);

  std::vector<uint8_t> masked_index(len, 0);
  std::vector<emp::block> receiver_mask(len * table_size, emp::zero_block);

  for (uint32_t i = 0; i < len; i++) {
    for (uint32_t j = 0; j < bw; j++) {
      receiver_mask[i] ^= recv_msg[i * bw + j];
      masked_index[i] = static_cast<uint8_t>(
          (masked_index[i] << 1) | recv_choice[i * bw + (bw - j - 1)]);
    }
    CcrFunctionHInplace(receiver_mask[i], common_AES);
  }

  ElementwiseSubmodInplace(len, masked_index.data(), x, index_mask);
  SendUint8Vector(masked_index.data(), len, bw, cur_IO);
  cur_IO->flush();

  // Receive masked tables
  std::vector<uint8_t> masked_data(2 * len * table_size, 0);
  RecvBoolVector(reinterpret_cast<bool*>(masked_data.data()),
                 2 * len * table_size, cur_IO);

  uint8_t* masked_mill = masked_data.data();
  uint8_t* masked_equal = masked_data.data() + len * table_size;

  // Extract results
  for (uint32_t i = 0; i < len; i++) {
    mill_res[i] = masked_mill[i * table_size + x[i]] ^
                  static_cast<uint8_t>(receiver_mask[i][0] & 1);
    equal_res[i] = masked_equal[i * table_size + x[i]] ^
                   static_cast<uint8_t>(receiver_mask[i][1] & 1);
  }
}

}  // namespace capricorn::operator_
