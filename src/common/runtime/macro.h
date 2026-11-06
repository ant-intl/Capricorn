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
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#define COMM_START(lctx, name)                                \
  auto name##_st_start = (lctx)->GetStats();                  \
  int64_t name##_send_bytes = name##_st_start->sent_bytes;    \
  int64_t name##_send_action = name##_st_start->sent_actions; \
  int64_t name##_recv_bytes = name##_st_start->recv_bytes;    \
  int64_t name##_recv_action = name##_st_start->recv_actions;

#define COMM_END(lctx, name)                                             \
  auto name##_st_end = (lctx)->GetStats();                               \
  name##_send_bytes = name##_st_end->sent_bytes - name##_send_bytes;     \
  name##_send_action = name##_st_end->sent_actions - name##_send_action; \
  name##_recv_bytes = name##_st_end->recv_bytes - name##_recv_bytes;     \
  name##_recv_action = name##_st_end->recv_actions - name##_recv_action;

#define COMM_PRINT(lctx, name)                                              \
  SPDLOG_INFO(                                                              \
      "[P{}](COMM) {} send bytes: {} ( {:.2f} MB ) && send actions: {} && " \
      "recv bytes: {} ( {:.2f} MB ) && recv actions: {}",                   \
      (lctx)->Rank(), std::string(#name), name##_send_bytes,                \
      name##_send_bytes * 1.0 / 1024 / 1024, name##_send_action,            \
      name##_recv_bytes, name##_recv_bytes * 1.0 / 1024 / 1024,             \
      name##_recv_action);

#define COMM_END_PRINT(lctx, name) \
  COMM_END(lctx, name);            \
  COMM_PRINT(lctx, name)

#define TIMER_START(name) \
  auto name##_begin = std::chrono::high_resolution_clock::now();

#define TIMER_END(name)                                                    \
  auto name##_end = std::chrono::high_resolution_clock::now();             \
  auto name##_elapse = name##_end - name##_begin;                          \
  double name##_ms =                                                       \
      std::chrono::duration_cast<std::chrono::milliseconds>(name##_elapse) \
          .count();

#define TIMER_PRINT(name)                                            \
  SPDLOG_INFO("(TIMER) {} need {} ms (or {} s)", std::string(#name), \
              name##_ms, name##_ms / 1000);

#define TIMER_END_PRINT(name) \
  TIMER_END(name);            \
  TIMER_PRINT(name);

#define COMM_TIMER_START(lctx, name) \
  COMM_START(lctx, name);            \
  TIMER_START(name)

#define COMM_TIMER_END_PRINT(lctx, name) \
  COMM_END_PRINT(lctx, name);            \
  TIMER_END_PRINT(name);

// ============================================================================
// RAII-style scoped timer for measuring communication and time
// Supports runtime name (unlike macros which require compile-time names)
// ============================================================================

template <typename LinkIOType>
class ScopedCommTimer {
 public:
  explicit ScopedCommTimer(std::shared_ptr<LinkIOType> lctx, std::string name)
      : lctx_(lctx), name_(std::move(name)) {
    // Record initial stats
    send_bytes_ = 0;
    send_action_ = 0;
    recv_bytes_ = 0;
    recv_action_ = 0;
    auto st = lctx->GetStats();
    send_bytes_ += st->sent_bytes;
    send_action_ += st->sent_actions;
    recv_bytes_ += st->recv_bytes;
    recv_action_ += st->recv_actions;
    begin_ = std::chrono::high_resolution_clock::now();
  }

  ~ScopedCommTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin_)
            .count();

    int64_t end_send_bytes = 0;
    int64_t end_send_action = 0;
    int64_t end_recv_bytes = 0;
    int64_t end_recv_action = 0;
    auto st = lctx_->GetStats();
    end_send_bytes += st->sent_bytes;
    end_send_action += st->sent_actions;
    end_recv_bytes += st->recv_bytes;
    end_recv_action += st->recv_actions;

    int64_t delta_send_bytes = end_send_bytes - send_bytes_;
    int64_t delta_send_action = end_send_action - send_action_;
    int64_t delta_recv_bytes = end_recv_bytes - recv_bytes_;
    int64_t delta_recv_action = end_recv_action - recv_action_;

    SPDLOG_INFO(
        "{} send bytes: {} ({:.2f} MB) && send actions: {} && "
        "recv bytes: {} ({:.2f} MB) && recv actions: {} | time: {:.2f} ms (or "
        "{} s)",
        name_, delta_send_bytes, delta_send_bytes * 1.0 / 1024 / 1024,
        delta_send_action, delta_recv_bytes,
        delta_recv_bytes * 1.0 / 1024 / 1024, delta_recv_action, ms, ms / 1000);
  }

  // Non-copyable, non-movable
  ScopedCommTimer(const ScopedCommTimer&) = delete;
  ScopedCommTimer& operator=(const ScopedCommTimer&) = delete;
  ScopedCommTimer(ScopedCommTimer&&) = delete;
  ScopedCommTimer& operator=(ScopedCommTimer&&) = delete;

 private:
  std::shared_ptr<LinkIOType> lctx_;
  std::string name_;
  int64_t send_bytes_;
  int64_t send_action_;
  int64_t recv_bytes_;
  int64_t recv_action_;
  std::chrono::high_resolution_clock::time_point begin_;
};
