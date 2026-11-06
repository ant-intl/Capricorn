// Copyright 2026 Ant International, Ant Group. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "spdlog/spdlog.h"

#include "yacl/link/context.h"

namespace capricorn::vmoe {

struct ProcessMemorySnapshot final {
  int64_t current_rss_bytes{-1};
  int64_t peak_rss_bytes{-1};
};

inline ProcessMemorySnapshot ReadProcessMemorySnapshot() {
  ProcessMemorySnapshot snapshot;

  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
    snapshot.peak_rss_bytes = static_cast<int64_t>(usage.ru_maxrss);
#else
    snapshot.peak_rss_bytes = static_cast<int64_t>(usage.ru_maxrss) * 1024;
#endif
  }

#if defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    snapshot.current_rss_bytes = static_cast<int64_t>(info.resident_size);
  }
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  int64_t total_pages = 0;
  int64_t resident_pages = 0;
  if (statm >> total_pages >> resident_pages) {
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0) {
      snapshot.current_rss_bytes = resident_pages * page_size;
    }
  }
#endif

  return snapshot;
}

struct LinkMetricSnapshot final {
  int64_t sent_bytes{0};
  int64_t sent_actions{0};
  int64_t recv_bytes{0};
  int64_t recv_actions{0};
};

inline LinkMetricSnapshot ReadLinkMetricSnapshot(
    const std::shared_ptr<yacl::link::Context>& link) {
  LinkMetricSnapshot snapshot;
  if (link) {
    const auto stats = link->GetStats();
    snapshot.sent_bytes = stats->sent_bytes;
    snapshot.sent_actions = stats->sent_actions;
    snapshot.recv_bytes = stats->recv_bytes;
    snapshot.recv_actions = stats->recv_actions;
  }
  return snapshot;
}

inline LinkMetricSnapshot operator-(const LinkMetricSnapshot& lhs,
                                    const LinkMetricSnapshot& rhs) {
  return {
      lhs.sent_bytes - rhs.sent_bytes,
      lhs.sent_actions - rhs.sent_actions,
      lhs.recv_bytes - rhs.recv_bytes,
      lhs.recv_actions - rhs.recv_actions,
  };
}

class StageMetric final {
 public:
  StageMetric(std::shared_ptr<yacl::link::Context> link, std::string stage,
              std::string operation, int64_t rank = -1)
      : link_(std::move(link)),
        stage_(std::move(stage)),
        operation_(std::move(operation)),
        rank_(link_ ? static_cast<int64_t>(link_->Rank()) : rank),
        begin_(std::chrono::steady_clock::now()) {
    if (link_) {
      const auto stats = link_->GetStats();
      sent_bytes_ = stats->sent_bytes;
      sent_actions_ = stats->sent_actions;
      recv_bytes_ = stats->recv_bytes;
      recv_actions_ = stats->recv_actions;
    }
  }

  ~StageMetric() {
    int64_t sent_bytes = 0;
    int64_t sent_actions = 0;
    int64_t recv_bytes = 0;
    int64_t recv_actions = 0;
    if (link_) {
      const auto stats = link_->GetStats();
      sent_bytes = stats->sent_bytes - sent_bytes_;
      sent_actions = stats->sent_actions - sent_actions_;
      recv_bytes = stats->recv_bytes - recv_bytes_;
      recv_actions = stats->recv_actions - recv_actions_;
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin_);
    const auto memory = ReadProcessMemorySnapshot();
    SPDLOG_INFO(
        "[VMOE_METRIC] rank={} stage={} operator={} elapsed_ms={:.3f} "
        "sent_bytes={} sent_actions={} recv_bytes={} recv_actions={} "
        "current_rss_bytes={} peak_rss_bytes={}",
        rank_, stage_, operation_, elapsed.count(), sent_bytes, sent_actions,
        recv_bytes, recv_actions, memory.current_rss_bytes,
        memory.peak_rss_bytes);
  }

  StageMetric(const StageMetric&) = delete;
  StageMetric& operator=(const StageMetric&) = delete;

 private:
  std::shared_ptr<yacl::link::Context> link_;
  std::string stage_;
  std::string operation_;
  int64_t rank_;
  int64_t sent_bytes_{0};
  int64_t sent_actions_{0};
  int64_t recv_bytes_{0};
  int64_t recv_actions_{0};
  std::chrono::steady_clock::time_point begin_;
};

class ModelStageRecorder final {
 public:
  struct CompletedStageMetric final {
    std::string stage;
    int64_t operation_count{0};
    double operation_elapsed_ms{0.0};
    LinkMetricSnapshot link;
    ProcessMemorySnapshot memory;
  };

  explicit ModelStageRecorder(std::shared_ptr<yacl::link::Context> link)
      : link_(std::move(link)),
        rank_(link_ ? static_cast<int64_t>(link_->Rank()) : -1),
        stage_begin_(std::chrono::steady_clock::now()),
        stage_stats_(ReadLinkMetricSnapshot(link_)) {}

  std::optional<CompletedStageMetric> Boundary(
      const std::string& completed_stage) {
    if (completed_stage == "end") {
      operations_.clear();
      stage_begin_ = std::chrono::steady_clock::now();
      stage_stats_ = ReadLinkMetricSnapshot(link_);
      return std::nullopt;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto stats = ReadLinkMetricSnapshot(link_);
    const auto delta = stats - stage_stats_;
    const auto elapsed =
        std::chrono::duration<double, std::milli>(now - stage_begin_);
    const auto memory = ReadProcessMemorySnapshot();
    CompletedStageMetric completed;
    completed.stage = completed_stage;
    completed.link = delta;
    completed.memory = memory;
    SPDLOG_INFO(
        "[VMOE_STAGE] rank={} stage={} elapsed_ms={:.3f} sent_bytes={} "
        "sent_actions={} recv_bytes={} recv_actions={} "
        "current_rss_bytes={} peak_rss_bytes={}",
        rank_, completed_stage, elapsed.count(), delta.sent_bytes,
        delta.sent_actions, delta.recv_bytes, delta.recv_actions,
        memory.current_rss_bytes, memory.peak_rss_bytes);
    for (const auto& [operation, aggregate] : operations_) {
      completed.operation_count += aggregate.count;
      completed.operation_elapsed_ms += aggregate.elapsed_ms;
      SPDLOG_INFO(
          "[VMOE_OPERATOR] rank={} stage={} operator={} count={} "
          "elapsed_ms={:.3f} sent_bytes={} sent_actions={} recv_bytes={} "
          "recv_actions={}",
          rank_, completed_stage, operation, aggregate.count,
          aggregate.elapsed_ms, aggregate.link.sent_bytes,
          aggregate.link.sent_actions, aggregate.link.recv_bytes,
          aggregate.link.recv_actions);
    }
    operations_.clear();
    stage_begin_ = now;
    stage_stats_ = stats;
    return completed;
  }

  void RecordOperation(const std::string& operation, double elapsed_ms,
                       const LinkMetricSnapshot& link_delta) {
    auto& aggregate = operations_[operation];
    ++aggregate.count;
    aggregate.elapsed_ms += elapsed_ms;
    aggregate.link.sent_bytes += link_delta.sent_bytes;
    aggregate.link.sent_actions += link_delta.sent_actions;
    aggregate.link.recv_bytes += link_delta.recv_bytes;
    aggregate.link.recv_actions += link_delta.recv_actions;
  }

 private:
  struct OperationAggregate final {
    int64_t count{0};
    double elapsed_ms{0.0};
    LinkMetricSnapshot link;
  };

  std::shared_ptr<yacl::link::Context> link_;
  int64_t rank_{-1};
  std::chrono::steady_clock::time_point stage_begin_;
  LinkMetricSnapshot stage_stats_;
  std::map<std::string, OperationAggregate> operations_;
};

}  // namespace capricorn::vmoe
