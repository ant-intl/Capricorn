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

#include <future>
#include <vector>

#include "fmt/format.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/link/context.h"
#include "yacl/link/factory.h"

namespace capricorn {

namespace yl = yacl::link;
namespace {
uint32_t link_recv_timeout_ms = 1000 * 60 * 30;  // 1s
}

inline std::vector<std::shared_ptr<yl::Context>> SetupBrpcWorld(
    const std::string& id, size_t world_size, uint64_t port_id = 0) {
  if (port_id == 0) {
    auto offset = yacl::crypto::RandVec<uint16_t>(2)[0];
    port_id = 10086 + (offset & 0x3FFF);
  }
  yl::ContextDesc ctx_desc;
  // ctx_desc.id = id;
  ctx_desc.recv_timeout_ms = link_recv_timeout_ms;
  for (size_t rank = 0; rank < world_size; rank++) {
    const auto party_id = fmt::format("{}-{}", id, rank);
    const auto host = fmt::format("127.0.0.1:{}", port_id + rank);
    ctx_desc.parties.push_back({party_id, host});
  }

  std::vector<std::shared_ptr<yl::Context>> contexts(world_size);
  for (size_t rank = 0; rank < world_size; rank++) {
    contexts[rank] = yl::FactoryBrpc().CreateContext(ctx_desc, rank);
  }

  auto proc = [&](size_t rank) {
    contexts[rank]->ConnectToMesh();
    // If throttle_window_size is not zero, "SendAsync" will block until
    // messages are processed
    contexts[rank]->SetThrottleWindowSize(0);
    contexts[rank]->SetRecvTimeout(link_recv_timeout_ms);
  };
  std::vector<std::future<void>> jobs(world_size);
  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank] = std::async(proc, rank);
  }

  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank].get();
  }
  return contexts;
}

inline std::vector<std::shared_ptr<yl::Context>> SetupMemWorld(
    const std::string& id, size_t world_size) {
  yl::ContextDesc ctx_desc;
  ctx_desc.id = id;
  ctx_desc.recv_timeout_ms = link_recv_timeout_ms;

  for (size_t rank = 0; rank < world_size; rank++) {
    ctx_desc.parties.push_back(
        {fmt::format("dummy_id:{}", rank), "dummy_host"});
  }

  std::vector<std::shared_ptr<yl::Context>> contexts(world_size);
  for (size_t rank = 0; rank < world_size; rank++) {
    contexts[rank] = yl::FactoryMem().CreateContext(ctx_desc, rank);
  }

  auto proc = [&](size_t rank) { contexts[rank]->ConnectToMesh(); };
  std::vector<std::future<void>> jobs(world_size);
  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank] = std::async(proc, rank);
  }

  for (size_t rank = 0; rank < world_size; rank++) {
    jobs[rank].get();
  }
  return contexts;
}

inline std::vector<std::shared_ptr<yl::Context>> Mock2PCLink(
    bool brpc_mode = true) {
  if (brpc_mode) {
    constexpr size_t kMaxPortAttempts = 32;
    for (size_t attempt = 0; attempt < kMaxPortAttempts; ++attempt) {
      try {
        return SetupBrpcWorld(fmt::format("sim-brpc.{}PC", 2), 2);
      } catch (...) {
        if (attempt + 1 == kMaxPortAttempts) {
          throw;
        }
      }
    }
  }
  return SetupMemWorld(fmt::format("sim-mem.{}PC", 2), 2);
}

inline std::vector<std::shared_ptr<yl::Context>> MockMPCLink(
    uint64_t party_size, bool brpc_mode = true) {
  if (brpc_mode) {
    constexpr size_t kMaxPortAttempts = 32;
    for (size_t attempt = 0; attempt < kMaxPortAttempts; ++attempt) {
      try {
        return SetupBrpcWorld(fmt::format("sim-brpc.{}PC", party_size),
                              party_size);
      } catch (...) {
        if (attempt + 1 == kMaxPortAttempts) {
          throw;
        }
      }
    }
  }
  return SetupMemWorld(fmt::format("sim-mem.{}PC", party_size), party_size);
}

}  // namespace capricorn
