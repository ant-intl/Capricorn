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

#include <array>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "fmt/format.h"
#include "src/common/runtime/link/communicator.h"
#include "src/common/runtime/link/link_io.h"
#include "src/common/runtime/test_utils.h"

#include "yacl/link/context.h"
#include "yacl/link/factory.h"
#include "yacl/utils/thread_pool.h"

namespace capricorn::link {

// A pair of LinkIO that can communicate with each other
using LinkIOPair = std::array<std::shared_ptr<LinkIO>, 2>;

// Create a LinkIO pair from a pair of yacl contexts
inline LinkIOPair MakeLinkIOPair(
    const std::vector<std::shared_ptr<yacl::link::Context>>& lctxs) {
  LinkIOPair pair;
  for (size_t rank = 0; rank < 2; rank++) {
    auto comm = std::make_shared<Communicator>(lctxs[rank]);
    pair[rank] = std::make_shared<LinkIO>(comm);
  }
  return pair;
}

// Generate a single pair of LinkIO for 2-party communication
// Uses in-memory transport only (no brpc dependency)
inline LinkIOPair MockLinkIOPair(const size_t index = 0) {
  auto lctxs = SetupBrpcWorld(fmt::format("sim-mem.{}-2PC", index), 2,
                              10096 + index * 2);
  return MakeLinkIOPair(lctxs);
}

// Generate multiple pairs of LinkIO for 2-party communication
// Each pair is independent and can communicate with each other
// num_pairs: number of LinkIO pairs to generate
inline std::vector<LinkIOPair> MockLinkIOPairs(size_t num_pairs) {
  std::vector<LinkIOPair> pairs;
  pairs.reserve(num_pairs);

  for (size_t i = 0; i < num_pairs; i++) {
    pairs.push_back(MockLinkIOPair(i));
  }

  return pairs;
}

// ---------------------------------------------------------------------------
// MockLinkIOGroup - Holds multiple LinkIO pairs with easy access to
// sender/receiver IO arrays
// ---------------------------------------------------------------------------
struct MockLinkIOGroup {
  std::vector<LinkIOPair> io_pairs;  // Owns the LinkIO objects
  std::vector<std::shared_ptr<LinkIO>>
      sender_io;  // Pointers for sender (party 1)
  std::vector<std::shared_ptr<LinkIO>>
      receiver_io;  // Pointers for receiver (party 0)

  MockLinkIOGroup() = default;
  explicit MockLinkIOGroup(size_t num_pairs) { Resize(num_pairs); }

  void Resize(size_t num_pairs) {
    io_pairs.resize(num_pairs);
    sender_io.resize(num_pairs);
    receiver_io.resize(num_pairs);
  }

  size_t size() const { return io_pairs.size(); }
};

// Generate a MockLinkIOGroup with multiple LinkIO pairs
// This is a convenience function that creates LinkIO pairs and extracts
// sender/receiver IO arrays for direct use with Splut classes.
inline MockLinkIOGroup MakeMockLinkIOGroup(size_t num_pairs) {
  MockLinkIOGroup group(num_pairs);

  for (size_t i = 0; i < num_pairs; i++) {
    group.io_pairs[i] = MockLinkIOPair(i);
    group.receiver_io[i] = group.io_pairs[i][0];  // Party 0 is receiver
    group.sender_io[i] = group.io_pairs[i][1];    // Party 1 is sender
  }

  return group;
}

// Generate multiple LinkIOs from an existing yacl::link::Context by spawning
// sub-contexts in parallel.
inline std::vector<std::shared_ptr<LinkIO>> SpawnLinkIOs(
    const std::shared_ptr<yacl::link::Context>& lctx, size_t num_threads,
    const std::string& tag_prefix = "thread") {
  std::vector<std::shared_ptr<LinkIO>> io_ptrs(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back([&lctx, &tag_prefix, &io_ptrs, i]() {
      auto sub_lctx = lctx->Spawn(fmt::format("{}_{}", tag_prefix, i));
      auto sub_comm = std::make_shared<Communicator>(std::move(sub_lctx));
      io_ptrs[i] = std::make_shared<LinkIO>(sub_comm);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  return io_ptrs;
}

inline std::vector<std::shared_ptr<LinkIO>> SpawnLinkIOs(
    const std::shared_ptr<yacl::link::Context>& lctx, size_t num_threads,
    yacl::ThreadPool& thread_pool, const std::string& tag_prefix = "thread") {
  std::vector<std::shared_ptr<LinkIO>> io_ptrs(num_threads);
  std::vector<std::future<void>> futures(num_threads);

  for (size_t i = 0; i < num_threads; i++) {
    futures[i] = thread_pool.Submit([&lctx, &tag_prefix, &io_ptrs, i]() {
      auto sub_lctx = lctx->Spawn(fmt::format("{}_{}", tag_prefix, i));
      auto sub_comm = std::make_shared<Communicator>(std::move(sub_lctx));
      io_ptrs[i] = std::make_shared<LinkIO>(sub_comm);
    });
  }

  for (auto& f : futures) {
    f.wait();
  }

  return io_ptrs;
}

// Convenience function to create a pair of spawned LinkIO arrays for 2-party
// computation. Creates two sets of LinkIOs, one for each party.
inline std::pair<std::vector<std::shared_ptr<LinkIO>>,
                 std::vector<std::shared_ptr<LinkIO>>>
SpawnLinkIOPair(const std::vector<std::shared_ptr<yacl::link::Context>>& lctxs,
                size_t num_threads, const std::string& tag_prefix = "thread") {
  if (lctxs.size() < 2) {
    throw std::invalid_argument("lctxs must have at least 2 contexts");
  }

  auto sender_ios = SpawnLinkIOs(lctxs[0], num_threads, tag_prefix);
  auto receiver_ios = SpawnLinkIOs(lctxs[1], num_threads, tag_prefix);

  return {std::move(sender_ios), std::move(receiver_ios)};
}

}  // namespace capricorn::link