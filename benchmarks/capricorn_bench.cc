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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/debugging/failure_signal_handler.h"
#include "absl/debugging/symbolize.h"
#include "absl/strings/str_split.h"
#include "fmt/format.h"
#include "gflags/gflags.h"
#include "spdlog/spdlog.h"
#include "src/capricorn/activation/secure_silu.h"
#include "src/capricorn/expert_compute.h"
#include "src/capricorn/routing.h"
#include "src/common/moe/combine.h"
#include "src/common/runtime/environment.h"
#include "src/common/runtime/macro.h"
#include "src/common/runtime/prot_helper.h"
#include "src/common/runtime/prot_interface.h"
#include "xtensor/containers/xadapt.hpp"

#include "yacl/crypto/rand/rand.h"
#include "yacl/link/context.h"
#include "yacl/link/factory.h"

DEFINE_bool(role, true, "true for server, false for client");
DEFINE_string(parties, "127.0.0.1:10086,127.0.0.1:10087",
              "comma-separated two-party endpoints");
DEFINE_string(job_id, "manual", "runner-assigned deterministic job id");
DEFINE_uint64(link_recv_timeout_ms, 86400000,
              "YACL receive timeout propagated to all spawned links");
DEFINE_int32(query_count, 1, "one full query with fresh protocol state");
DEFINE_double(capacity_factor, 2.0, "balanced-routing capacity factor");

DEFINE_int32(token_nums, 16, "number of input tokens");
DEFINE_int32(embedding_dimension, 128, "token embedding dimension");
DEFINE_int32(expert_nums, 64, "total routed expert count");
DEFINE_int32(expert_dimension, 64, "expert hidden dimension");
DEFINE_int32(active_expert_nums, 6,
             "reference token-wise Top-k used to derive capacity");
DEFINE_int32(dispatch_expert_per_token, 0,
             "per-expert capacity; zero derives it from capacity_factor");

DEFINE_bool(routing_only, false,
            "run only Routing, with legacy routing metrics");

namespace capricorn {
namespace {

struct BenchParams {
  size_t token_nums;
  size_t embedding_dimension;
  size_t expert_nums;
  size_t expert_dimension;
  size_t k_base;
  size_t dispatch_capacity;
  size_t query_count;
  double capacity_factor;

  std::string Serialize() const {
    return fmt::format("{},{},{},{},{},{},{},{:.17g}", token_nums,
                       embedding_dimension, expert_nums, expert_dimension,
                       k_base, dispatch_capacity, query_count, capacity_factor);
  }
};

// Validate signed flags before converting them to sizes or deriving capacity.
BenchParams ReadParams() {
  YACL_ENFORCE(FLAGS_token_nums > 0 && FLAGS_embedding_dimension > 0 &&
                   FLAGS_expert_nums > 0 && FLAGS_expert_dimension > 0 &&
                   FLAGS_active_expert_nums > 0 &&
                   FLAGS_active_expert_nums <= FLAGS_expert_nums,
               "benchmark dimensions and Top-k must be positive and valid");
  YACL_ENFORCE(FLAGS_dispatch_expert_per_token >= 0,
               "dispatch capacity must be non-negative");
  YACL_ENFORCE(
      std::isfinite(FLAGS_capacity_factor) && FLAGS_capacity_factor > 0,
      "capacity factor must be finite and positive");
  YACL_ENFORCE(FLAGS_link_recv_timeout_ms > 0,
               "link receive timeout must be positive");
  YACL_ENFORCE(FLAGS_query_count == 1, "Capricorn requires --query_count=1");

  const double derived_capacity = std::ceil(
      FLAGS_capacity_factor * FLAGS_token_nums * FLAGS_active_expert_nums /
      static_cast<double>(FLAGS_expert_nums));
  const double capacity = FLAGS_dispatch_expert_per_token == 0
                              ? derived_capacity
                              : FLAGS_dispatch_expert_per_token;
  YACL_ENFORCE(
      std::isfinite(capacity) && capacity > 0 &&
          capacity < static_cast<double>(std::numeric_limits<size_t>::max()),
      "derived dispatch capacity is out of range");
  return BenchParams{
      static_cast<size_t>(FLAGS_token_nums),
      static_cast<size_t>(FLAGS_embedding_dimension),
      static_cast<size_t>(FLAGS_expert_nums),
      static_cast<size_t>(FLAGS_expert_dimension),
      static_cast<size_t>(FLAGS_active_expert_nums),
      static_cast<size_t>(capacity),
      FLAGS_routing_only ? 1U : static_cast<size_t>(FLAGS_query_count),
      FLAGS_capacity_factor,
  };
}

struct Configures {
  RoutingConfig routing_config;
  SiLUExpertComputeConfig expert_compute_config;
  CombineConfig combine_config;
};

std::shared_ptr<yacl::link::Context> MakeLink(const std::string& parties,
                                              size_t rank,
                                              uint64_t recv_timeout_ms) {
  yacl::link::ContextDesc desc;
  std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  YACL_ENFORCE(hosts.size() == 2, "benchmark requires exactly two parties");
  for (size_t party = 0; party < hosts.size(); ++party) {
    desc.parties.emplace_back(fmt::format("party{}", party), hosts[party]);
  }
  desc.throttle_window_size = 0;
  desc.recv_timeout_ms = recv_timeout_ms;
  auto link = yacl::link::FactoryBrpc().CreateContext(desc, rank);
  link->ConnectToMesh();
  link->SetRecvTimeout(recv_timeout_ms);
  return link;
}

void SyncParams(const std::shared_ptr<yacl::link::Context>& link,
                const BenchParams& params) {
  // Control flags also affect the peer communication sequence.
  const auto serialized =
      fmt::format("{},{}", params.Serialize(), FLAGS_routing_only);
  const yacl::ByteContainerView bytes(serialized);
  yacl::Buffer remote;
  if (link->Rank() == 0) {
    remote = link->Recv(link->NextRank(), "SyncParams");
    link->Send(link->NextRank(), bytes, "SyncParams");
  } else {
    link->Send(link->NextRank(), bytes, "SyncParams");
    remote = link->Recv(link->NextRank(), "SyncParams");
  }
  YACL_ENFORCE(yacl::ByteContainerView(remote) == bytes,
               "server and client benchmark parameters differ");
}

// Match the original benchmark: fresh random shares/weights, approximately
// [0, 2), without a deterministic seed. Floating-point rounding is unchanged.
std::vector<double> MockFxpData(size_t size) {
  const auto random = yacl::crypto::RandVec<uint64_t>(size);
  std::vector<double> output(size);
  std::transform(random.begin(), random.end(), output.begin(),
                 [](uint64_t value) { return value * 1.0 / std::pow(2, 63); });
  return output;
}

std::unique_ptr<Environment> InitializeEnvironment(
    const std::shared_ptr<yacl::link::Context>& link, const BenchParams& params,
    const Configures& configs) {
  auto env = std::make_unique<Environment>(link);
  auto token_share =
      MockFxpData(params.token_nums * params.embedding_dimension);

  if (link->Rank() == 0) {
    auto gate_weights =
        MockFxpData(params.expert_nums * params.embedding_dimension);
    auto w1 = MockFxpData(params.expert_nums * params.embedding_dimension *
                          params.expert_dimension);
    auto w2 = MockFxpData(params.expert_nums * params.expert_dimension *
                          params.embedding_dimension);

    env->SetPrivate(configs.routing_config.kGateWeights,
                    xt::adapt(gate_weights, {params.expert_nums,
                                             params.embedding_dimension}));
    env->SetPrivate(
        configs.expert_compute_config.kExpertsWeightsOne,
        xt::adapt(w1, {params.expert_nums, params.embedding_dimension,
                       params.expert_dimension}));
    env->SetPrivate(configs.expert_compute_config.kExpertsWeightsTwo,
                    xt::adapt(w2, {params.expert_nums, params.expert_dimension,
                                   params.embedding_dimension}));
  } else {
    env->GetPrivate(configs.routing_config.kGateWeights);
    env->GetPrivate(configs.expert_compute_config.kExpertsWeightsOne);
    env->GetPrivate(configs.expert_compute_config.kExpertsWeightsTwo);
  }

  env->FeedShare(
      configs.routing_config.kTokenEmbeddings,
      xt::adapt(token_share, {params.token_nums, params.embedding_dimension}));
  return env;
}

void ComputeAndSteal(std::unique_ptr<Environment>& env,
                     std::unique_ptr<ProtInterface> protocol) {
  protocol->InitEnv(std::move(env));
  protocol->Compute();
  env = protocol->StealEnv();
}

// Keep the established routing-only CLI and its parser-facing log labels.
// This path never creates persistent expert sessions or full-query records.
void RunRoutingOnly(std::unique_ptr<Environment>& env,
                    const Configures& configs) {
  const auto link = env->lctx_;
  const bool is_server = link->Rank() == 0;
  COMM_TIMER_START(link, DENSE_MOE);
  COMM_START(link, ROUTING);
  ComputeAndSteal(env, MakeProtocol(configs.routing_config, is_server));
  COMM_END_PRINT(link, ROUTING);
  COMM_TIMER_END_PRINT(link, DENSE_MOE);
  SPDLOG_INFO("--- Routing-only benchmark DONE ---");
}

void RunQuery(std::unique_ptr<Environment>& env, const Configures& configs) {
  const auto link = env->lctx_;
  const bool is_server = link->Rank() == 0;
  COMM_START(link, ROUTING);
  ComputeAndSteal(env, MakeProtocol(configs.routing_config, is_server));
  COMM_END_PRINT(link, ROUTING);
  ComputeAndSteal(env, MakeProtocol(configs.expert_compute_config, is_server));
  COMM_START(link, COMBINE);
  ComputeAndSteal(env, MakeProtocol(configs.combine_config, is_server));
  COMM_END_PRINT(link, COMBINE);
}

void ShutdownRootLinkOnce(const std::shared_ptr<yacl::link::Context>& link,
                          bool* link_shutdown) {
  YACL_ENFORCE(link_shutdown != nullptr);
  if (link != nullptr && !*link_shutdown) {
    link->WaitLinkTaskFinish();
    *link_shutdown = true;
  }
}

}  // namespace
}  // namespace capricorn

int main(int argc, char** argv) {
  absl::InitializeSymbolizer(argv[0]);
  absl::FailureSignalHandlerOptions failure_options;
  absl::InstallFailureSignalHandler(failure_options);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace capricorn;
  const size_t rank = FLAGS_role ? 0 : 1;
  std::shared_ptr<yacl::link::Context> link;
  bool link_shutdown = false;

  try {
    const auto params = ReadParams();
    link = MakeLink(FLAGS_parties, rank, FLAGS_link_recv_timeout_ms);
    SyncParams(link, params);
    // Configuration evidence only; outside all original measurement intervals.
    SPDLOG_INFO(
        "BENCH_CONFIG framework=capricorn role={} job_id={} params={} "
        "input_profile=legacy_rand_u64_div_2p63",
        rank, FLAGS_job_id, params.Serialize());

    Configures configs{
        RoutingConfig(params.token_nums, params.expert_nums,
                      params.dispatch_capacity, params.embedding_dimension),
        SiLUExpertComputeConfig(params.expert_nums, params.dispatch_capacity,
                                params.embedding_dimension,
                                params.expert_dimension),
        CombineConfig(params.token_nums, params.expert_nums,
                      params.dispatch_capacity, params.embedding_dimension),
    };
    if (FLAGS_routing_only) {
      auto env = InitializeEnvironment(link, params, configs);
      RunRoutingOnly(env, configs);
      ShutdownRootLinkOnce(link, &link_shutdown);
      return 0;
    }

    // Input/environment creation stays outside the original full-layer timer.
    // Empty session pointers preserve one-shot initialization inside W1/W2.
    auto env = InitializeEnvironment(link, params, configs);
    COMM_TIMER_START(link, DENSE_MOE);
    RunQuery(env, configs);
    COMM_TIMER_END_PRINT(link, DENSE_MOE);
    ShutdownRootLinkOnce(link, &link_shutdown);
    return 0;
  } catch (const std::exception& error) {
    ShutdownRootLinkOnce(link, &link_shutdown);
    SPDLOG_ERROR("Benchmark failed: {}", error.what());
    return 1;
  }
}
