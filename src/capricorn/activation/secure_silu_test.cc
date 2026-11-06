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

#include "src/capricorn/activation/secure_silu.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <vector>

#include "emp-tool/emp-tool.h"
#include "gtest/gtest.h"
#include "src/common/mpc/elementwise.h"
#include "src/common/mpc/preprocess.h"
#include "src/common/runtime/link/mock_link_io.h"
#include "src/common/runtime/macro.h"

#include "yacl/link/link.h"

namespace capricorn::silu {

// Import constants from operator_ namespace
constexpr uint32_t kServerId = operator_::kServerId;
constexpr uint32_t kClientId = operator_::kClientId;

// Reference SiLU function for testing
double ReferenceSilu(double x) {
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  return x / (1.0 + std::exp(-x));
}

double ReferenceSiluPiecewise(double x) {
  if (x < -8) {
    return 0.0;
  } else if (x >= 8) {
    return x;
  } else {
    int index = static_cast<int>(floor(x + 8));
    if (index < 0) index = 0;
    if (index > 15) index = 15;
    return kSlope[index] * x + kIntercept[index];
  }
}

// Convert double to fixed-point
uint64_t ToFixedPoint(double x, uint32_t scale) {
  return static_cast<uint64_t>(
      std::llround(x * static_cast<double>(1ULL << scale)));
}

// Convert fixed-point to double
double FromFixedPoint(uint64_t x, uint32_t scale) {
  return static_cast<double>(static_cast<int64_t>(x)) /
         static_cast<double>(1ULL << scale);
}

struct SecureSiluTestParams {
  const char* kProfile;
  bool kEnforceAccuracy;
  uint32_t kTestIOBw;
  uint32_t kTestIOScaleBw;
  uint32_t kTestInputBw;
  uint32_t kTestInputIntBw;
  uint32_t kTestInputScaleBw;
  uint32_t kTestTableABw;
  uint32_t kTestTableDBw;
  uint32_t kTestTableAScaleBw;
  uint32_t kTestTableDScaleBw;
  uint32_t kTestThreadNum;
  uint64_t kTestDataSize;
};

class SecureSiluTest : public ::testing::TestWithParam<SecureSiluTestParams> {
 protected:
  void SetUp() override {
    lctxs_ = capricorn::SetupBrpcWorld("secure_silu_test", 2);
  }

  std::vector<std::shared_ptr<yacl::link::Context>> lctxs_;
  SiluConfig config_;
  uint64_t data_size_;
};

TEST_P(SecureSiluTest, RandomDataTest) {
  auto params = GetParam();
  config_.io_bw = params.kTestIOBw;
  config_.io_scale_bw = params.kTestIOScaleBw;
  config_.input_bw = params.kTestInputBw;
  config_.input_int_bw = params.kTestInputIntBw;
  config_.input_scale_bw = params.kTestInputScaleBw;
  config_.table_a_bw = params.kTestTableABw;
  config_.table_d_bw = params.kTestTableDBw;
  config_.table_a_scale_bw = params.kTestTableAScaleBw;
  config_.table_d_scale_bw = params.kTestTableDScaleBw;
  config_.thread_num = params.kTestThreadNum;
  data_size_ = params.kTestDataSize;

  std::cout << "SILU_EXPERIMENT"
            << " profile=" << params.kProfile << " data_size=" << data_size_
            << " accuracy_enforced=" << std::boolalpha
            << params.kEnforceAccuracy << " io_format=(" << config_.io_bw << ","
            << config_.io_scale_bw << ")"
            << " input_format=(" << config_.input_bw << ","
            << config_.input_scale_bw << ")"
            << " slope_format=(" << config_.table_a_bw << ","
            << config_.table_a_scale_bw << ")"
            << " intercept_format=(" << config_.table_d_bw << ","
            << config_.table_d_scale_bw << ")"
            << " threads=" << config_.thread_num << std::endl;

  const uint64_t mod_mask =
      config_.io_bw == 64 ? -1ULL : ((1ULL << config_.io_bw) - 1);
  const uint64_t max_value = (8) * (1ULL << config_.io_scale_bw);

  emp::PRG prg;
  std::vector<uint64_t> server_input(data_size_);
  std::vector<uint64_t> client_input(data_size_);
  std::vector<int64_t> original_values(data_size_);

  prg.random_data(original_values.data(), data_size_ * sizeof(uint64_t));
  prg.random_data(client_input.data(), data_size_ * sizeof(uint64_t));
  for (uint32_t i = 0; i < data_size_; i++) {
    original_values[i] = (original_values[i] % (2 * max_value)) - max_value;
    client_input[i] &= mod_mask;
    server_input[i] = (original_values[i] - client_input[i]) & mod_mask;
  }
  // Run protocol
  std::vector<uint64_t> server_result(data_size_);
  std::vector<uint64_t> client_result(data_size_);

  auto server_thread = std::async([&]() {
    auto spawned_link_ios = link::SpawnLinkIOs(lctxs_[0], config_.thread_num);
    server_result =
        SecureSiluWitLinks(spawned_link_ios, server_input, kServerId, config_);
  });

  auto client_thread = std::async([&]() {
    auto spawned_link_ios = link::SpawnLinkIOs(lctxs_[1], config_.thread_num);
    client_result =
        SecureSiluWitLinks(spawned_link_ios, client_input, kClientId, config_);
  });

  server_thread.get();
  client_thread.get();

  // Verify results
  uint64_t max_error = 0;
  uint32_t max_error_index = 0;
  int64_t max_error_result = 0;
  int64_t max_error_expected = 0;
  uint64_t total_error = 0;
  for (uint32_t i = 0; i < data_size_; i++) {
    int64_t result = operator_::GetSignedValue(
        (server_result[i] + client_result[i]) & mod_mask, config_.io_bw);
    auto x = FromFixedPoint(original_values[i], config_.io_scale_bw);
    int64_t expected =
        ToFixedPoint(ReferenceSiluPiecewise(x), config_.io_scale_bw);
    uint64_t error = std::abs(result - expected);
    total_error += error;
    // SPDLOG_INFO("Index: {}, Input: {}, Result: {}, Expected: {}, Error: {}",
    // i, x, result, expected, error);
    if (error > max_error) {
      max_error = error;
      max_error_index = i;
      max_error_result = result;
      max_error_expected = expected;
    }
  }

  const double average_error =
      static_cast<double>(total_error) / static_cast<double>(data_size_);
  std::cout << "SILU_ACCURACY"
            << " profile=" << params.kProfile << " data_size=" << data_size_
            << " average_ulp_error=" << average_error
            << " max_ulp_error=" << max_error
            << " max_error_index=" << max_error_index
            << " max_error_input=" << original_values[max_error_index]
            << " max_error_result=" << max_error_result
            << " max_error_expected=" << max_error_expected << std::endl;
  if (params.kEnforceAccuracy) {
    EXPECT_LT(max_error, 200) << "Max error exceeds threshold for random data";
  } else if (max_error >= 200) {
    std::cout << "SILU_ACCURACY_WARNING"
              << " profile=" << params.kProfile << " performance_only=true"
              << " max_ulp_error=" << max_error << std::endl;
  }
}

TEST(SecureSiluSessionTest, EvaluatesTwoBatchesWithOneInitialization) {
  auto lctxs = capricorn::SetupBrpcWorld("secure_silu_session_test", 2);
  SiluConfig config;
  config.thread_num = 2;
  constexpr uint32_t data_size = 32;
  const uint64_t mod_mask = -1ULL;

  std::vector<int64_t> first_values(data_size);
  std::vector<int64_t> second_values(data_size);
  std::vector<uint64_t> first_server(data_size);
  std::vector<uint64_t> first_client(data_size);
  std::vector<uint64_t> second_server(data_size);
  std::vector<uint64_t> second_client(data_size);

  emp::PRG prg;
  prg.random_data(first_client.data(), data_size * sizeof(uint64_t));
  prg.random_data(second_client.data(), data_size * sizeof(uint64_t));
  for (uint32_t index = 0; index < data_size; ++index) {
    const double first =
        static_cast<double>(static_cast<int32_t>(index % 9) - 4);
    const double second =
        static_cast<double>(4 - static_cast<int32_t>(index % 9));
    first_values[index] =
        static_cast<int64_t>(ToFixedPoint(first, config.io_scale_bw));
    second_values[index] =
        static_cast<int64_t>(ToFixedPoint(second, config.io_scale_bw));
    first_server[index] =
        (static_cast<uint64_t>(first_values[index]) - first_client[index]) &
        mod_mask;
    second_server[index] =
        (static_cast<uint64_t>(second_values[index]) - second_client[index]) &
        mod_mask;
  }

  std::vector<uint64_t> first_server_result;
  std::vector<uint64_t> first_client_result;
  std::vector<uint64_t> second_server_result;
  std::vector<uint64_t> second_client_result;
  uint64_t server_initializations = 0;
  uint64_t client_initializations = 0;
  uint64_t server_generations = 0;
  uint64_t client_generations = 0;

  auto server_thread = std::async([&]() {
    SecureSiluSession session(lctxs[0], kServerId, config);
    first_server_result = session.Evaluate(first_server);
    EXPECT_ANY_THROW(session.Prepare(data_size, 1));
    EXPECT_ANY_THROW(session.Online(data_size, first_server));
    second_server_result = session.Evaluate(second_server);
    server_initializations = session.GetInitializationCount();
    server_generations = session.GetGenerationCount();
  });
  auto client_thread = std::async([&]() {
    SecureSiluSession session(lctxs[1], kClientId, config);
    first_client_result = session.Evaluate(first_client);
    EXPECT_ANY_THROW(session.Prepare(data_size, 1));
    EXPECT_ANY_THROW(session.Online(data_size, first_client));
    second_client_result = session.Evaluate(second_client);
    client_initializations = session.GetInitializationCount();
    client_generations = session.GetGenerationCount();
  });
  server_thread.get();
  client_thread.get();

  auto expect_close = [&](const std::vector<uint64_t>& server_result,
                          const std::vector<uint64_t>& client_result,
                          const std::vector<int64_t>& values) {
    for (uint32_t index = 0; index < data_size; ++index) {
      const int64_t actual = operator_::GetSignedValue(
          (server_result[index] + client_result[index]) & mod_mask,
          config.io_bw);
      const double input = FromFixedPoint(values[index], config.io_scale_bw);
      const int64_t expected = static_cast<int64_t>(
          ToFixedPoint(ReferenceSiluPiecewise(input), config.io_scale_bw));
      EXPECT_LT(std::abs(actual - expected), 200) << "index=" << index;
    }
  };
  expect_close(first_server_result, first_client_result, first_values);
  expect_close(second_server_result, second_client_result, second_values);
  EXPECT_EQ(server_initializations, 1);
  EXPECT_EQ(client_initializations, 1);
  EXPECT_EQ(server_generations, 2);
  EXPECT_EQ(client_generations, 2);
}

TEST(SecureSiluSessionTest, ReusesIndependentStateForDifferentBatchSizes) {
  auto lctxs =
      capricorn::SetupBrpcWorld("secure_silu_session_mixed_sizes_test", 2);
  SiluConfig config;
  config.thread_num = 2;
  constexpr uint32_t large_size = 32;
  constexpr uint32_t small_size = 16;
  const uint64_t mod_mask = -1ULL;

  auto make_values = [&](uint32_t size, int32_t offset) {
    std::vector<int64_t> values(size);
    for (uint32_t index = 0; index < size; ++index) {
      const double value =
          static_cast<double>((static_cast<int32_t>(index) + offset) % 9 - 4);
      values[index] =
          static_cast<int64_t>(ToFixedPoint(value, config.io_scale_bw));
    }
    return values;
  };
  const auto large_values = make_values(large_size, 0);
  const auto small_values = make_values(small_size, 3);

  emp::PRG prg;
  auto make_shares = [&](const std::vector<int64_t>& values) {
    std::vector<uint64_t> client(values.size());
    std::vector<uint64_t> server(values.size());
    prg.random_data(client.data(), client.size() * sizeof(uint64_t));
    for (size_t index = 0; index < values.size(); ++index) {
      server[index] =
          (static_cast<uint64_t>(values[index]) - client[index]) & mod_mask;
    }
    return std::pair{std::move(server), std::move(client)};
  };
  auto large_shares = make_shares(large_values);
  auto large_server = std::move(large_shares.first);
  auto large_client = std::move(large_shares.second);
  auto small_shares = make_shares(small_values);
  auto small_server = std::move(small_shares.first);
  auto small_client = std::move(small_shares.second);

  std::vector<uint64_t> first_large_server_result;
  std::vector<uint64_t> first_large_client_result;
  std::vector<uint64_t> small_server_result;
  std::vector<uint64_t> small_client_result;
  std::vector<uint64_t> second_large_server_result;
  std::vector<uint64_t> second_large_client_result;
  uint64_t server_initializations = 0;
  uint64_t client_initializations = 0;
  uint64_t server_generations = 0;
  uint64_t client_generations = 0;

  auto server_thread = std::async([&]() {
    SecureSiluSession session(lctxs[0], kServerId, config);
    first_large_server_result = session.Evaluate(large_server);
    small_server_result = session.Evaluate(small_server);
    second_large_server_result = session.Evaluate(large_server);
    server_initializations = session.GetInitializationCount();
    server_generations = session.GetGenerationCount();
  });
  auto client_thread = std::async([&]() {
    SecureSiluSession session(lctxs[1], kClientId, config);
    first_large_client_result = session.Evaluate(large_client);
    small_client_result = session.Evaluate(small_client);
    second_large_client_result = session.Evaluate(large_client);
    client_initializations = session.GetInitializationCount();
    client_generations = session.GetGenerationCount();
  });
  server_thread.get();
  client_thread.get();

  auto expect_close = [&](const std::vector<uint64_t>& server_result,
                          const std::vector<uint64_t>& client_result,
                          const std::vector<int64_t>& values) {
    ASSERT_EQ(server_result.size(), values.size());
    ASSERT_EQ(client_result.size(), values.size());
    for (size_t index = 0; index < values.size(); ++index) {
      const int64_t actual = operator_::GetSignedValue(
          (server_result[index] + client_result[index]) & mod_mask,
          config.io_bw);
      const double input = FromFixedPoint(values[index], config.io_scale_bw);
      const int64_t expected = static_cast<int64_t>(
          ToFixedPoint(ReferenceSiluPiecewise(input), config.io_scale_bw));
      EXPECT_LT(std::abs(actual - expected), 200) << "index=" << index;
    }
  };
  expect_close(first_large_server_result, first_large_client_result,
               large_values);
  expect_close(small_server_result, small_client_result, small_values);
  expect_close(second_large_server_result, second_large_client_result,
               large_values);
  EXPECT_EQ(server_initializations, 2);
  EXPECT_EQ(client_initializations, 2);
  EXPECT_EQ(server_generations, 3);
  EXPECT_EQ(client_generations, 3);
}

TEST(SecureSiluSessionTest, PadsInputsWithoutEmptyWorkerSlices) {
  SiluConfig config;
  config.thread_num = 2;
  EXPECT_EQ(PaddedInputSize(1, config), 16);
  EXPECT_EQ(PaddedInputSize(16, config), 16);
  EXPECT_EQ(PaddedInputSize(17, config), 32);
  EXPECT_ANY_THROW(PaddedInputSize(0, config));
  EXPECT_ANY_THROW(PaddedInputSize(std::numeric_limits<size_t>::max(), config));
  config.thread_num = 0;
  EXPECT_ANY_THROW(PaddedInputSize(1, config));
}

TEST(SecureSiluSessionTest, RejectsInvalidPreparationBeforeCommunication) {
  auto lctxs = capricorn::SetupBrpcWorld("secure_silu_invalid_prepare_test", 2);
  SiluConfig config;
  config.thread_num = 2;
  SecureSiluSession session(lctxs[0], kServerId, config);
  EXPECT_ANY_THROW(session.Online(1, std::vector<uint64_t>(16)));
  EXPECT_ANY_THROW(session.Prepare(0, 1));
  EXPECT_ANY_THROW(session.Prepare(1, 0));
  EXPECT_ANY_THROW(session.Prepare(1, std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(session.is_prepared());
  EXPECT_EQ(session.GetInitializationCount(), 0);
  EXPECT_EQ(session.GetGenerationCount(), 0);
  EXPECT_EQ(session.queries_used(), 0);
}

TEST(SecureSiluSessionTest, PreparedQueriesUseOneBatchAndEnforceBudget) {
  auto lctxs = capricorn::SetupBrpcWorld("secure_silu_prepared_test", 2);
  SiluConfig config;
  config.thread_num = 2;
  constexpr size_t logical_size = 17;
  constexpr size_t query_count = 2;
  const size_t padded_size = PaddedInputSize(logical_size, config);

  std::vector<std::vector<uint64_t>> server_inputs(
      query_count, std::vector<uint64_t>(padded_size));
  std::vector<std::vector<uint64_t>> client_inputs = server_inputs;
  std::vector<std::vector<uint64_t>> server_outputs(query_count);
  std::vector<std::vector<uint64_t>> client_outputs(query_count);
  emp::PRG prg;
  for (size_t query = 0; query < query_count; ++query) {
    prg.random_data(client_inputs[query].data(),
                    padded_size * sizeof(uint64_t));
    for (size_t index = 0; index < padded_size; ++index) {
      const double value =
          index < logical_size
              ? static_cast<double>(static_cast<int>((index + query) % 9) - 4)
              : 0.0;
      server_inputs[query][index] =
          ToFixedPoint(value, config.io_scale_bw) - client_inputs[query][index];
    }
  }

  auto run_party = [&](size_t rank, uint32_t party_id,
                       const std::vector<std::vector<uint64_t>>& inputs,
                       std::vector<std::vector<uint64_t>>& outputs) {
    SecureSiluSession session(lctxs[rank], party_id, config,
                              "prepared-session-test");
    session.Prepare(logical_size, query_count);
    EXPECT_TRUE(session.is_prepared());
    EXPECT_EQ(session.logical_input_size(), logical_size);
    EXPECT_EQ(session.padded_input_size(), padded_size);
    EXPECT_EQ(session.query_count(), query_count);
    EXPECT_ANY_THROW(session.Prepare(logical_size, query_count));
    EXPECT_ANY_THROW(session.Evaluate(inputs[0]));
    EXPECT_ANY_THROW(session.Online(logical_size + 1, inputs[0]));
    EXPECT_ANY_THROW(
        session.Online(logical_size, std::vector<uint64_t>(logical_size)));
    EXPECT_EQ(session.queries_used(), 0);
    for (size_t query = 0; query < query_count; ++query) {
      outputs[query] = session.Online(logical_size, inputs[query]);
      EXPECT_EQ(session.queries_used(), query + 1);
    }
    EXPECT_ANY_THROW(session.Online(logical_size, inputs[0]));
    EXPECT_ANY_THROW(session.Evaluate(inputs[0]));
    EXPECT_EQ(session.queries_used(), query_count);
    EXPECT_EQ(session.GetInitializationCount(), 1);
    EXPECT_EQ(session.GetGenerationCount(), 1);
  };
  auto server = std::async(std::launch::async, [&]() {
    run_party(0, kServerId, server_inputs, server_outputs);
  });
  auto client = std::async(std::launch::async, [&]() {
    run_party(1, kClientId, client_inputs, client_outputs);
  });
  server.get();
  client.get();

  for (size_t query = 0; query < query_count; ++query) {
    ASSERT_EQ(server_outputs[query].size(), padded_size);
    ASSERT_EQ(client_outputs[query].size(), padded_size);
    for (size_t index = 0; index < padded_size; ++index) {
      const double value =
          index < logical_size
              ? static_cast<double>(static_cast<int>((index + query) % 9) - 4)
              : 0.0;
      const int64_t actual = static_cast<int64_t>(server_outputs[query][index] +
                                                  client_outputs[query][index]);
      const int64_t expected = static_cast<int64_t>(
          ToFixedPoint(ReferenceSiluPiecewise(value), config.io_scale_bw));
      EXPECT_LT(std::abs(actual - expected), 200)
          << "query=" << query << " index=" << index;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CapricornDynamic, SecureSiluTest,
    testing::Values(SecureSiluTestParams{"capricorn_dynamic", true, 64, 18, 40,
                                         8, 18, 14, 20, 12, 18, 16, 1 << 18},
                    SecureSiluTestParams{"capricorn_dynamic", true, 64, 18, 40,
                                         8, 18, 14, 20, 12, 18, 16, 1 << 20},
                    SecureSiluTestParams{"capricorn_dynamic", true, 64, 18, 40,
                                         8, 18, 14, 20, 12, 18, 16, 1 << 22}));

INSTANTIATE_TEST_SUITE_P(
    FixedBitwidthQ18, SecureSiluTest,
    testing::Values(
        SecureSiluTestParams{"fixed_bitwidth_q18", false, 64, 18, 40, 8, 18, 40,
                             40, 18, 18, 16, 1 << 18},
        SecureSiluTestParams{"fixed_bitwidth_q18", false, 64, 18, 40, 8, 18, 40,
                             40, 18, 18, 16, 1 << 20},
        SecureSiluTestParams{"fixed_bitwidth_q18", false, 64, 18, 40, 8, 18, 40,
                             40, 18, 18, 16, 1 << 22}));

// Paper Table 3 sizes. These are intentionally disabled for normal unit-test
// runs and are enabled only by the isolated LAN/WAN benchmark runner.
INSTANTIATE_TEST_SUITE_P(
    DISABLED_Table3, SecureSiluTest,
    testing::Values(SecureSiluTestParams{"capricorn_dynamic", true, 64, 18, 40,
                                         8, 18, 14, 20, 12, 18, 16, 1 << 18},
                    SecureSiluTestParams{"capricorn_dynamic", true, 64, 18, 40,
                                         8, 18, 14, 20, 12, 18, 16, 1 << 20},
                    SecureSiluTestParams{"capricorn_dynamic", true, 64, 18, 40,
                                         8, 18, 14, 20, 12, 18, 16, 1 << 22}));

}  // namespace capricorn::silu

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
