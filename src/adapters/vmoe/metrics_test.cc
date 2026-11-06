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

#include "src/adapters/vmoe/metrics.h"

#include "gtest/gtest.h"

namespace capricorn::vmoe {
namespace {

TEST(ProcessMemorySnapshotTest, ReportsCurrentAndPeakRssInBytes) {
  const auto snapshot = ReadProcessMemorySnapshot();

  EXPECT_GT(snapshot.current_rss_bytes, 0);
  EXPECT_GT(snapshot.peak_rss_bytes, 0);
}

TEST(ModelStageRecorderTest, AttributesPendingOperationsToCompletedMarker) {
  ModelStageRecorder recorder(nullptr);
  recorder.RecordOperation("pphlo.multiply", 3.5,
                           LinkMetricSnapshot{10, 1, 20, 2});

  const auto completed = recorder.Boundary("encoder.block_0.attention_norm");

  ASSERT_TRUE(completed.has_value());
  EXPECT_EQ(completed->stage, "encoder.block_0.attention_norm");
  EXPECT_EQ(completed->operation_count, 1);
  EXPECT_DOUBLE_EQ(completed->operation_elapsed_ms, 3.5);
}

}  // namespace
}  // namespace capricorn::vmoe
