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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "nlohmann/json.hpp"
#include "src/adapters/vmoe/inference_artifact.h"
#include "src/adapters/vmoe/metrics.h"
#include "src/adapters/vmoe/pphlo_runner.h"
#include "src/common/runtime/environment.h"
#include "xtensor/containers/xadapt.hpp"

#include "yacl/base/exception.h"
#include "yacl/link/factory.h"

namespace capricorn::vmoe {
namespace {

struct RunnerFlags {
  int role = -1;
  std::string parties = "127.0.0.1:10086,127.0.0.1:10087";
  std::string artifact_dir;
  std::string output;
  bool compile_only = false;
  ExpertTopTMode expert_top_t_mode = ExpertTopTMode::SerialQuickSelect;
};

struct OutputSpec {
  std::string name;
  std::vector<int64_t> shape;
};

RunnerFlags ParseFlags(int argc, char** argv) {
  RunnerFlags flags;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto separator = argument.find('=');
    YACL_ENFORCE(argument.rfind("--", 0) == 0 && separator != std::string::npos,
                 "Expected --key=value, got {}", argument);
    const auto key = argument.substr(2, separator - 2);
    const auto value = argument.substr(separator + 1);
    if (key == "role") {
      YACL_ENFORCE(value == "0" || value == "1",
                   "--role must be 0 or 1, got {}", value);
      flags.role = value == "0" ? 0 : 1;
    } else if (key == "parties") {
      flags.parties = value;
    } else if (key == "artifact_dir") {
      flags.artifact_dir = value;
    } else if (key == "output") {
      flags.output = value;
    } else if (key == "compile_only") {
      YACL_ENFORCE(value == "0" || value == "1",
                   "--compile_only must be 0 or 1, got {}", value);
      flags.compile_only = value == "1";
    } else if (key == "expert_top_t_mode") {
      flags.expert_top_t_mode = ParseExpertTopTMode(value);
    } else {
      YACL_THROW("Unknown flag --{}", key);
    }
  }
  return flags;
}

std::shared_ptr<yacl::link::Context> MakeLink(std::string_view parties,
                                              size_t rank) {
  const std::vector<std::string> hosts = absl::StrSplit(parties, ',');
  YACL_ENFORCE(hosts.size() == 2, "Exactly two parties are required, got {}",
               hosts.size());
  yacl::link::ContextDesc description;
  for (size_t index = 0; index < hosts.size(); ++index) {
    YACL_ENFORCE(!hosts[index].empty(), "Party address must not be empty");
    description.parties.emplace_back("party" + std::to_string(index),
                                     hosts[index]);
  }
  description.throttle_window_size = 0;
  auto link = yacl::link::FactoryBrpc().CreateContext(description, rank);
  link->ConnectToMesh();
  return link;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  YACL_ENFORCE(input.is_open(), "Unable to open {}", path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<OutputSpec> LoadOutputSpecs(
    const std::filesystem::path& artifact_dir) {
  const auto output_order = artifact_dir / "output_order.json";
  if (!std::filesystem::exists(output_order)) {
    return {{"vmoe_output", LoadExpectedOutputShape(artifact_dir)}};
  }
  const auto parsed = nlohmann::json::parse(ReadText(output_order));
  YACL_ENFORCE(parsed.contains("outputs") && parsed["outputs"].is_array() &&
                   !parsed["outputs"].empty(),
               "{} must contain a non-empty outputs array",
               output_order.string());
  std::vector<OutputSpec> outputs;
  for (const auto& entry : parsed["outputs"]) {
    YACL_ENFORCE(entry.contains("name") && entry["name"].is_string(),
                 "output entry requires a string name");
    YACL_ENFORCE(entry.contains("shape") && entry["shape"].is_array(),
                 "output entry requires a shape array");
    outputs.push_back({entry["name"].get<std::string>(),
                       entry["shape"].get<std::vector<int64_t>>()});
    YACL_ENFORCE(!outputs.back().name.empty() && !outputs.back().shape.empty(),
                 "output name and shape must be non-empty");
  }
  return outputs;
}

void Infeed(Environment* environment, const ManifestEntry& entry, size_t rank) {
  YACL_ENFORCE(environment != nullptr, "Environment must not be null");
  if (entry.visibility == InputVisibility::kPublic) {
    auto tensor = LoadFloat32Npy(entry.npy_path);
    auto adapted = xt::adapt(tensor.values, tensor.shape);
    environment->SetPublic(entry.symbol_name, adapted);
    return;
  }
  if (entry.visibility == InputVisibility::kSecret) {
    auto tensor = LoadFloat32Npy(entry.npy_path);
    if (rank != entry.owner_rank) {
      std::fill(tensor.values.begin(), tensor.values.end(), 0.0F);
    }
    auto adapted = xt::adapt(tensor.values, tensor.shape);
    environment->FeedShare(entry.symbol_name, adapted);
    return;
  }

  if (rank == entry.owner_rank) {
    auto tensor = LoadFloat32Npy(entry.npy_path);
    auto adapted = xt::adapt(tensor.values, tensor.shape);
    environment->SetPrivate(entry.symbol_name, adapted);
  } else {
    environment->GetPrivate(entry.symbol_name);
  }
}

void Run(const RunnerFlags& flags) {
  YACL_ENFORCE(flags.role == 0 || flags.role == 1,
               "--role must be 0 or 1, got {}", flags.role);
  YACL_ENFORCE(!flags.artifact_dir.empty(), "--artifact_dir is required");
  YACL_ENFORCE(flags.compile_only || !flags.output.empty(),
               "--output is required unless --compile_only=1");

  const auto rank = static_cast<size_t>(flags.role);
  const std::filesystem::path artifact_dir(flags.artifact_dir);
  const auto entries = LoadInputManifest(artifact_dir);
  const auto outputs = LoadOutputSpecs(artifact_dir);
  const auto stablehlo = ReadText(artifact_dir / "model.stablehlo.mlir");
  std::vector<std::string> input_names;
  std::vector<spu::Visibility> input_visibilities;
  input_names.reserve(entries.size());
  input_visibilities.reserve(entries.size());
  for (const auto& entry : entries) {
    input_names.push_back(entry.symbol_name);
    if (entry.visibility == InputVisibility::kPublic) {
      input_visibilities.push_back(spu::Visibility::VIS_PUBLIC);
    } else if (entry.visibility == InputVisibility::kSecret) {
      input_visibilities.push_back(spu::Visibility::VIS_SECRET);
    } else {
      input_visibilities.push_back(spu::Visibility::VIS_PRIVATE);
    }
  }
  std::vector<std::string> output_names;
  output_names.reserve(outputs.size());
  for (const auto& output : outputs) {
    output_names.push_back(output.name);
  }
  const auto executable = [&] {
    StageMetric metric(nullptr, "runner", "compile_stablehlo", rank);
    return CompileCapricornStablehlo(stablehlo, input_names, input_visibilities,
                                     output_names);
  }();
  if (std::getenv("CAPRICORN_DUMP_PPHLO") != nullptr) {
    std::cerr << executable.code() << std::endl;
  }
  if (flags.compile_only) {
    return;
  }

  auto link = MakeLink(flags.parties, rank);
  StageMetric total_metric(link, "runner", "total_after_link");
  Environment environment(link);
  {
    StageMetric metric(link, "runner", "infeed_all_operands");
    for (const auto& entry : entries) {
      Infeed(&environment, entry, rank);
    }
  }
  {
    StageMetric metric(link, "runner", "execute_pphlo");
    ExecuteVmoePphlo(environment.spu_ctx_.get(), executable,
                     &environment.cio_->deviceSymbols(),
                     flags.expert_top_t_mode);
  }
  for (const auto& output_spec : outputs) {
    const auto output = environment.cio_->deviceGetVar(output_spec.name);
    YACL_ENFORCE(output.isSecret(), "{} output must remain secret",
                 output_spec.name);
    const spu::Shape expected_shape(output_spec.shape.begin(),
                                    output_spec.shape.end());
    YACL_ENFORCE(output.shape() == expected_shape,
                 "{} output must have shape {}, got {}", output_spec.name,
                 expected_shape, output.shape());
    spu::Value revealed;
    {
      StageMetric metric(link, "runner", "reveal_" + output_spec.name);
      revealed = spu::kernel::hal::reveal(environment.spu_ctx_.get(), output);
    }
    const auto decoded = spu::kernel::hal::dump_public_as<double>(
        environment.spu_ctx_.get(), revealed);
    std::vector<float> values;
    values.reserve(decoded.size());
    std::transform(decoded.begin(), decoded.end(), std::back_inserter(values),
                   [](double value) { return static_cast<float>(value); });
    const auto output_path =
        outputs.size() == 1 ? std::filesystem::path(flags.output)
                            : std::filesystem::path(flags.output + "." +
                                                    output_spec.name + ".npy");
    {
      StageMetric metric(nullptr, "runner", "save_" + output_spec.name, rank);
      SaveFloat32Npy(output_path, output_spec.shape, values);
    }
  }
}

}  // namespace
}  // namespace capricorn::vmoe

int main(int argc, char** argv) {
  capricorn::vmoe::Run(capricorn::vmoe::ParseFlags(argc, argv));
  return 0;
}
