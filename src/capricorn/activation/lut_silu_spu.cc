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

#include "src/capricorn/activation/lut_silu_spu.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "libspu/core/encoding.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "src/capricorn/activation/secure_silu.h"

namespace capricorn::customized {

spu::Value CapricornLutSiLU(spu::SPUContext* spu_ctx, const spu::Value& input,
                            const uint32_t party_id,
                            silu::SecureSiluSession* session) {
  using spu::Ring2kTrait;  // to allow DISPATCH_ALL_FIELDS

  auto lctx = spu_ctx->lctx();
  auto field = spu_ctx->getField();
  auto numel = input.numel();
  YACL_ENFORCE_GT(numel, 0);
  const silu::SiluConfig default_config;
  const auto& silu_config = session ? session->config() : default_config;
  const size_t padded_numel =
      silu::PaddedInputSize(static_cast<size_t>(numel), silu_config);

  auto nd_input = input.data();
  std::vector<uint64_t> raw_input(padded_numel, 0);
  // copy to raw_input
  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    auto view_input = spu::NdArrayView<const u2k>(nd_input);
    spu::pforeach(0, numel, [&](int64_t i) { raw_input[i] = view_input[i]; });
  });

  std::vector<uint64_t> raw_output;
  if (session != nullptr && session->is_prepared()) {
    raw_output = session->Online(static_cast<size_t>(numel), raw_input);
  } else if (session != nullptr) {
    raw_output = session->Evaluate(raw_input);
  } else {
    raw_output = silu::SecureSilu(lctx, raw_input, party_id, silu_config);
  }
  raw_output.resize(numel);

  // fill back to output
  auto output = input.clone();
  auto nd_output = output.data();
  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    auto view_output = spu::NdArrayView<u2k>(nd_output);
    spu::pforeach(0, numel, [&](int64_t i) { view_output[i] = raw_output[i]; });
  });

  return output;
}

spu::Value CapricornLutSiLU(
    spu::SPUContext* spu_ctx, const spu::Value& input, uint32_t party_id,
    const std::shared_ptr<silu::SecureSiluSession>& session) {
  return CapricornLutSiLU(spu_ctx, input, party_id, session.get());
}

}  // namespace capricorn::customized
