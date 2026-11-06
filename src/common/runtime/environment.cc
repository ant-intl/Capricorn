#include "src/common/runtime/environment.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "libspu/core/config.h"
#include "libspu/core/encoding.h"
#include "libspu/core/half.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/device/io.h"
#include "libspu/kernel/hal/constants.h"  // bad reference
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/type_cast.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/basic_unary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/indexing.h"
#include "libspu/mpc/factory.h"
#include "spdlog/spdlog.h"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xcsv.hpp"
#include "xtensor/views/xview.hpp"

namespace capricorn {

namespace {

std::unique_ptr<spu::SPUContext> makeSPUContext(
    spu::RuntimeConfig config,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  populateRuntimeConfig(config);
  auto sctx_ptr = std::make_unique<spu::SPUContext>(config, lctx);
  spu::mpc::Factory::RegisterProtocol(sctx_ptr.get(), lctx);
  return sctx_ptr;
}

std::unique_ptr<spu::SPUContext> makeSPUContext(
    spu::ProtocolKind prot_kind, spu::FieldType field,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  spu::RuntimeConfig config;
  config.set_protocol(prot_kind);
  config.set_field(field);
  config.set_fxp_fraction_bits(18);
  config.set_experimental_enable_colocated_optimization(true);

  // [Warning] please do NOT use FM32, it would make the score WRONG.
  //   config.set_fxp_fraction_bits(18);
  //   config.set_fxp_div_goldschmidt_iters(2);

  return makeSPUContext(config, lctx);
}

}  // namespace

Environment::Environment(std::shared_ptr<yacl::link::Context> lctx,
                         spu::ProtocolKind prot_kind, spu::FieldType field) {
  lctx_ = lctx;
  spu_ctx_ = makeSPUContext(prot_kind, field, lctx_);
  cio_ = std::make_unique<spu::device::ColocatedIo>(spu_ctx_.get());
}

void Environment::FeedShare(const std::string name,
                            const spu::PtBufferView& bv) {
  cio_->hostSetVar(fmt::format("rank_{}_{}", spu_ctx_->lctx()->Rank(), name),
                   bv);
  cio_->sync();

  auto lambda = [&](const std::string& var_name) {
    auto rank_0_var = cio_->deviceGetVar(fmt::format("rank_0_{}", var_name));
    auto rank_1_var = cio_->deviceGetVar(fmt::format("rank_1_{}", var_name));
    return spu::kernel::hlo::Add(spu_ctx_.get(), rank_0_var, rank_1_var);
  };

  auto var = lambda(name);
  SPDLOG_INFO("[Environment] cio var name: {} shape: {} vtype: {}", name,
              var.shape(), var.vtype());

  cio_->deviceSetVar(name, var);
}

void Environment::SetShare(const std::string name, const spu::Value& var) {
  SPDLOG_INFO("[Environment] cio var name: {} shape: {} vtype: {}", name,
              var.shape(), var.vtype());
  cio_->deviceSetVar(name, var);
}

void Environment::SetPrivate(const std::string name,
                             const spu::PtBufferView& bv) {
  cio_->hostSetVar(fmt::format("{}", name), bv, spu::Visibility::VIS_PRIVATE);
  cio_->sync();

  // Reveal gate weights to the owner (Server)
  // auto lambda_a2v = [&](const std::string& var_name) {
  //   auto _var = cio_->deviceGetVar(fmt::format("__{}", var_name));
  //   SPDLOG_INFO("[Environment] cio var name: {} shape: {} vtype: {}",
  //               fmt::format("__{}", var_name), _var.shape(), _var.vtype());
  //   return spu::kernel::hlo::RevealTo(spu_ctx_.get(), _var,
  //                                     spu_ctx_->lctx()->Rank());
  // };

  // auto var = lambda_a2v(name);
  // SPDLOG_INFO("[Environment] cio var name: {} shape: {} vtype: {}", name,
  //             var.shape(), var.vtype());
  // cio_->deviceSetVar(name, var);
}

void Environment::GetPrivate(const std::string name) {
  cio_->sync();
  // auto lambda_a2v = [&](const std::string& var_name) {
  //   auto _var = cio_->deviceGetVar(fmt::format("__{}", var_name));
  //   SPDLOG_INFO("[Environment] cio var name: {} shape: {} vtype: {}",
  //               fmt::format("__{}", var_name), _var.shape(), _var.vtype());
  //   return spu::kernel::hlo::RevealTo(spu_ctx_.get(), _var,
  //                                     spu_ctx_->lctx()->NextRank());
  // };
  // auto var = lambda_a2v(name);
  // SPDLOG_INFO("[Environment] cio var name: {} shape: {} vtype: {}", name,
  //             var.shape(), var.vtype());
  // cio_->deviceSetVar(name, var);
}

void Environment::SetPublic(const std::string name,
                            const spu::PtBufferView& bv) {
  if (spu_ctx_->lctx()->Rank() == 0) {
    cio_->hostSetVar(fmt::format("{}", name), bv, spu::Visibility::VIS_PUBLIC);
  }
  cio_->sync();
}

}  // namespace capricorn
