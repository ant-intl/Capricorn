#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"

#include "yacl/base/buffer.h"
#include "yacl/link/context.h"

namespace capricorn::link {

class Communicator {
 public:
  explicit Communicator(std::shared_ptr<yacl::link::Context> lctx)
      : lctx_(std::move(lctx)) {}

  explicit Communicator(std::shared_ptr<yacl::link::Context> lctx,
                        uint64_t recv_timeout_ms)
      : lctx_(std::move(lctx)) {
    lctx_->SetRecvTimeout(recv_timeout_ms);
  }

  size_t nextRank() const { return lctx_->NextRank(); }

  // Asynchronously send data to the specified rank
  template <typename T>
  void sendAsync(size_t dst_rank, absl::Span<T const> in, std::string_view tag);

  // Receive data from the specified rank
  template <typename T>
  std::vector<T> recv(size_t src_rank, std::string_view tag);
  std::shared_ptr<const yacl::link::Statistics> GetStats() {
    return lctx_->GetStats();
  }

 private:
  std::shared_ptr<yacl::link::Context> lctx_;
};

template <typename T>
void Communicator::sendAsync(size_t dst_rank, absl::Span<T const> in,
                             std::string_view tag) {
  yacl::ByteContainerView bv(reinterpret_cast<uint8_t const*>(in.data()),
                             sizeof(T) * in.size());
  lctx_->SendAsync(dst_rank, bv, tag);
}

template <typename T>
std::vector<T> Communicator::recv(size_t src_rank, std::string_view tag) {
  auto buf = lctx_->Recv(src_rank, tag);
  assert(buf.size() % sizeof(T) == 0);
  auto numel = buf.size() / sizeof(T);
  return std::vector<T>(buf.data<T>(), buf.data<T>() + numel);
}

}  // namespace capricorn::link
