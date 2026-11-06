#pragma once

#include "absl/types/span.h"
#include "emp-tool/io/io_channel.h"
#include "src/common/runtime/link/communicator.h"

#include "yacl/base/buffer.h"
#include "yacl/link/context.h"
#include "yacl/link/link.h"

namespace capricorn::link {

class LinkIO : public emp::IOChannel<LinkIO> {
 public:
  std::shared_ptr<link::Communicator> conn_;

  constexpr static uint64_t SEND_BUFFER_SIZE = 1024 * 1024 * 1024;

  uint32_t send_op_;
  uint32_t recv_op_;

  std::vector<uint8_t> send_buffer_;
  uint64_t send_buffer_used_;

  std::vector<uint8_t> recv_buffer_;
  uint64_t recv_buffer_used_;

  bool immediate_flush_;

  LinkIO() = default;

  LinkIO(std::shared_ptr<link::Communicator> conn)
      : conn_(std::move(conn)),
        send_op_(0),
        recv_op_(0),
        send_buffer_used_(0),
        recv_buffer_used_(0),
        immediate_flush_(false) {
    send_buffer_.resize(SEND_BUFFER_SIZE);
  }

  void set_immediate_flush(bool enable) { immediate_flush_ = enable; }

  ~LinkIO() {
    try {
      flush();
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in flush: {}", e.what());
    }
  }

  void flush() {
    if (send_buffer_used_ == 0) {
      return;
    }

    conn_->sendAsync(
        conn_->nextRank(),
        absl::Span<const uint8_t>{send_buffer_.data(), send_buffer_used_},
        fmt::format("LinkIO send:{}", send_op_++));

    std::memset(send_buffer_.data(), 0, SEND_BUFFER_SIZE);
    send_buffer_used_ = 0;
  }

  void fill_recv() {
    recv_buffer_ = conn_->recv<uint8_t>(
        conn_->nextRank(), fmt::format("LinkIO recv:{}", recv_op_++));
    recv_buffer_used_ = 0;
  }

  void send_data_internal(const void* data, int len) {
    if (immediate_flush_) {
      conn_->sendAsync(
          conn_->nextRank(),
          absl::Span<const uint8_t>{static_cast<const uint8_t*>(data),
                                    static_cast<size_t>(len)},
          fmt::format("LinkIO send:{}", send_op_++));
      return;
    }

    size_t send_buffer_left = SEND_BUFFER_SIZE - send_buffer_used_;
    if (send_buffer_left <= static_cast<size_t>(len)) {
      std::memcpy(send_buffer_.data() + send_buffer_used_, data,
                  send_buffer_left);
      send_buffer_used_ += send_buffer_left;
      flush();

      send_data_internal(static_cast<const char*>(data) + send_buffer_left,
                         len - send_buffer_left);
    } else {
      std::memcpy(send_buffer_.data() + send_buffer_used_, data, len);
      send_buffer_used_ += len;
    }
  }

  void recv_data_internal(void* data, int len) {
    if (send_buffer_used_ > 0) {
      flush();
    }

    size_t recv_buffer_left = recv_buffer_.size() - recv_buffer_used_;
    if (recv_buffer_left >= static_cast<size_t>(len)) {
      std::memcpy(data, recv_buffer_.data() + recv_buffer_used_, len);
      recv_buffer_used_ += len;
    } else {
      if (recv_buffer_.size() != 0) {
        std::memcpy(data, recv_buffer_.data() + recv_buffer_used_,
                    recv_buffer_left);
      }
      fill_recv();

      recv_data_internal(static_cast<char*>(data) + recv_buffer_left,
                         len - recv_buffer_left);
    }
  }

  void sync() {}
  std::shared_ptr<const yacl::link::Statistics> GetStats() {
    return conn_->GetStats();
  }
};

}  // namespace capricorn::link
