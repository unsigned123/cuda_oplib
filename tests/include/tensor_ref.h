#pragma once

#include "tensor.h"

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <expected>
#include <stdexcept>
#include <memory>

namespace cudaoplib_test
{

// ── Protocol ──────────────────────────────────────────────────
//
// Request (C++ → Python):
//   OP:<name>\n
//   NTENSORS:<N>\n
//   \n
//   DTYPE:<dtype>\n NUMEL:<n>\n NDIM:<d>\n SHAPE:<s0,s1,...>\n
//   SIZE:<bytes>\n
//   \n
//   <binary data, SIZE bytes>
//   ... (repeat NTENSORS times)
//
// Response (Python → C++):
//   STATUS:OK\n or STATUS:ERR\n
//   MSG:<error_msg>\n   (only if ERR)
//   DTYPE:<dtype>\n NUMEL:<n>\n NDIM:<d>\n SHAPE:<s0,s1,...>\n
//   TIME_US:<us>\n SIZE:<bytes>\n
//   \n
//   <binary data, SIZE bytes>
// ───────────────────────────────────────────────────────────────

// dtype  -> protocol string
inline const char* dtype_to_str(cudaoplib_kernel::DType dtype)
{
    switch (dtype)
    {
        case cudaoplib_kernel::DType::Float32: return "float32";
        case cudaoplib_kernel::DType::Float16: return "float16";
        case cudaoplib_kernel::DType::Int32:   return "int32";
        case cudaoplib_kernel::DType::Int8:    return "int8";
        case cudaoplib_kernel::DType::Bool:    return "bool";
        default: throw std::runtime_error("Unknown dtype");
    }
}

// protocol string -> dtype
inline cudaoplib_kernel::DType str_to_dtype(const std::string& s)
{
    if (s == "float32") return cudaoplib_kernel::DType::Float32;
    if (s == "float16") return cudaoplib_kernel::DType::Float16;
    if (s == "int32")   return cudaoplib_kernel::DType::Int32;
    if (s == "int8")    return cudaoplib_kernel::DType::Int8;
    if (s == "bool")    return cudaoplib_kernel::DType::Bool;
    throw std::runtime_error("Unknown dtype string: " + s);
}

// Parsed tensor metadata (used on both send and receive sides)
struct TensorMeta
{
    cudaoplib_kernel::DType dtype;
    size_t numel;
    std::vector<size_t> shape;
    size_t data_size;  // numel * elem_size
};

// Result of a torch reference call
struct TorchRefResult
{
    cudaoplib_kernel::Tensor raw;
    uint64_t time_us = 0;
};

// ── TorchPipe: manages a popen'd Python worker ───────────────

class TorchPipe
{
public:
    TorchPipe(const std::string& worker_script);
    ~TorchPipe();

    // Send an operation request. Returns false on pipe error.
    bool send(const std::string& op_name,
              const std::vector<cudaoplib_kernel::Tensor>& tensors);

    // Receive the result. Returns TorchRefResult on success,
    // error message string on failure.
    std::expected<TorchRefResult, std::string> recv();

    bool is_alive() const { return child_pid_ > 0 && pipe_alive_; }

private:
    int to_child_fd_   = -1;  // C++ → Python  (write)
    int from_child_fd_ = -1;  // Python → C++  (read)
    pid_t child_pid_   = 0;
    bool  pipe_alive_  = false;

    // Low-level I/O
    bool write_line(int fd, const std::string& line);
    bool write_exact(int fd, const void* data, size_t len);
    std::expected<std::string, bool> read_line(int fd);
    std::expected<std::vector<uint8_t>, bool> read_exact(int fd, size_t len);

    // Helpers
    static TensorMeta parse_tensor_header(const std::string& dtype_str,
                                          const std::string& numel_str,
                                          const std::string& ndim_str,
                                          const std::string& shape_str,
                                          const std::string& size_str);
    void write_tensor_meta(int fd, const TensorMeta& meta);

    static constexpr int kLineBufferSize = 256;
};

// ── TorchRef: singleton managing a persistent Python worker ───

class TorchRef
{
public:
    static TorchRef& instance();
    bool is_alive() const { return pipe_ && pipe_->is_alive(); }

    // Low-level: send raw kernel tensors, receive result
    std::expected<TorchRefResult, std::string>
    run_op(const std::string& op_name,
           const std::vector<cudaoplib_kernel::Tensor>& tensors);

    // High-level: per-operator, type-safe, returns Tensor<T>
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<T> add(const cudaoplib::Tensor<T>& a,
                             const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<T> sub(const cudaoplib::Tensor<T>& a,
                             const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<T> mul(const cudaoplib::Tensor<T>& a,
                             const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<T> div(const cudaoplib::Tensor<T>& a,
                             const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<T> mod(const cudaoplib::Tensor<T>& a,
                             const cudaoplib::Tensor<T>& b);

    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> eq(const cudaoplib::Tensor<T>& a,
                               const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> neq(const cudaoplib::Tensor<T>& a,
                                const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> lt(const cudaoplib::Tensor<T>& a,
                               const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> le(const cudaoplib::Tensor<T>& a,
                               const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> gt(const cudaoplib::Tensor<T>& a,
                               const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> ge(const cudaoplib::Tensor<T>& a,
                               const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> logical_and(const cudaoplib::Tensor<T>& a,
                                         const cudaoplib::Tensor<T>& b);
    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<bool> logical_or(const cudaoplib::Tensor<T>& a,
                                        const cudaoplib::Tensor<T>& b);

    // Elapsed time of the last op (microseconds)
    uint64_t last_time_us() const { return last_time_us_; }

private:
    TorchRef();
    ~TorchRef();

    std::unique_ptr<TorchPipe> pipe_;
    uint64_t last_time_us_ = 0;

    template <cudaoplib::SupportedDType T>
    cudaoplib::Tensor<T> _run_binary(const std::string& op,
                                      const cudaoplib::Tensor<T>& a,
                                      const cudaoplib::Tensor<T>& b);
};

// Include template implementations (header-only)
#include "torch_ref_ops.hpp"

} // namespace cudaoplib_test
