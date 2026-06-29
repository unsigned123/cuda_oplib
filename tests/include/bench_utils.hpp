#pragma once
/// Reusable benchmark utilities for cudaoplib.
/// All benchmarks use raw kernel calls (pre-allocated output) to exclude
/// cudaMalloc overhead. TorchRef is used for PyTorch reference timing.

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"

#include <vector>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>

namespace cudaoplib_bench {

// ── helpers ───────────────────────────────────────────────────

inline bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

/// Time a raw kernel launch (pre-allocated output, NO cudaMalloc inside).
/// kernel_launch() is called once before timing (JIT warm-up), then N times.
/// Returns average milliseconds per call (GPU time via CUDA events).
template <typename F>
float time_raw_kernel_ms(F&& kernel_launch, int warmup = 1, int iters = 1)
{
    // JIT warm-up
    for (int i = 0; i < warmup; ++i)
        kernel_launch();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        kernel_launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

/// Time a torch op via TorchRef (CPU timer, GPU-sync'd, repeat N times).
/// Returns average milliseconds per call.
template <cudaoplib::SupportedDType T>
float time_torch_ms(const cudaoplib::Tensor<T>& a_cpu,
                    const cudaoplib::Tensor<T>& b_cpu,
                    const std::string& /*op_name*/, int iters)
{
    auto& ref = cudaoplib_test::TorchRef::instance();
    if (!ref.is_alive())
        throw std::runtime_error("TorchRef not alive");
    ref.set_repeat(iters);
    ref.add(a_cpu, b_cpu);  // warm-up + timed (repeat handled inside)
    float total_ms = ref.last_time_us() / 1000.0f;
    ref.set_repeat(1);
    return total_ms / iters;
}

// ── result struct ──────────────────────────────────────────────

struct BenchResult
{
    std::string op;
    std::string dtype;
    std::string shape_desc;   // e.g. "1024x1024"
    size_t numel;
    float    our_ms = 0;      // cudaoplib raw kernel (CUDA events)
    float    torch_ms = 0;    // PyTorch via TorchRef
};

// ── table printer ─────────────────────────────────────────────

inline void print_bench_table(const std::vector<BenchResult>& results)
{
    printf("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Performance: cudaoplib vs PyTorch  (raw kernel only, no alloc overhead)    ║\n");
    printf("╠══════════╤══════╤═══════════════╤══════════╤══════════╤══════════╤════════╣\n");
    printf("║ op       │ dtype│ shape         │ numel    │ our (ms) │ torch    │ ratio  ║\n");
    printf("║          │      │               │          │          │ (ms)     │ o/t    ║\n");
    printf("╠══════════╪══════╪═══════════════╪══════════╪══════════╪══════════╪════════╣\n");
    for (auto& r : results)
    {
        float ratio = (r.torch_ms > 0) ? r.our_ms / r.torch_ms : 0;
        printf("║ %-8s │ %-4s │ %-13s │ %8zu │ %8.4f │ %8.4f │ %6.2f ║\n",
               r.op.c_str(), r.dtype.c_str(), r.shape_desc.c_str(),
               r.numel, r.our_ms, r.torch_ms, ratio);
    }
    printf("╚══════════╧══════╧═══════════════╧══════════╧══════════╧══════════╧════════╝\n\n");
}

// ── utility: shape to string ──────────────────────────────────

inline std::string shape_to_string(const cudaoplib::TensorShape& s)
{
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0) r += "x";
        r += std::to_string(s[i]);
    }
    return r;
}

// ── element-wise add helper (pre-allocated output) ────────────

template <cudaoplib::SupportedDType T>
BenchResult bench_add(const cudaoplib::TensorShape& shape, int warmup = 1, int iters = 100,
                       const std::string& shape_desc = "")
{
    using namespace cudaoplib;
    BenchResult r;
    r.op = "add";
    r.dtype = cudaoplib_test::dtype_to_str(dtype_to_enum(T{}));
    r.shape_desc = shape_desc.empty() ? shape_to_string(shape) : shape_desc;
    r.numel = numel_from_shape(shape);

    auto a = ones<T>(shape), b = ones<T>(shape);
    auto ag = a.to_gpu(), bg = b.to_gpu();
    auto araw = ag.get_raw(), braw = bg.get_raw();
    auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
        dtype_to_enum(T{}), shape);

    // Measure torch FIRST (cooler GPU), then ours (warmer GPU).
    // This way if we still win, it's real — not thermal advantage.
    r.torch_ms = time_torch_ms<T>(a, b, "add", iters);
    r.our_ms = time_raw_kernel_ms([&]{ cudaoplib_kernel::add(araw, braw, out_raw); },
                                   warmup, iters);

    cudaoplib_kernel::free_gpu_tensor(out_raw);
    return r;
}

// ── element-wise mul helper ───────────────────────────────────

template <cudaoplib::SupportedDType T>
BenchResult bench_mul(const cudaoplib::TensorShape& shape, int warmup = 1, int iters = 100,
                       const std::string& shape_desc = "")
{
    using namespace cudaoplib;
    BenchResult r;
    r.op = "mul";
    r.dtype = cudaoplib_test::dtype_to_str(dtype_to_enum(T{}));
    r.shape_desc = shape_desc.empty() ? shape_to_string(shape) : shape_desc;
    r.numel = numel_from_shape(shape);

    auto a = ones<T>(shape), b = ones<T>(shape);
    auto ag = a.to_gpu(), bg = b.to_gpu();
    auto araw = ag.get_raw(), braw = bg.get_raw();
    auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
        dtype_to_enum(T{}), shape);

    r.torch_ms = time_torch_ms<T>(a, b, "mul", iters);
    r.our_ms = time_raw_kernel_ms([&]{ cudaoplib_kernel::mul(araw, braw, out_raw); },
                                   warmup, iters);

    cudaoplib_kernel::free_gpu_tensor(out_raw);
    return r;
}

// ── element-wise add with broadcast ───────────────────────────

template <cudaoplib::SupportedDType T>
BenchResult bench_add_broadcast(const cudaoplib::TensorShape& a_shape, const cudaoplib::TensorShape& b_shape,
                                 int warmup = 1, int iters = 100,
                                 const std::string& shape_desc = "")
{
    using namespace cudaoplib;
    BenchResult r;
    r.op = "add_br";
    r.dtype = cudaoplib_test::dtype_to_str(dtype_to_enum(T{}));
    r.shape_desc = shape_desc.empty()
        ? (shape_to_string(a_shape) + "+" + shape_to_string(b_shape)) : shape_desc;
    r.numel = 0;

    auto a = ones<T>(a_shape), b = ones<T>(b_shape);
    auto ag = a.to_gpu(), bg = b.to_gpu();
    auto araw = ag.get_raw(), braw = bg.get_raw();

    size_t ad = a_shape.size(), bd = b_shape.size(), td = std::max(ad, bd);
    std::vector<size_t> out_shape(td, 1);
    for (size_t i = 0; i < td; ++i) {
        size_t av = (i < ad) ? a_shape[ad - 1 - i] : 1;
        size_t bv = (i < bd) ? b_shape[bd - 1 - i] : 1;
        out_shape[td - 1 - i] = std::max(av, bv);
    }
    r.numel = numel_from_shape(out_shape);
    auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
        dtype_to_enum(T{}), out_shape);

    r.torch_ms = time_torch_ms<T>(a, b, "add", iters);
    r.our_ms = time_raw_kernel_ms([&]{ cudaoplib_kernel::add(araw, braw, out_raw); },
                                   warmup, iters);

    cudaoplib_kernel::free_gpu_tensor(out_raw);
    return r;
}

/// Torch sum timing (handles dim parameter differently from binary ops)
template <cudaoplib::SupportedDType T>
float time_torch_sum_ms(const cudaoplib::Tensor<T>& a_cpu, int dim, int iters)
{
    auto& ref = cudaoplib_test::TorchRef::instance();
    if (!ref.is_alive())
        throw std::runtime_error("TorchRef not alive");
    ref.set_repeat(iters);
    ref.sum(a_cpu, dim);
    float total_ms = ref.last_time_us() / 1000.0f;
    ref.set_repeat(1);
    return total_ms / iters;
}

// ── reduce sum ───────────────────────────────────────────────

template <cudaoplib::SupportedDType T>
BenchResult bench_sum(const cudaoplib::TensorShape& shape, int dim,
                       int warmup = 1, int iters = 100,
                       const std::string& shape_desc = "")
{
    using namespace cudaoplib;
    BenchResult r;
    r.op = "sum";
    r.dtype = cudaoplib_test::dtype_to_str(dtype_to_enum(T{}));
    r.shape_desc = shape_desc.empty() ? shape_to_string(shape) : shape_desc;

    auto a = ones<T>(shape);
    auto ag = a.to_gpu();
    auto araw = ag.get_raw();

    // compute output shape
    cudaoplib::TensorShape out_shape = shape;
    out_shape[dim] = 1;
    r.numel = numel_from_shape(out_shape);
    auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
        dtype_to_enum(T{}), out_shape);

    r.our_ms = time_raw_kernel_ms(
        [&]{ cudaoplib_kernel::sum(araw, out_raw, dim); },
        warmup, iters);
    r.torch_ms = time_torch_sum_ms<T>(a, dim, iters);

    cudaoplib_kernel::free_gpu_tensor(out_raw);
    return r;
}

} // namespace cudaoplib_bench
