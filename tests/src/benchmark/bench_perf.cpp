#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"
#include "elementwise.h"

#include <vector>
#include <cstdio>
#include <cuda_runtime.h>

using namespace cudaoplib;
using namespace cudaoplib_test;

static bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
#define REQUIRE_CUDA() \
    if (!cuda_available()) { GTEST_SKIP() << "No CUDA device"; }

struct BenchResult
{
    std::string dtype;
    std::string shape;
    float op_ms;        // operator+ with cudaMalloc (CUDA events)
    float raw_ms;       // raw kernel, pre-allocated (CUDA events)
    float tms;          // torch (CPU timer, sync'd)
};

// ── bench: operator+ (includes cudaMalloc) ────────────────────

template <SupportedDType T>
static float bench_op_ms(Tensor<T>& ag, Tensor<T>& bg,
                          Tensor<T> (Tensor<T>::*op)(const Tensor<T>&) const)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    [[maybe_unused]] volatile auto r = (ag.*op)(bg);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

// ── bench: raw kernel call (pre-allocated output, NO cudaMalloc) ──

static float bench_raw_add_ms(cudaoplib_kernel::Tensor& a,
                               cudaoplib_kernel::Tensor& b,
                               cudaoplib_kernel::Tensor& out)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaoplib_kernel::add(a, b, out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

static float bench_raw_mul_ms(cudaoplib_kernel::Tensor& a,
                               cudaoplib_kernel::Tensor& b,
                               cudaoplib_kernel::Tensor& out)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaoplib_kernel::mul(a, b, out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

// ── bench: torch (CPU timer + sync, repeat N times) ───────────

template <SupportedDType T>
static float bench_torch_ms(Tensor<T>& a, Tensor<T>& b, int repeat)
{
    auto& ref = TorchRef::instance();
    if (!ref.is_alive()) throw std::runtime_error("TorchRef not alive");
    ref.set_repeat(repeat);
    ref.add(a, b);
    float total_ms = ref.last_time_us() / 1000.0f;
    ref.set_repeat(1);
    return total_ms / repeat;
}

// ── benchmark body ────────────────────────────────────────────

TEST(Benchmark, LargeTensors)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    const int REPEAT = 10000;
    const int REPEAT_L = 1000000;  // 1M for stability-sensitive cases
    std::vector<BenchResult> results;

    std::cout << "\n=== Performance Benchmark (add=10k, mul/bcast=1M iters) ===" << std::endl;

    // ── warm-up ──
    std::cout << "  warming up ..." << std::endl;
    {
        auto wa = rand<float>({8, 8}).to_gpu(), wb = rand<float>({8, 8}).to_gpu();
        bench_op_ms<float>(wa, wb, &Tensor<float>::operator+);
        bench_op_ms<float>(wa, wb, &Tensor<float>::operator*);
        auto wi = randint<int>(0, 10, {8, 8}).to_gpu();
        auto wi2 = randint<int>(0, 10, {8, 8}).to_gpu();
        bench_op_ms<int>(wi, wi2, &Tensor<int>::operator+);
    }
    {
        auto wa = rand<float>({4, 4}), wb = rand<float>({4, 4});
        ref.add(wa, wb);
        auto wi = randint<int>(0, 10, {4, 4});
        auto wi2 = randint<int>(0, 10, {4, 4});
        ref.add(wi, wi2);
    }
    std::cout << "  warm-up complete.\n" << std::endl;

    // ── float add 1024x1024 ──
    {
        std::cout << "  float add 1024x1024 ..." << std::endl;
        auto a = rand<float>({1024, 1024});
        auto b = rand<float>({1024, 1024});
        auto ag = a.to_gpu(), bg = b.to_gpu();

        // raw kernel: pre-allocate output
        auto araw = ag.get_raw(), braw = bg.get_raw();
        auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
            DType::Float32, {1024, 1024});
        // warm-up raw kernel
        bench_raw_add_ms(araw, braw, out_raw);
        cudaDeviceSynchronize();

        float op_ms  = bench_op_ms<float>(ag, bg, &Tensor<float>::operator+);
        float raw_ms = bench_raw_add_ms(araw, braw, out_raw);
        float tms    = bench_torch_ms<float>(a, b, REPEAT);
        results.push_back({"float32", "1024x1024", op_ms, raw_ms, tms});
        cudaoplib_kernel::free_gpu_tensor(out_raw);
    }

    // ── float add 4096x4096 ──
    {
        std::cout << "  float add 4096x4096 ..." << std::endl;
        auto a = rand<float>({4096, 4096});
        auto b = rand<float>({4096, 4096});
        auto ag = a.to_gpu(), bg = b.to_gpu();

        auto araw = ag.get_raw(), braw = bg.get_raw();
        auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
            DType::Float32, {4096, 4096});
        bench_raw_add_ms(araw, braw, out_raw);

        float op_ms  = bench_op_ms<float>(ag, bg, &Tensor<float>::operator+);
        float raw_ms = bench_raw_add_ms(araw, braw, out_raw);
        float tms    = bench_torch_ms<float>(a, b, REPEAT);
        results.push_back({"float32", "4096x4096", op_ms, raw_ms, tms});
        cudaoplib_kernel::free_gpu_tensor(out_raw);
    }

    // ── float mul 1024x1024 ──
    {
        std::cout << "  float mul 1024x1024 ..." << std::endl;
        auto a = rand<float>({1024, 1024});
        auto b = rand<float>({1024, 1024});
        auto ag = a.to_gpu(), bg = b.to_gpu();

        auto araw = ag.get_raw(), braw = bg.get_raw();
        auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
            DType::Float32, {1024, 1024});
        bench_raw_mul_ms(araw, braw, out_raw);
        cudaDeviceSynchronize();

        float op_ms  = bench_op_ms<float>(ag, bg, &Tensor<float>::operator*);
        float raw_ms = bench_raw_mul_ms(araw, braw, out_raw);
        float tms    = bench_torch_ms<float>(a, b, REPEAT_L);
        results.push_back({"float32", "1024x1024", op_ms, raw_ms, tms});
        cudaoplib_kernel::free_gpu_tensor(out_raw);
    }

    // ── int add 1024x1024 ──
    {
        std::cout << "  int add 1024x1024 ..." << std::endl;
        auto a = randint<int>(0, 100, {1024, 1024});
        auto b = randint<int>(0, 100, {1024, 1024});
        auto ag = a.to_gpu(), bg = b.to_gpu();

        auto araw = ag.get_raw(), braw = bg.get_raw();
        auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
            DType::Int32, {1024, 1024});
        bench_raw_add_ms(araw, braw, out_raw);

        float op_ms  = bench_op_ms<int>(ag, bg, &Tensor<int>::operator+);
        float raw_ms = bench_raw_add_ms(araw, braw, out_raw);
        float tms    = bench_torch_ms<int>(a, b, REPEAT);
        results.push_back({"int32", "1024x1024", op_ms, raw_ms, tms});
        cudaoplib_kernel::free_gpu_tensor(out_raw);
    }

    // ── float add broadcast ──
    {
        std::cout << "  float add broadcast 1024x1 + 1x4096 ..." << std::endl;
        auto a = rand<float>({1024, 1});
        auto b = rand<float>({1, 4096});
        auto ag = a.to_gpu(), bg = b.to_gpu();

        auto araw = ag.get_raw(), braw = bg.get_raw();
        auto out_raw = cudaoplib_kernel::create_empty_gpu_tensor(
            DType::Float32, {1024, 4096});
        bench_raw_add_ms(araw, braw, out_raw);

        float op_ms  = bench_op_ms<float>(ag, bg, &Tensor<float>::operator+);
        float raw_ms = bench_raw_add_ms(araw, braw, out_raw);
        float tms    = bench_torch_ms<float>(a, b, REPEAT_L);
        results.push_back({"float32", "1024x1+1x4096", op_ms, raw_ms, tms});
        cudaoplib_kernel::free_gpu_tensor(out_raw);
    }

    // ── print table ──
    printf("\n╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Perf Comparison: cudaoplib vs PyTorch (kernel only, 10000x avg)       ║\n");
    printf("╠══════════╤═══════════════╤══════════╤══════════╤══════════╤════════════╣\n");
    printf("║ dtype    │ shape         │ op+(ms)  │ raw(ms)  │ torch    │ raw/torch  ║\n");
    printf("║          │               │          │          │ (ms)     │            ║\n");
    printf("╠══════════╪═══════════════╪══════════╪══════════╪══════════╪════════════╣\n");
    for (auto& r : results)
    {
        float ratio = (r.tms > 0) ? r.raw_ms / r.tms : 0;
        printf("║ %-8s │ %-13s │ %8.4f │ %8.4f │ %8.4f │ %10.2f ║\n",
               r.dtype.c_str(), r.shape.c_str(),
               r.op_ms, r.raw_ms, r.tms, ratio);
    }
    printf("╚══════════╧═══════════════╧══════════╧══════════╧══════════╧════════════╝\n");

    printf("\n  op+(ms)  = operator+ with cudaMalloc  (CUDA events)\n");
    printf("  raw(ms)  = raw kernel, pre-allocated   (CUDA events)\n");
    printf("  torch    = torch via TorchRef          (CPU timer, sync'd)\n\n");

    ref.set_repeat(1);
}
