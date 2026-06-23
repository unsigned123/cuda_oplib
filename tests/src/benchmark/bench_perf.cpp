#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"
#include "bench_utils.hpp"

using namespace cudaoplib;
using namespace cudaoplib_bench;

#define REQUIRE_CUDA() \
    if (!cudaoplib_bench::cuda_available()) { GTEST_SKIP() << "No CUDA device"; }

// ── warm-up all kernel + torch ────────────────────────────────

static void warmup()
{
    auto& ref = cudaoplib_test::TorchRef::instance();

    // raw kernel warm-up (JIT)
    {
        auto out = cudaoplib_kernel::create_empty_gpu_tensor(DType::Float32, {8, 8});
        auto ag = ones<float>({8, 8}).to_gpu(), bg = ones<float>({8, 8}).to_gpu();
        auto ar = ag.get_raw(), br = bg.get_raw();

        cudaoplib_kernel::add(ar, br, out);
        cudaoplib_kernel::mul(ar, br, out);
        cudaDeviceSynchronize();

        auto out_i32 = cudaoplib_kernel::create_empty_gpu_tensor(DType::Int32, {8, 8});
        auto ig = ones<int>({8, 8}).to_gpu();
        auto ir = ig.get_raw();
        cudaoplib_kernel::add(ir, ir, out_i32);
        cudaDeviceSynchronize();

        cudaoplib_kernel::free_gpu_tensor(out);
        cudaoplib_kernel::free_gpu_tensor(out_i32);
    }
    // torch worker warm-up
    {
        auto a = ones<float>({4, 4}), b = ones<float>({4, 4});
        ref.add(a, b);
        auto ia = ones<int>({4, 4}), ib = ones<int>({4, 4});
        ref.add(ia, ib);
    }
}

// ── benchmarks ────────────────────────────────────────────────

TEST(Benchmark, ElementWiseFloat)
{
    REQUIRE_CUDA();
    warmup();

    const int ITERS = 10000;
    std::vector<BenchResult> results;

    results.push_back(bench_add<float>({1024, 1024},         1, ITERS));
    results.push_back(bench_add<float>({4096, 4096},         1, ITERS));
    results.push_back(bench_mul<float>({1024, 1024},         1, ITERS));
    results.push_back(bench_add_broadcast<float>({1024, 1}, {1, 4096}, 1, ITERS));

    print_bench_table(results);

    auto& ref = cudaoplib_test::TorchRef::instance();
    ref.set_repeat(1);
}

TEST(Benchmark, ElementWiseInt)
{
    REQUIRE_CUDA();
    warmup();

    const int ITERS = 10000;
    std::vector<BenchResult> results;

    results.push_back(bench_add<int>({1024, 1024}, 1, ITERS));

    print_bench_table(results);
}
