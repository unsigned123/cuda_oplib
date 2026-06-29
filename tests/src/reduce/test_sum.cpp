#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "bench_utils.hpp"

#include <vector>
#include <cuda_runtime.h>

using namespace cudaoplib;
using namespace cudaoplib_bench;

#define REQUIRE_CUDA() \
    if (!cudaoplib_bench::cuda_available()) { GTEST_SKIP() << "No CUDA device"; }

// ── correctness: dense (reduce last dim) ──────────────────────

TEST(SumCorrectness, DenseLastDim)
{
    REQUIRE_CUDA();

    auto t = ones<float>({3, 4}).to_gpu();
    auto r = t.sum(1, true).to_cpu();  // sum along last dim, keepdim

    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 1}));
    for (size_t i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(r.data()[i], 4.0f) << "at row " << i;
}

// ── correctness: striding (reduce middle dim) ─────────────────

TEST(SumCorrectness, StridingMiddleDim)
{
    REQUIRE_CUDA();

    auto t = ones<float>({2, 3, 4}).to_gpu();
    auto r = t.sum(1, true).to_cpu();

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 1, 4}));
    for (size_t i = 0; i < 8; i++)
        EXPECT_FLOAT_EQ(r.data()[i], 3.0f) << "at index " << i;
}

// ── correctness: no keepdim ──────────────────────────────────

TEST(SumCorrectness, NoKeepdim)
{
    REQUIRE_CUDA();

    auto t = ones<float>({2, 3, 4}).to_gpu();
    auto r = t.sum(1, false).to_cpu();

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 4}));
    for (size_t i = 0; i < 8; i++)
        EXPECT_FLOAT_EQ(r.data()[i], 3.0f);
}

// ── correctness: int8_t ──────────────────────────────────────

TEST(SumCorrectness, Int8)
{
    REQUIRE_CUDA();

    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<int8_t> t(data, {2, 4}, Device::CPU, true);
    auto r = t.to_gpu().sum(1, true).to_cpu();

    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(r.data()[0], int8_t(10));  // 1+2+3+4
    EXPECT_EQ(r.data()[1], int8_t(26));  // 5+6+7+8
}

// ── correctness: large dense ─────────────────────────────────

TEST(SumCorrectness, LargeDense)
{
    REQUIRE_CUDA();

    auto t = ones<float>({128, 1024}).to_gpu();
    auto r = t.sum(1, true).to_cpu();

    EXPECT_EQ(r.shape(), (std::vector<size_t>{128, 1}));
    for (size_t i = 0; i < 128; i++)
        EXPECT_NEAR(r.data()[i], 1024.0f, 1.0f) << "at row " << i;
}

// ── correctness vs pytorch (random data) ─────────────────────

TEST(SumCorrectness, AgainstTorch)
{
    REQUIRE_CUDA();
    auto& ref = cudaoplib_test::TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    struct Case {
        std::string desc;
        TensorShape shape;
        int dim;
        bool keepdim;
    };
    std::vector<Case> cases = {
        {"dense 1024",        {1024, 1024},        1, true},
        {"dense 4096",        {4096, 4096},        1, true},
        {"dense non-pow2",    {1000, 1000},        1, true},
        {"striding dim=0",    {1024, 1024},        0, true},
        {"striding dim=0 L",  {4096, 4096},        0, true},
        {"striding dim=1",    {2, 1024, 1024},     1, true},
        {"striding 3D",       {100, 300, 500},     1, false},
        {"high-dim 4d",       {16, 16, 16, 16},    1, true},
        {"high-dim 6d",       {4, 4, 4, 4, 4, 4}, 5, true},
        {"small",             {4, 8},              1, true},
        {"medium",            {128, 256},          1, true},
        {"extreme dense",     {64, 65536},         1, true},
        // narrow correctness checks
        {"1048576x2 dim1",    {1048576, 2},        1, true},
        {"1048576x2 dim0",    {1048576, 2},        0, true},
        {"2x1048576 dim0",    {2, 1048576},        0, true},
    };

    for (auto& c : cases)
    {
        auto t = rand<float>(c.shape);
        auto our_r = t.to_gpu().sum(c.dim, c.keepdim).to_cpu();
        auto torch_r = ref.sum(t, c.dim);

        ASSERT_EQ(our_r.numel(), torch_r.numel()) << c.desc;
        // higher tolerance for large reductions (float accumulation error)
        size_t red_len = c.shape[c.dim];
        float tol = 0.01f;
        if (red_len > 1000000) tol = 1.0f;
        else if (red_len > 10000) tol = 0.1f;
        for (size_t i = 0; i < our_r.numel(); i++)
            EXPECT_NEAR(our_r.data()[i], torch_r.data()[i], tol)
                << c.desc << " at index " << i;
    }
}

// ── ncu profiling: ultra-fast cases (< 0.01x torch) ──────────

TEST(SumProfile, UltraFastCases)
{
    REQUIRE_CUDA();
    auto& ref = cudaoplib_test::TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    struct Case { std::string desc; TensorShape shape; int dim; };
    std::vector<Case> cases = {
        {"8d",                     {8,8,8,8,8,8,8,8}, 7},
        {"2x1048576x2 dim1",       {2,1048576,2},      1},
        {"1048576x2 dim1",         {1048576,2},        1},
        {"2x1048576 dim0",         {2,1048576},        0},
        {"1048576x2 dim0",         {1048576,2},        0},
        {"33554432x2 dim1",        {33554432,2},       1},
        {"2x33554432 dim0",        {2,33554432},       0},
        {"33554432x2 dim0",        {33554432,2},       0},
    };
    for (auto& c : cases) {
        auto t = rand<float>(c.shape);
        auto our = t.to_gpu().sum(c.dim, true).to_cpu();
        auto torch_r = ref.sum(t, c.dim);

        double our_sum = 0, torch_sum = 0;
        for (size_t i = 0; i < our.numel(); i++) {
            our_sum += our.data()[i];
            torch_sum += torch_r.data()[i];
        }
        printf("[%s] shape=(", c.desc.c_str());
        for (size_t i = 0; i < c.shape.size(); i++)
            printf("%s%zu", i?",":"", c.shape[i]);
        printf(") dim=%d our_sum=%.6f torch_sum=%.6f diff=%.6f\n",
               c.dim, our_sum, torch_sum, fabs(our_sum - torch_sum));
        EXPECT_NEAR(our_sum, torch_sum, our_sum * 0.01 + 0.1)
            << c.desc << " total sum mismatch";
    }
}

// ── nsys profiling (no torch, pure kernel calls) ─────────────

TEST(SumProfile, VariousShapes)
{
    REQUIRE_CUDA();

    struct Case { std::string desc; TensorShape shape; int dim; };
    std::vector<Case> cases = {
        // dense path (dim == last)
        {"dense 4x4",               {4, 4},             1},
        {"dense 32x32",             {32, 32},           1},
        {"dense 128x128",           {128, 128},         1},
        {"dense 1024x1024",         {1024, 1024},       1},
        {"dense 4096x4096",         {4096, 4096},       1},
        {"dense 64x65536",          {64, 65536},        1},
        {"dense 1000x1000",         {1000, 1000},       1},
        // striding path (dim < last)
        {"strid dim0 1024x1024",    {1024, 1024},       0},
        {"strid dim0 4096x4096",    {4096, 4096},       0},
        {"strid 2x1024x1024",       {2, 1024, 1024},    1},
        {"strid 100x300x500",       {100, 300, 500},    1},
        {"strid 4096x4096x64",      {4096, 4096, 64},   1},
        // high-dim
        {"4d 16x16x16x16",          {16, 16, 16, 16},  1},
        {"6d 4x4x4x4x4x4",          {4, 4, 4, 4, 4, 4},5},
        // ultra-long reduction dim
        {"dense 1x1048576",          {1, 1048576},       1},
        {"dense 16x1048576",         {16, 1048576},      1},
        {"strid 1048576x16 dim0",    {1048576, 16},      0},
        {"strid 2x1048576x2 dim1",   {2, 1048576, 2},    1},
        // extreme narrow: n_col ≤ TILE_SIZE (32)
        {"dense 2x1048576 dim=1",    {2, 1048576},       1},
        {"dense 1048576x2 dim=1",    {1048576, 2},       1},
        {"strid 2x1048576 dim=0",    {2, 1048576},       0},
        {"strid 1048576x2 dim=0",    {1048576, 2},       0},
        // gigantor
        {"dense 2x33554432 dim=1",   {2, 33554432},      1},
        {"dense 33554432x2 dim=1",   {33554432, 2},      1},
        {"strid 2x33554432 dim=0",   {2, 33554432},      0},
        {"strid 33554432x2 dim=0",   {33554432, 2},      0},
        // all benchmark shapes
        {"dense 1x1048576",          {1, 1048576},       1},
        {"dense 16x1048576",         {16, 1048576},      1},
        {"dense 1024x1024",          {1024, 1024},       1},
        {"strid 1048576x16 dim0",    {1048576, 16},      0},
        {"strid 2x1048576x2 dim1",   {2, 1048576, 2},    1},
        {"strid 2x1048576 dim1",     {2, 1048576},       1},
        {"strid 1048576x2 dim1",     {1048576, 2},       1},
        {"strid 2x1048576 dim0",     {2, 1048576},       0},
        {"strid 1048576x2 dim0",     {1048576, 2},       0},
    };

    for (auto& c : cases)
    {
        auto t = ones<float>(c.shape).to_gpu();
        auto r = t.sum(c.dim, true).to_cpu();
        EXPECT_GT(r.numel(), 0u) << c.desc;
    }
}

// ── benchmark ─────────────────────────────────────────────────

TEST(SumBenchmark, VariousSizes)
{
    REQUIRE_CUDA();

    const int ITERS = 100;
    std::vector<BenchResult> results;

    // warm-up
    bench_sum<float>({64, 64}, 1, 3, 100);
    bench_sum<float>({64, 64}, 0, 3, 100);
    cudaDeviceSynchronize();

    // ── dense path (reduce last dim) ──
    results.push_back(bench_sum<float>({1024, 1024},   1, 1, ITERS));
    results.push_back(bench_sum<float>({4096, 4096},   1, 1, ITERS));
    results.push_back(bench_sum<float>({64, 65536},    1, 1, ITERS));
    results.push_back(bench_sum<float>({1000, 1000},   1, 1, ITERS));

    // ── striding path ──
    results.push_back(bench_sum<float>({1024, 1024},   0, 1, ITERS));
    results.push_back(bench_sum<float>({4096, 4096},   0, 1, ITERS));
    results.push_back(bench_sum<float>({2, 1024, 1024}, 1, 1, ITERS, "2x1024x1024 dim1"));
    results.push_back(bench_sum<float>({100, 300, 500}, 1, 1, ITERS, "100x300x500 dim1"));

    // ── high-dim ──
    results.push_back(bench_sum<float>({8, 8, 8, 8, 8, 8, 8, 8}, 7, 1, ITERS, "8d"));
    results.push_back(bench_sum<float>({4, 4, 4, 4, 4, 4},       5, 1, ITERS, "6d"));
    results.push_back(bench_sum<float>({16, 16, 16, 16},         1, 1, ITERS, "4d"));

    // ── narrow / ultra-long (commented out for now) ──
    results.push_back(bench_sum<float>({1, 1048576},     1, 1, ITERS, "1x1M dim1"));
    results.push_back(bench_sum<float>({1048576, 16},    0, 1, ITERS, "1Mx16 dim0"));
    results.push_back(bench_sum<float>({2, 1048576},     1, 1, ITERS, "2x1M dim1"));
    results.push_back(bench_sum<float>({1048576, 2},     1, 1, ITERS, "1Mx2 dim1"));
    results.push_back(bench_sum<float>({2, 33554432},    0, 1, ITERS, "2x33M dim0"));
    results.push_back(bench_sum<float>({33554432, 2},    0, 1, ITERS, "33Mx2 dim0"));

    // ── small ──
    results.push_back(bench_sum<float>({4, 4},       1, 1, ITERS));
    results.push_back(bench_sum<float>({32, 32},     1, 1, ITERS));
    results.push_back(bench_sum<float>({128, 128},   1, 1, ITERS));
    results.push_back(bench_sum<float>({16, 48},     1, 1, ITERS));  // non-square small

    print_bench_table(results);
}
