#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"

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

// ── Bool tensor ───────────────────────────────────────────────

TEST(BoolEdge, ConstructAndPrint)
{
    bool data[] = {true, false, true, false, true};
    Tensor<bool> t(data, {5}, Device::CPU, true);
    EXPECT_EQ(t.numel(), 5u);
    EXPECT_EQ(t.data()[0], true);
    EXPECT_EQ(t.data()[1], false);
    EXPECT_EQ(t.data()[4], true);
}

TEST(BoolEdge, CompareWithTorch)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    bool ad[] = {true, false, true, true};
    bool bd[] = {false, false, true, false};
    Tensor<bool> a(ad, {4}, Device::CPU, true);
    Tensor<bool> b(bd, {4}, Device::CPU, true);

    auto r_eq  = (a.to_gpu() == b.to_gpu()).to_cpu();
    auto ex_eq = ref.eq(a, b);
    EXPECT_EQ(r_eq.data()[0], false);  // true == false
    EXPECT_EQ(r_eq.data()[1], true);   // false == false
    EXPECT_EQ(r_eq.data()[2], true);   // true == true
    EXPECT_EQ(r_eq.data()[3], false);  // true == false
}

TEST(BoolEdge, LogicalWithTorch)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    bool ad[] = {true, false, true, false};
    bool bd[] = {true, true, false, false};
    Tensor<bool> a(ad, {4}, Device::CPU, true);
    Tensor<bool> b(bd, {4}, Device::CPU, true);

    auto r_and = (a.to_gpu() && b.to_gpu()).to_cpu();
    auto ex_and = ref.logical_and(a, b);
    EXPECT_EQ(r_and.data()[0], true);   // T && T
    EXPECT_EQ(r_and.data()[1], false);  // F && T
    EXPECT_EQ(r_and.data()[2], false);  // T && F
    EXPECT_EQ(r_and.data()[3], false);  // F && F

    auto r_or = (a.to_gpu() || b.to_gpu()).to_cpu();
    auto ex_or = ref.logical_or(a, b);
    EXPECT_EQ(r_or.data()[0], true);   // T || T
    EXPECT_EQ(r_or.data()[1], true);   // F || T
    EXPECT_EQ(r_or.data()[2], true);   // T || F
    EXPECT_EQ(r_or.data()[3], false);  // F || F
}

// ── scalar / empty ────────────────────────────────────────────

TEST(EdgeCase, ScalarTensor)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = randn<float>({1});
    auto b = randn<float>({1});

    auto r = (a.to_gpu() + b.to_gpu()).to_cpu();
    auto expected = ref.add(a, b);
    EXPECT_NEAR(r.data()[0], expected.data()[0], 1e-5);
}

TEST(EdgeCase, LargeNDim)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = randn<float>({2, 2, 2, 2, 2, 2, 2, 2});  // 8-d, 256 elems
    auto b = randn<float>({2, 2, 2, 2, 2, 2, 2, 2});

    auto r = (a.to_gpu() * b.to_gpu()).to_cpu();
    auto expected = ref.mul(a, b);
    ASSERT_EQ(r.numel(), 256u);
    for (size_t i = 0; i < 256; i++)
        EXPECT_NEAR(r.data()[i], expected.data()[i], 1e-5) << "at " << i;
}

TEST(EdgeCase, NegativeValues)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    int ad[] = {-5, -3, 0, 2, 7};
    int bd[] = {3, 3, 3, 3, 3};
    Tensor<int> a(ad, {5}, Device::CPU, true);
    Tensor<int> b(bd, {5}, Device::CPU, true);

    auto r = (a.to_gpu() % b.to_gpu()).to_cpu();
    auto expected = ref.mod(a, b);
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(r.data()[i], expected.data()[i]) << "at index " << i;
}

// ── Move semantics ────────────────────────────────────────────

TEST(Memory, MoveTransfersOwnership)
{
    Tensor<int> src = {{1, 2, 3}};
    void* src_ptr = const_cast<int*>(src.data());
    ASSERT_TRUE(src.owns_data());

    Tensor<int> dst = std::move(src);
    EXPECT_FALSE(src.owns_data());
    EXPECT_TRUE(dst.owns_data());
    EXPECT_EQ(dst.data(), static_cast<const int*>(src_ptr));
    EXPECT_EQ(dst.numel(), 3u);
}

TEST(Memory, CopyIsDeep)
{
    Tensor<int> src = {{1, 2, 3}};
    Tensor<int> dst = src;
    EXPECT_NE(dst.data(), src.data());
    EXPECT_TRUE(dst.owns_data());
    EXPECT_EQ(dst.numel(), src.numel());
    for (size_t i = 0; i < 3; i++)
        EXPECT_EQ(dst.data()[i], src.data()[i]);
}

// ── Factory basics ────────────────────────────────────────────

TEST(Factory, ZerosAllZero)
{
    auto t = zeros<float>({4, 4});
    for (size_t i = 0; i < 16; i++)
        EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
}

TEST(Factory, OnesAllOne)
{
    auto t = ones<int>({3, 3});
    for (size_t i = 0; i < 9; i++)
        EXPECT_EQ(t.data()[i], 1);
}

TEST(Factory, RandInRange)
{
    auto t = rand<float>({1000});
    for (size_t i = 0; i < 1000; i++)
    {
        EXPECT_GE(t.data()[i], 0.0f);
        EXPECT_LT(t.data()[i], 1.0f);
    }
}

TEST(Factory, RandIntInRange)
{
    auto t = randint<int8_t>(0, 10, {100});
    for (size_t i = 0; i < 100; i++)
    {
        EXPECT_GE(t.data()[i], 0);
        EXPECT_LE(t.data()[i], 10);
    }
}
