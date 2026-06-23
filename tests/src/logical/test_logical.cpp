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

static void expect_bool_match(const Tensor<bool>& a, const Tensor<bool>& b)
{
    ASSERT_EQ(a.numel(), b.numel());
    ASSERT_EQ(a.shape(), b.shape());
    for (size_t i = 0; i < a.numel(); i++)
        EXPECT_EQ(a.data()[i], b.data()[i]) << "Mismatch at index " << i;
}

// ── type-parameterized logical tests ──────────────────────────

template <typename T>
class LogicalTest : public ::testing::Test {
protected:
    void SetUp() override { REQUIRE_CUDA(); }
};
using LogicalTypes = ::testing::Types<int, int8_t, float, bool>;
TYPED_TEST_SUITE(LogicalTest, LogicalTypes);

TYPED_TEST(LogicalTest, And)
{
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = ones<TypeParam>({16, 32});
    auto b = zeros<TypeParam>({16, 32});

    auto r = (a.to_gpu() && b.to_gpu()).to_cpu();
    auto expected = ref.logical_and(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(LogicalTest, Or)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({8, 16});
    auto b = zeros<TypeParam>({8, 16});

    auto r = (a.to_gpu() || b.to_gpu()).to_cpu();
    auto expected = ref.logical_or(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(LogicalTest, AndBroadcast)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({3, 1, 4});
    auto b = ones<TypeParam>({1, 5, 4});

    auto r = (a.to_gpu() && b.to_gpu()).to_cpu();
    auto expected = ref.logical_and(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(LogicalTest, Truthiness)
{
    auto& ref = TorchRef::instance();
    // Use all zeros vs all ones to test truthiness behavior
    auto a = zeros<TypeParam>({8});
    auto b = ones<TypeParam>({8});

    {
        auto r = (a.to_gpu() && b.to_gpu()).to_cpu();
        auto expected = ref.logical_and(a, b);
        expect_bool_match(r, expected);
        // zeros && ones → all false
        for (size_t i = 0; i < 8; i++) EXPECT_FALSE(r.data()[i]);
    }
    {
        auto r = (a.to_gpu() || b.to_gpu()).to_cpu();
        auto expected = ref.logical_or(a, b);
        expect_bool_match(r, expected);
        // zeros || ones → all true
        for (size_t i = 0; i < 8; i++) EXPECT_TRUE(r.data()[i]);
    }
}
