#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"

#include <cuda_runtime.h>
#include <vector>

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

// ── type-parameterized comparison tests ───────────────────────

template <typename T>
class ComparisonTest : public ::testing::Test {
protected:
    void SetUp() override { REQUIRE_CUDA(); }
};
using ComparisonTypes = ::testing::Types<int, int8_t, float, bool>;
TYPED_TEST_SUITE(ComparisonTest, ComparisonTypes);

TYPED_TEST(ComparisonTest, Eq)
{
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = ones<TypeParam>({32, 64});
    auto b = zeros<TypeParam>({32, 64});

    auto r = (a.to_gpu() == b.to_gpu()).to_cpu();
    auto expected = ref.eq(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(ComparisonTest, Neq)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({16, 16});
    auto b = zeros<TypeParam>({16, 16});

    auto r = (a.to_gpu() != b.to_gpu()).to_cpu();
    auto expected = ref.neq(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(ComparisonTest, Lt)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({8, 16});
    auto b = zeros<TypeParam>({8, 16});

    auto r = (a.to_gpu() < b.to_gpu()).to_cpu();
    auto expected = ref.lt(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(ComparisonTest, Le)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({12, 8});
    auto b = zeros<TypeParam>({12, 8});

    auto r = (a.to_gpu() <= b.to_gpu()).to_cpu();
    auto expected = ref.le(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(ComparisonTest, Gt)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({4, 32});
    auto b = zeros<TypeParam>({4, 32});

    auto r = (a.to_gpu() > b.to_gpu()).to_cpu();
    auto expected = ref.gt(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(ComparisonTest, Ge)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({24, 4});
    auto b = zeros<TypeParam>({24, 4});

    auto r = (a.to_gpu() >= b.to_gpu()).to_cpu();
    auto expected = ref.ge(a, b);
    expect_bool_match(r, expected);
}

TYPED_TEST(ComparisonTest, EqBroadcast)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({4, 1, 8});
    auto b = ones<TypeParam>({1, 6, 8});

    auto r = (a.to_gpu() == b.to_gpu()).to_cpu();
    auto expected = ref.eq(a, b);
    expect_bool_match(r, expected);
}

// ── __half comparison ─────────────────────────────────────────

TEST(ComparisonHalfTest, EqLtGt)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = rand<__half>({8, 8});
    auto b = rand<__half>({8, 8});

    {
        auto r = (a.to_gpu() == b.to_gpu()).to_cpu();
        auto expected = ref.eq(a, b);
        expect_bool_match(r, expected);
    }
    {
        auto r = (a.to_gpu() < b.to_gpu()).to_cpu();
        auto expected = ref.lt(a, b);
        expect_bool_match(r, expected);
    }
    {
        auto r = (a.to_gpu() > b.to_gpu()).to_cpu();
        auto expected = ref.gt(a, b);
        expect_bool_match(r, expected);
    }
}
