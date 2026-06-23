#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace cudaoplib;
using namespace cudaoplib_test;

// ── helpers ───────────────────────────────────────────────────

static bool cuda_available()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

#define REQUIRE_CUDA() \
    if (!cuda_available()) { GTEST_SKIP() << "No CUDA device"; }

template <SupportedDType T>
static void expect_allclose(const Tensor<T>& a, const Tensor<T>& b,
                            double rtol = 1e-4, double atol = 1e-6)
{
    ASSERT_EQ(a.numel(), b.numel());
    ASSERT_EQ(a.shape(), b.shape());
    const T* ad = a.data();
    const T* bd = b.data();
    for (size_t i = 0; i < a.numel(); i++)
    {
        if constexpr (std::is_same_v<T, float>)
            EXPECT_NEAR(ad[i], bd[i], atol + rtol * std::fabs(bd[i]))
                << "Mismatch at index " << i;
        else if constexpr (std::is_same_v<T, __half>)
            EXPECT_NEAR(__half2float(ad[i]), __half2float(bd[i]),
                        atol + rtol * std::fabs(__half2float(bd[i])))
                << "Mismatch at index " << i;
        else
            EXPECT_EQ(ad[i], bd[i]) << "Mismatch at index " << i;
    }
}

// ── test fixture ──────────────────────────────────────────────

template <typename T>
class ArithmeticTest : public ::testing::Test {
protected:
    void SetUp() override { REQUIRE_CUDA(); }
};
using ArithmeticTypes = ::testing::Types<int, int8_t, float>;
TYPED_TEST_SUITE(ArithmeticTest, ArithmeticTypes);

// ── add ───────────────────────────────────────────────────────

TYPED_TEST(ArithmeticTest, Add)
{
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = ones<TypeParam>({64, 128});
    auto b = ones<TypeParam>({64, 128});

    auto a_gpu = a.to_gpu();
    auto b_gpu = b.to_gpu();
    auto r_gpu = a_gpu + b_gpu;
    auto r = r_gpu.to_cpu();

    auto expected = ref.add(a, b);
    expect_allclose(r, expected);
}

TYPED_TEST(ArithmeticTest, AddBroadcast)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({2, 3});
    auto b = ones<TypeParam>({1, 3});

    auto r = (a.to_gpu() + b.to_gpu()).to_cpu();
    auto expected = ref.add(a, b);
    expect_allclose(r, expected);
}

TYPED_TEST(ArithmeticTest, AddScalar)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({5});
    auto b = zeros<TypeParam>({1});

    auto r = (a.to_gpu() + b.to_gpu()).to_cpu();
    auto expected = ref.add(a, b);
    expect_allclose(r, expected);
}

// ── sub ───────────────────────────────────────────────────────

TYPED_TEST(ArithmeticTest, Sub)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({64, 128});
    auto b = ones<TypeParam>({64, 128});

    auto r = (a.to_gpu() - b.to_gpu()).to_cpu();
    auto expected = ref.sub(a, b);
    expect_allclose(r, expected);
}

TYPED_TEST(ArithmeticTest, SubBroadcast)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({4, 8, 16});
    auto b = ones<TypeParam>({1, 8, 1});

    auto r = (a.to_gpu() - b.to_gpu()).to_cpu();
    auto expected = ref.sub(a, b);
    expect_allclose(r, expected);
}

// ── mul ───────────────────────────────────────────────────────

TYPED_TEST(ArithmeticTest, Mul)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({32, 64});
    auto b = ones<TypeParam>({32, 64});

    auto r = (a.to_gpu() * b.to_gpu()).to_cpu();
    auto expected = ref.mul(a, b);
    expect_allclose(r, expected);
}

TYPED_TEST(ArithmeticTest, MulBroadcast)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({3, 1, 5});
    auto b = ones<TypeParam>({1, 4, 1});

    auto r = (a.to_gpu() * b.to_gpu()).to_cpu();
    auto expected = ref.mul(a, b);
    expect_allclose(r, expected);
}

// ── div ───────────────────────────────────────────────────────

TYPED_TEST(ArithmeticTest, Div)
{
    auto& ref = TorchRef::instance();
    // Use randn to avoid zeros in denominator
    auto a = ones<TypeParam>({16, 32});
    auto b = ones<TypeParam>({16, 32});

    auto r = (a.to_gpu() / b.to_gpu()).to_cpu();
    auto expected = ref.div(a, b);
    expect_allclose(r, expected);
}

// ── mod ──────────────────────────────────────────────────────

TYPED_TEST(ArithmeticTest, Mod)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({4, 8});
    auto b = ones<TypeParam>({4, 8});

    auto r = (a.to_gpu() % b.to_gpu()).to_cpu();
    auto expected = ref.mod(a, b);
    expect_allclose(r, expected);
}

// ── compound assignment ───────────────────────────────────────

TYPED_TEST(ArithmeticTest, CompoundAdd)
{
    auto& ref = TorchRef::instance();
    auto a = ones<TypeParam>({8, 8});
    auto b = zeros<TypeParam>({8, 8});
    auto a_copy = a;  // deep copy for reference

    auto a_gpu = a.to_gpu();
    a_gpu += b.to_gpu();
    auto r = a_gpu.to_cpu();

    auto expected = ref.add(a_copy, b);
    expect_allclose(r, expected);
}

// ── __half special case ───────────────────────────────────────

TEST(ArithmeticTestHalf, Add)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    ASSERT_TRUE(ref.is_alive());

    auto a = rand<__half>({8, 16});
    auto b = rand<__half>({8, 16});

    auto r = (a.to_gpu() + b.to_gpu()).to_cpu();
    auto expected = ref.add(a, b);
    expect_allclose(r, expected, 1e-2, 1e-3);  // half precision
}

TEST(ArithmeticTestHalf, Mul)
{
    REQUIRE_CUDA();
    auto& ref = TorchRef::instance();
    auto a = rand<__half>({16, 8});
    auto b = rand<__half>({16, 8});

    auto r = (a.to_gpu() * b.to_gpu()).to_cpu();
    auto expected = ref.mul(a, b);
    expect_allclose(r, expected, 1e-2, 1e-3);
}
