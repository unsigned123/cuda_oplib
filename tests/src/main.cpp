#include <iostream>

#include <gtest/gtest.h>

#include "tensor.h"
#include "factory.h"
#include "tensor_ref.h"

// ── basic smoke tests ─────────────────────────────────────────

TEST(Construction, ScalarI32)
{
    cudaoplib::Tensor<int> t(42, cudaoplib_kernel::Device::CPU);
    EXPECT_EQ(t.numel(), 1u);
    EXPECT_EQ(t.shape(), std::vector<size_t>{1});
    EXPECT_EQ(t.get_device(), cudaoplib_kernel::Device::CPU);
    EXPECT_TRUE(t.owns_data());
    EXPECT_EQ(t.data()[0], 42);
}

TEST(Construction, InitializerList)
{
    cudaoplib::Tensor<float> t{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    EXPECT_EQ(t.numel(), 6u);
    EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 3}));
}

TEST(Construction, BoolTensor)
{
    bool data[] = {true, false, true};
    cudaoplib::Tensor<bool> t(data, {3}, cudaoplib_kernel::Device::CPU, true);
    EXPECT_EQ(t.data()[0], true);
    EXPECT_EQ(t.data()[1], false);
    EXPECT_EQ(t.data()[2], true);
}

TEST(Factory, ZerosAndOnes)
{
    auto z = cudaoplib::zeros<float>({2, 2});
    EXPECT_FLOAT_EQ(z.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(z.data()[3], 0.0f);

    auto o = cudaoplib::ones<int>({3});
    EXPECT_EQ(o.data()[0], 1);
    EXPECT_EQ(o.data()[2], 1);
}

// ── old scratch code (kept for reference) ─────────────────────

int original_main()
{
    using namespace cudaoplib;
    Tensor<int> t1{{1, 2, 3}, {1, 2, 3}};
    Tensor<int> t2 = 1;
    Tensor<int> t3{{1, 2, 3}, {1, 2, 5}};

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << (t1.to_gpu() == t3.to_gpu()) << std::endl;

    std::cout << "Hello world!" << std::endl;
    return 0;
}