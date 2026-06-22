#pragma once

#include "kernel_tensor.h"
#include "elementwise.h"

#include <type_traits>
#include <vector>
#include <initializer_list>
#include <concepts>
#include <ranges>
#include <stdexcept>

#include <iostream>


namespace cudaoplib
{

using DType = cudaoplib_kernel::DType;
using Device = cudaoplib_kernel::Device;

using TensorShape = std::vector<size_t>;

template <typename T>
concept SupportedDType = std::same_as<T, int> || std::same_as<T, int8_t> || std::same_as<T, float> || std::same_as<T, __half> || std::same_as<T, bool>;

template<typename T>
concept Scalar = !std::ranges::range<T>;

template<typename T>
concept CUDAFloatingPoint = std::is_floating_point_v<T> || std::is_same_v<T, __half>;

template <typename T>
concept ArithmeticType = !std::is_same_v<T, bool>;

template <typename T>
concept Interger = std::is_same_v<T, int> || std::is_same_v<T, int8_t>;


struct _ForceToUseRangeConstructor {};

namespace _nested_initializer_list 
{
    template<typename T, size_t N>
    struct nested_initializer_list 
    {
        using type = std::initializer_list<typename nested_initializer_list<T, N - 1>::type>;
    };
    template<typename T>
    struct nested_initializer_list<T, 1> 
    {
        using type = std::initializer_list<T>;
    };
} // namespace cudaoplib::detail

#define TENSOR_INIALIZER_LIST_CONSTRUCTOR(N) \
    Tensor(typename cudaoplib::_nested_initializer_list::nested_initializer_list<LogicalDType, N>::type list) : \
        Tensor(list, _ForceToUseRangeConstructor{}) \
    {}

template <SupportedDType LogicalDType> class Tensor;
template <SupportedDType LogicalDType>
std::ostream& operator<<(std::ostream& stream, const Tensor<LogicalDType>& input);

} // namespace cudaoplib