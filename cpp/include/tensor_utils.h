#pragma once

#include "kernel_tensor.h"
#include "elementwise.h"

#include "tensor_defines.h"

#include <type_traits>
#include <vector>
#include <initializer_list>
#include <concepts>
#include <ranges>
#include <stdexcept>

namespace cudaoplib
{
// Utilities

template <SupportedDType T>
constexpr DType dtype_to_enum([[maybe_unused]] T)
{
    if constexpr (std::is_same_v<T, float>)
        return DType::Float32;
    else if constexpr (std::is_same_v<T, __half>)
        return DType::Float16;
    else if constexpr (std::is_same_v<T, int>)
        return DType::Int32;
    else if constexpr (std::is_same_v<T, int8_t>)
        return DType::Int8;
    else if constexpr (std::is_same_v<T, bool>)
        return DType::Bool;
    else
        static_assert(false, "FATAL: cudaoplib::dtype_to_enum failed: Unsupported DType");
}

inline size_t numel_from_shape(const TensorShape& shape)
{
    size_t numel = 1;
    for (const auto& dim : shape)
        numel *= dim;
    return numel;
}

template<std::ranges::range Range>
constexpr std::vector<size_t> deduce_shape(const Range& range, bool is_recursive_call=false)
{
    using Inner = std::ranges::range_value_t<Range>;

    std::vector<size_t> shape;
    size_t count = 0, inner_uniform_size = 0;

    for (const auto& sub : range)
    {
        if constexpr (std::ranges::range<Inner>)
        {
            auto sub_shape = deduce_shape<Inner>(sub, true);
            if (count == 0)
            {
                inner_uniform_size = sub_shape.empty() ? 0 : sub_shape.back();
                
                if (inner_uniform_size == 0)
                {
                    if consteval {
                        throw "FATAL: Empty ranges to initialize a tensor are prohibited.";
                    } else {
                        throw std::runtime_error("FATAL: Empty ranges to initialize a tensor are prohibited.");
                    }
                }
                shape = std::move(sub_shape);
            }
            else
            {
                if (sub_shape.empty())
                {
                    if consteval {
                        throw "FATAL: Empty ranges to initialize a tensor are prohibited.";
                    } else {
                        throw std::runtime_error("FATAL: Empty ranges to initialize a tensor are prohibited.");
                    }
                }
                
                if (sub_shape.back() != inner_uniform_size)
                {
                    if consteval {
                        throw "FATAL: Inconsistent dimension sizes to initialize a tensor are prohibited.";
                    } else {
                        throw std::runtime_error("FATAL: Inconsistent dimension sizes to initialize a tensor are prohibited.");
                    }
                }
            }
        }
        count++;
    }
    shape.insert(shape.begin(), count);

    if (!is_recursive_call)
    {
        if (shape.size() > MAX_TENSOR_DIM)
        {
            if consteval {
                throw "FATAL: The number of tensor dims greater than MAX_TENSOR_DIM is unsupported.";
            } else {
                throw std::runtime_error("FATAL: The number of tensor dims greater than MAX_TENSOR_DIM is unsupported.");
            }
        }
    }

    return shape;
}

template<std::ranges::range Range, SupportedDType T>
void _fill_data_recursive(const Range& range, T*& out)
{
    using Inner = std::ranges::range_value_t<Range>;
    
    for (const auto& sub : range)
    {
        if constexpr (std::ranges::range<Inner>)
            _fill_data_recursive(sub, out);
        else
            *out++ = static_cast<T>(sub);
    }
}

template<std::ranges::range Range, SupportedDType T>
void fill_data(const Range& range, T* buffer)
{
    T* out = buffer;
    _fill_data_recursive<Range, T>(range, out);
}



} // namespace cudaoplib