#pragma once

#include "tensor.h"
#include "random.h"

#include <random>

#include <type_traits>
#include <optional>
#include <memory.h>

namespace cudaoplib
{

template <SupportedDType LogicalDType = float>
Tensor<LogicalDType> zeros(const TensorShape& shape, Device device=Device::CPU)
{
    size_t numel = numel_from_shape(shape), size = sizeof(LogicalDType) * numel;
    LogicalDType* buffer = static_cast<LogicalDType*>(operator new(size));
    memset(buffer, 0, size);

    Tensor<LogicalDType> ret = Tensor<LogicalDType>(buffer, shape, device, false, true);
    if (device != Device::CPU)
        operator delete(buffer);
    return ret;
}

template <SupportedDType LogicalDType = float>
Tensor<LogicalDType> ones(const TensorShape& shape, Device device=Device::CPU)
{
    size_t numel = numel_from_shape(shape), size = sizeof(LogicalDType) * numel;
    LogicalDType* buffer = static_cast<LogicalDType*>(operator new(size));
    for (size_t i = 0;i < numel;i++) buffer[i] = 1;

    Tensor<LogicalDType> ret = Tensor<LogicalDType>(buffer, shape, device, false, true);
    if (device != Device::CPU)
        operator delete(buffer);
    return ret;
}

template <CUDAFloatingPoint LogicalDType = float>
Tensor<LogicalDType> rand(const TensorShape& shape, Device device=Device::CPU)
{
    std::uniform_real_distribution<LogicalDType> dist(0.0, 1.0);
    size_t numel = numel_from_shape(shape), size = sizeof(LogicalDType) * numel;
    LogicalDType* buffer = static_cast<LogicalDType*>(operator new(size));
    for (size_t i = 0;i < numel;i++) buffer[i] = dist(get_random_engine());

    Tensor<LogicalDType> ret = Tensor<LogicalDType>(buffer, shape, device, false, true);
    if (device != Device::CPU)
        operator delete(buffer);
    return ret;
}

template <CUDAFloatingPoint LogicalDType = float>
Tensor<LogicalDType> randn(const TensorShape& shape, Device device=Device::CPU)
{
    std::normal_distribution<LogicalDType> dist(0.0, 1.0);
    size_t numel = numel_from_shape(shape), size = sizeof(LogicalDType) * numel;
    LogicalDType* buffer = static_cast<LogicalDType*>(operator new(size));
    for (size_t i = 0;i < numel;i++) buffer[i] = dist(get_random_engine());

    Tensor<LogicalDType> ret = Tensor<LogicalDType>(buffer, shape, device, false, true);
    if (device != Device::CPU)
        operator delete(buffer);
    return ret;
}

template <Interger LogicalDType = int>
Tensor<LogicalDType> randint(LogicalDType min, LogicalDType max, const TensorShape& shape, Device device=Device::CPU)
{
    std::uniform_int_distribution<LogicalDType> dist(min, max);
    size_t numel = numel_from_shape(shape), size = sizeof(LogicalDType) * numel;
    LogicalDType* buffer = static_cast<LogicalDType*>(operator new(size));
    for (size_t i = 0;i < numel;i++) buffer[i] = dist(get_random_engine());

    Tensor<LogicalDType> ret = Tensor<LogicalDType>(buffer, shape, device, false, true);
    if (device != Device::CPU)
        operator delete(buffer);
    return ret;
}

template <typename ReturnDType=void, typename OriginalDType>
requires (std::is_same_v<ReturnDType, void> || SupportedDType<ReturnDType>)
         && SupportedDType<OriginalDType>
Tensor<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
zeros_like(const Tensor<OriginalDType>& another, std::optional<Device> device=std::nullopt)
{
    return zeros<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
    (another.shape, device.value_or(another.get_device()));
}

template <typename ReturnDType=void, typename OriginalDType>
requires (std::is_same_v<ReturnDType, void> || SupportedDType<ReturnDType>)
         && SupportedDType<OriginalDType>
Tensor<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
ones_like(const Tensor<OriginalDType>& another, std::optional<Device> device=std::nullopt)
{
    return ones<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
    (another.shape, device.value_or(another.get_device()));
}

template <typename ReturnDType=void, typename OriginalDType>
requires (std::is_same_v<ReturnDType, void> || CUDAFloatingPoint<ReturnDType>)
         && CUDAFloatingPoint<OriginalDType>
Tensor<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
rand_like(const Tensor<OriginalDType>& another, std::optional<Device> device=std::nullopt)
{
    return rand<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
    (another.shape, device.value_or(another.get_device()));
}

template <typename ReturnDType=void, typename OriginalDType>
requires (std::is_same_v<ReturnDType, void> || CUDAFloatingPoint<ReturnDType>)
         && CUDAFloatingPoint<OriginalDType>
Tensor<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
randn_like(const Tensor<OriginalDType>& another, std::optional<Device> device=std::nullopt)
{
    return randn<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
    (another.shape, device.value_or(another.get_device()));
}

template <typename ReturnDType=void, typename OriginalDType>
requires (std::is_same_v<ReturnDType, void> || Interger<ReturnDType>)
         && Interger<OriginalDType>
Tensor<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
randint_like(
    std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType> min,
    std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType> max,
    const Tensor<OriginalDType>& another, std::optional<Device> device=std::nullopt)
{
    return randint<std::conditional_t<std::is_same_v<ReturnDType, void>, OriginalDType, ReturnDType>>
    (min, max, another.shape, device.value_or(another.get_device()));
}

}; // namespace cudaoplib