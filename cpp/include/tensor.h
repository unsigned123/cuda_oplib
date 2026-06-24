#pragma once

#include "kernel_tensor.h"
#include "elementwise.h"
#include "reduce.h"

#include "tensor_defines.h"
#include "tensor_utils.h"

#include <type_traits>
#include <vector>
#include <concepts>
#include <ranges>
#include <initializer_list>
#include <stdexcept>

#include <algorithm>

#include <string>
#include <iostream>
#include <iomanip>

namespace cudaoplib
{
// Classes

template <SupportedDType LogicalDType>
class Tensor
{
private:
    using InternalDType = LogicalDType;
    cudaoplib_kernel::Tensor raw;

    Tensor(cudaoplib_kernel::Tensor _raw) { this->raw = _raw; }
    Device binary_device(Device another_device) const;

    void swap(Tensor<LogicalDType>& another);

public:
    TensorShape binary_broadcast(const TensorShape& another_shape) const;
    template <SupportedDType> friend class Tensor;

    // Constructors and deconstructors

    Tensor(LogicalDType* data, const TensorShape& shape, Device device=Device::CPU, bool need_copy=true, bool take_over=false);
    Tensor(LogicalDType value, Device device=Device::GPU);
    template<std::ranges::range Range> Tensor(const Range& range, _ForceToUseRangeConstructor dummy={});
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(1)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(2)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(3)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(4)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(5)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(6)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(7)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(8)
    Tensor(const Tensor<LogicalDType>& another);
    Tensor(Tensor<LogicalDType>&& another);
    virtual ~Tensor();


    // Tensor transfer

    Tensor<LogicalDType> to_device(Device device) const;
    Tensor<LogicalDType> to_gpu() const { return this->to_device(Device::GPU); }
    Tensor<LogicalDType> to_cpu() const { return this->to_device(Device::CPU); }



    // Operators

    Tensor<LogicalDType> operator+(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>;
    Tensor<LogicalDType> operator-(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>;
    Tensor<LogicalDType> operator*(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>;
    Tensor<LogicalDType> operator/(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>;
    Tensor<LogicalDType> operator%(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>;

    Tensor<LogicalDType>& operator+=(const Tensor<LogicalDType>& another) requires ArithmeticType<LogicalDType> { *this = *this + another; return *this; }
    Tensor<LogicalDType>& operator-=(const Tensor<LogicalDType>& another) requires ArithmeticType<LogicalDType> { *this = *this - another; return *this; }
    Tensor<LogicalDType>& operator*=(const Tensor<LogicalDType>& another) requires ArithmeticType<LogicalDType> { *this = *this * another; return *this; }
    Tensor<LogicalDType>& operator/=(const Tensor<LogicalDType>& another) requires ArithmeticType<LogicalDType> { *this = *this / another; return *this; }
    Tensor<LogicalDType>& operator%=(const Tensor<LogicalDType>& another) requires ArithmeticType<LogicalDType> { *this = *this % another; return *this; }

    Tensor<bool> operator==(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator!=(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator<(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator<=(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator>(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator>=(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator&&(const Tensor<LogicalDType>& another) const;
    Tensor<bool> operator||(const Tensor<LogicalDType>& another) const;

    Tensor<LogicalDType>& operator=(const Tensor<LogicalDType>& another);
    Tensor<LogicalDType>& operator=(Tensor<LogicalDType>&& another);
    friend std::ostream& operator<< <LogicalDType>(std::ostream& stream, const Tensor<LogicalDType>& tensor);
    
    // Type convertion
    Tensor<float> to_float32() const;
    Tensor<__half> to_float16() const;
    Tensor<int> to_int32() const;
    Tensor<int8_t> to_int8() const;
    Tensor<bool> to_bool() const;

    explicit operator Tensor<float>() const { return to_float32(); }
    explicit operator Tensor<__half>() const { return to_float16(); }
    explicit operator Tensor<int>() const { return to_int32(); }
    explicit operator Tensor<int8_t>() const { return to_int8(); }
    explicit operator Tensor<bool>() const { return to_bool(); }

    // Single ops
    Tensor<LogicalDType> sum(const std::vector<int>& dims, bool keepdim=false) const;
    Tensor<LogicalDType> sum(int dim, bool keepdim=false) const;

    // Simple getters
    TensorShape shape() const { return this->raw.shape; }
    const LogicalDType* data() const { return static_cast<const LogicalDType*>(this->raw.data); }
    LogicalDType* data() { return static_cast<LogicalDType*>(this->raw.data); }
    Device get_device() const { return this->raw.device; }
    bool owns_data() const { return this->raw.owns_data; }
    size_t numel() const { return this->raw.numel; }
    cudaoplib_kernel::Tensor get_raw() const { return this->raw; }
};

// Implementations
template <SupportedDType LogicalDType>
Tensor<LogicalDType>::Tensor(LogicalDType* data, const TensorShape& shape, Device device, bool need_copy, bool take_over)
{
    for (size_t dim : shape)
    {
        if (dim <= 0)
            throw std::runtime_error("FATAL: cudaoplib::Tensor constructor failed: Invalid tensor shape.");
    }

    if (device == Device::CPU)
        this->raw = cudaoplib_kernel::create_cpu_tensor_from_cpu_data(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), data, shape, need_copy);
    else
    {
        if (!need_copy)
            throw std::runtime_error("FATAL: cudaoplib::Tensor constructor failed: Copy is required when creating a GPU tensor.");
        this->raw = cudaoplib_kernel::create_gpu_tensor_from_cpu_data(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), data, shape);
    }

    if (device == Device::CPU && take_over)
        this->raw.owns_data = true;
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType>::Tensor(LogicalDType value, Device device)
{
    LogicalDType* data = new LogicalDType[1];
    data[0] = value;

    if (device == Device::CPU)
       this->raw = cudaoplib_kernel::create_cpu_tensor_from_cpu_data(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), data, {1}, true);
    else
    {
        this->raw = cudaoplib_kernel::create_gpu_tensor_from_cpu_data(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), data, {1});
    }

    delete[] data;
}

template <SupportedDType LogicalDType>
template<std::ranges::range Range> 
Tensor<LogicalDType>::Tensor(const Range& range, [[maybe_unused]] _ForceToUseRangeConstructor)
{
    auto shape = deduce_shape(range);
    raw = cudaoplib_kernel::create_empty_cpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);
    fill_data(range, static_cast<LogicalDType*>(raw.data));
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType>::Tensor(const Tensor<LogicalDType>& another)
{
    if (another.raw.device == Device::CPU)
    {
        this->raw = cudaoplib_kernel::copy_to_cpu_from_cpu(another.raw);
    }
    else
    {
        this->raw = cudaoplib_kernel::copy_to_gpu_from_gpu(another.raw);
    }
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType>::Tensor(Tensor<LogicalDType>&& another)
{
    this->raw.device = another.raw.device;
    this->raw.dtype = another.raw.dtype;
    this->raw.numel = another.raw.numel;
    this->raw.shape = std::move(another.raw.shape);
    this->raw.stride = std::move(another.raw.stride);
    this->raw.is_contiguous = another.raw.is_contiguous;

    this->raw.owns_data = another.raw.owns_data;
    another.raw.owns_data = false;

    this->raw.data = another.raw.data;
    another.raw.data = nullptr;
}

template <SupportedDType LogicalDType>
void Tensor<LogicalDType>::swap(Tensor<LogicalDType>& another)
{
    std::swap(this->raw, another.raw);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType>& Tensor<LogicalDType>::operator=(const Tensor<LogicalDType>& another)
{
    Tensor<LogicalDType> temp(another);
    temp.swap(*this);
    return *this;
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType>& Tensor<LogicalDType>::operator=(Tensor<LogicalDType>&& another)
{
    if (this != &another)
    {
        if (this->owns_data())
        {
            if (this->get_device() == Device::CPU)
                cudaoplib_kernel::free_cpu_tensor(this->raw);
            else
                cudaoplib_kernel::free_gpu_tensor(this->raw);
        }

        this->raw.device = another.raw.device;
        this->raw.dtype = another.raw.dtype;
        this->raw.numel = another.raw.numel;
        this->raw.shape = std::move(another.raw.shape);
        this->raw.stride = std::move(another.raw.stride);
        this->raw.is_contiguous = another.raw.is_contiguous;

        this->raw.owns_data = another.raw.owns_data;
        another.raw.owns_data = false;

        this->raw.data = another.raw.data;
        another.raw.data = nullptr;
    }
    return *this;
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType>::~Tensor()
{
    if (this->owns_data())
    {
        if (this->get_device() == Device::CPU)
            cudaoplib_kernel::free_cpu_tensor(this->raw);
        else
            cudaoplib_kernel::free_gpu_tensor(this->raw);
    }
}

template <SupportedDType LogicalDType>
TensorShape Tensor<LogicalDType>::binary_broadcast(const TensorShape& another_shape) const
{
    const TensorShape& this_shape = this->raw.shape;

    size_t this_dims = this_shape.size(), another_dims = another_shape.size();

    size_t target_dims = std::max(this_dims, another_dims);

    TensorShape broadcast_shape(target_dims, 1);

    for (size_t i = 0;i < target_dims;i++)
    {
        if (i < this_dims && i < another_dims)
            if (this_shape[this_dims - i - 1] != another_shape[another_dims - i - 1] && another_shape[another_dims - i - 1] > 1 && this_shape[this_dims - i - 1] > 1)
                throw std::runtime_error("FATAL: cudaoplib::Tensor::binary_broadcast Broadcast failed: dimensions mismatch.");
 
        if (i < this_dims)
            broadcast_shape[target_dims - i - 1] = std::max(this_shape[this_dims - i - 1], broadcast_shape[target_dims - i - 1]);

        if (i < another_dims)
            broadcast_shape[target_dims - i - 1] = std::max(another_shape[another_dims - i - 1], broadcast_shape[target_dims - i - 1]);
    }

    return broadcast_shape;
}

template <SupportedDType LogicalDType>
Device Tensor<LogicalDType>::binary_device(Device another_device) const
{
    if (this->raw.device != another_device)
        throw std::runtime_error("FATAL: cudaoplib::Tensor::binary_device: tensors are on different devices.");

    return another_device;
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::to_device(Device device) const
{
    if (device == this->raw.device)
        throw std::runtime_error("FATAL: cudaoplib::Tensor::to_device: Invalid destination: the same as the source.");
    
    if (device == Device::CPU)
        return Tensor<LogicalDType>(cudaoplib_kernel::copy_to_cpu_from_gpu(this->raw));
    else
        return Tensor<LogicalDType>(cudaoplib_kernel::copy_to_gpu_from_cpu(this->raw));
}

template <SupportedDType LogicalDType>
Tensor<float> Tensor<LogicalDType>::to_float32() const
{
    if (Device::CPU == this->raw.device)
    {
        cudaoplib_kernel::Tensor new_raw = cudaoplib_kernel::create_empty_cpu_tensor(DType::Float32, this->raw.shape);
        for (size_t i = 0;i < this->raw.numel;i++)
            static_cast<float*>(new_raw.data)[i] = static_cast<const Tensor<LogicalDType>::InternalDType*>(this->raw.data)[i];
        return Tensor<float>(new_raw);
    }
    else throw std::runtime_error("Type conversion on gpu not implemented yet.");
}

template <SupportedDType LogicalDType>
Tensor<__half> Tensor<LogicalDType>::to_float16() const
{
    if (Device::CPU == this->raw.device)
    {
        cudaoplib_kernel::Tensor new_raw = cudaoplib_kernel::create_empty_cpu_tensor(DType::Float16, this->raw.shape);
        for (size_t i = 0;i < this->raw.numel;i++)
            static_cast<__half*>(new_raw.data)[i] = static_cast<const Tensor<LogicalDType>::InternalDType*>(this->raw.data)[i];
        return Tensor<__half>(new_raw);
    }
    else throw std::runtime_error("Type conversion on gpu not implemented yet.");
}

template <SupportedDType LogicalDType>
Tensor<int> Tensor<LogicalDType>::to_int32() const
{
    if (Device::CPU == this->raw.device)
    {
        cudaoplib_kernel::Tensor new_raw = cudaoplib_kernel::create_empty_cpu_tensor(DType::Int32, this->raw.shape);
        for (size_t i = 0;i < this->raw.numel;i++)
            static_cast<int*>(new_raw.data)[i] = static_cast<const Tensor<LogicalDType>::InternalDType*>(this->raw.data)[i];
        return Tensor<int>(new_raw);
    }
    else throw std::runtime_error("Type conversion on gpu not implemented yet.");
}

template <SupportedDType LogicalDType>
Tensor<int8_t> Tensor<LogicalDType>::to_int8() const
{
    if (Device::CPU == this->raw.device)
    {
        cudaoplib_kernel::Tensor new_raw = cudaoplib_kernel::create_empty_cpu_tensor(DType::Int8, this->raw.shape);
        for (size_t i = 0;i < this->raw.numel;i++)
            static_cast<int8_t*>(new_raw.data)[i] = static_cast<const Tensor<LogicalDType>::InternalDType*>(this->raw.data)[i];
        return Tensor<int8_t>(new_raw);
    }
    else throw std::runtime_error("Type conversion on gpu not implemented yet.");
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::to_bool() const
{
    if (Device::CPU == this->raw.device)
    {
        cudaoplib_kernel::Tensor new_raw = cudaoplib_kernel::create_empty_cpu_tensor(DType::Bool, this->raw.shape);
        for (size_t i = 0;i < this->raw.numel;i++)
            static_cast<bool*>(new_raw.data)[i] = (bool)(static_cast<const Tensor<LogicalDType>::InternalDType*>(this->raw.data)[i]);
        return Tensor<bool>(new_raw);
    }
    else throw std::runtime_error("Type conversion on gpu not implemented yet.");
}

template <SupportedDType LogicalDType>
std::ostream& operator<<(std::ostream& stream, const Tensor<LogicalDType>& input)
{
    auto print_cpu_tensor = [&](const Tensor<LogicalDType>& tensor) {
        
        std::ios::fmtflags old_flags = stream.flags();
        std::streamsize old_precision = stream.precision();

        const int INDENT_SPACES = 2;
        const int FIXED_WIDTH = 10;
        [[maybe_unused]] const int PRECISION = 4;

        if (tensor.raw.numel == 0)
        {   
            stream << "Tensor()" << std::endl;
            return;
        }

        stream << "Tensor(shape=[";
        for (size_t i = 0;i < tensor.raw.shape.size();i++)
        {
            if (i > 0) stream << ", ";
            stream << tensor.raw.shape[i];
        }
        stream << "], dtype=";
        if constexpr (std::is_same_v<LogicalDType, float>) stream << "float32";
        else if constexpr (std::is_same_v<LogicalDType, __half>) stream << "float16";
        else if constexpr (std::is_same_v<LogicalDType, int>) stream << "int32";
        else if constexpr (std::is_same_v<LogicalDType, int8_t>) stream << "int8";
        else if constexpr (std::is_same_v<LogicalDType, bool>) stream << "bool";
        else stream << "UNSUPPORTED DTYPE(" << typeid(LogicalDType).name() << ")";

        stream << ", device=" << (input.raw.device == Device::CPU ? "\"CPU\"" : "\"GPU\"");
        stream << ")" << std::endl;

        auto print_recursive = [&](this auto self, size_t current_dim, size_t offset, int indent = 0) {
            std::string prefix(indent * INDENT_SPACES, ' ');

            // baseline condition
            if (current_dim == tensor.raw.shape.size() - 1) {
                stream << prefix << "[";
                for (size_t i = 0;i < tensor.raw.shape[current_dim];i++) {
                    if (i > 0) stream << ", ";
                    size_t index = offset + i * tensor.raw.stride[current_dim];

                    if constexpr (std::is_same_v<LogicalDType, int8_t>) {
                        stream << std::setw(FIXED_WIDTH) << static_cast<int>(static_cast<LogicalDType*>(tensor.raw.data)[index]);
                    }
                    else if constexpr (std::is_same_v<LogicalDType, bool>) {
                        stream << std::setw(FIXED_WIDTH) << ((static_cast<bool*>(tensor.raw.data))[index] ? "true" : "false");
                    }
                    else if constexpr (requires { requires CUDAFloatingPoint<LogicalDType>; }) {
                        stream << std::fixed << std::setprecision(PRECISION) << std::setw(FIXED_WIDTH)
                        << static_cast<LogicalDType*>(tensor.raw.data)[index];
                    } else {
                        stream << std::setw(FIXED_WIDTH) << static_cast<LogicalDType*>(tensor.raw.data)[index];
                    }
                }
                stream << "]";
                return;
            }
            
            // resursive condition
            stream << prefix << "[";
            for (size_t i = 0;i < tensor.raw.shape[current_dim];i++) {
                if (i > 0) {
                    stream << ",";
                }
                stream << std::endl;
                self(current_dim + 1, offset + i * tensor.raw.stride[current_dim], indent + 1);
            }
            stream << std::endl << prefix << "]";
        };

        print_recursive(0, 0);

        stream.flags(old_flags);
        stream.precision(old_precision);   
    };

    if (input.get_device() == Device::CPU)
        print_cpu_tensor(input);
    else
    {
        Tensor<LogicalDType> cpu_copy = input.to_cpu();
        print_cpu_tensor(cpu_copy);
    }

    return stream;
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::operator+(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);

    cudaoplib_kernel::add(this->raw, another.raw, result_raw);
    return Tensor<LogicalDType>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::operator-(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);

    cudaoplib_kernel::sub(this->raw, another.raw, result_raw);
    return Tensor<LogicalDType>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::operator*(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);

    cudaoplib_kernel::mul(this->raw, another.raw, result_raw);
    return Tensor<LogicalDType>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::operator/(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);

    cudaoplib_kernel::div(this->raw, another.raw, result_raw);
    return Tensor<LogicalDType>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::operator%(const Tensor<LogicalDType>& another) const requires ArithmeticType<LogicalDType>
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);

    cudaoplib_kernel::mod(this->raw, another.raw, result_raw);
    return Tensor<LogicalDType>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator==(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::eq(this->raw, another.raw, result_raw);
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator!=(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::neq(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator<(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::lt(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator<=(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::le(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator>(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::gt(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator>=(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::ge(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator&&(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::logical_and(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<bool> Tensor<LogicalDType>::operator||(const Tensor<LogicalDType>& another) const
{   
    Device device = binary_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(DType::Bool, shape);

    cudaoplib_kernel::logical_or(this->raw, another.raw, result_raw);
    
    return Tensor<bool>(result_raw);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::sum(const std::vector<int>& dims, bool keepdim) const
{
    if (this->raw.device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    if (dims.size() == 0)
        throw std::runtime_error("FATAL: cudaoplib::Tensor::sum: empty dims vector.");

    TensorShape shape = this->shape();
    cudaoplib_kernel::Tensor input = cudaoplib_kernel::create_gpu_tensor_from_gpu_data(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}),
            this->raw.data, this->raw.shape);
    cudaoplib_kernel::Tensor output;

    for (auto dim : dims)
    {
        if (dim < 0 || dim >= MAX_TENSOR_DIM)
            throw std::runtime_error("FATAL: cudaoplib::Tensor::sum: tensor dims out of range.");

        shape[dim] = 1;
        output = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);
        
        cudaoplib_kernel::sum(input, output, dim);
        std::swap(input, output);

        cudaoplib_kernel::free_gpu_tensor(output);
    }

    std::swap(input, output);

    if (!keepdim)
    {
        TensorShape new_shape;
        for (size_t i = 0;i < output.shape.size();i++)
        {
            if (std::find(dims.begin(), dims.end(), i) == dims.end())
                new_shape.push_back(output.shape[i]);
        }
        cudaoplib_kernel::calc_stride_and_numel(output.stride, new_shape);
        output.shape = new_shape;
    }

    return Tensor<LogicalDType>(output);
}

template <SupportedDType LogicalDType>
Tensor<LogicalDType> Tensor<LogicalDType>::sum(int dim, bool keepdim) const
{
    if (this->raw.device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = this->shape();

    cudaoplib_kernel::Tensor output;

    if (dim < 0 || dim >= MAX_TENSOR_DIM)
        throw std::runtime_error("FATAL: cudaoplib::Tensor::sum: tensor dims out of range.");

    shape[dim] = 1;
    output = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(Tensor<LogicalDType>::InternalDType{}), shape);

    cudaoplib_kernel::sum(this->raw.data, output, dim);

    if (!keepdim)
    {
        output.shape.erase(output.shape.begin() + dim);
        cudaoplib_kernel::calc_stride_and_numel(output.stride, output.shape);
    }

    return Tensor<LogicalDType>(output);
}

} // namespace cudaoplib