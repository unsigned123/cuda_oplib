#pragma once

#include "kernel_tensor.h"
#include "elementwise.h"

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

template <SupportedDType T>
class Tensor
{
private:
    cudaoplib_kernel::Tensor raw;

    Tensor(cudaoplib_kernel::Tensor _raw) { this->raw = _raw; }
    TensorShape binary_broadcast(const TensorShape& another_shape) const;
    Device binaray_device(Device another_device) const;
    
public:
    // Constructors and deconstructors

    Tensor(T* data, const TensorShape& shape, Device device=Device::CPU, bool need_copy=true);
    Tensor(T value, Device device=Device::CPU);
    template<std::ranges::range Range> Tensor(const Range& range, _ForceToUseRangeConstructor dummy={});
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(1)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(2)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(3)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(4)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(5)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(6)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(7)
    TENSOR_INIALIZER_LIST_CONSTRUCTOR(8)
    Tensor(const Tensor<T>& another);
    Tensor(Tensor<T>&& another);
    virtual ~Tensor();
    



    // Tensor transfer

    Tensor<T> to_device(Device device) const;
    Tensor<T> to_gpu() const { return this->to_device(Device::GPU); }
    Tensor<T> to_cpu() const { return this->to_device(Device::CPU); }



    // Operators

    Tensor<T> operator+(const Tensor<T>& another) const;
    Tensor<T> operator=(const Tensor<T>&) = delete;
    Tensor<T> operator=(Tensor<T>&&) = delete;
    friend std::ostream& operator<< <T>(std::ostream& stream, const Tensor<T>& tensor);
    


    // Simple getters

    Device get_device() const { return this->raw.device; }
    bool owns_data() const { return this->raw.owns_data; }
};

// Implementations
template <SupportedDType T>
Tensor<T>::Tensor(T* data, const TensorShape& shape, Device device, bool need_copy)
{
    for (size_t dim : shape)
    {
        if (dim <= 0)
            throw std::runtime_error("FATAL: cudaoplib::Tensor constructor failed: Invalid tensor shape.");
    }

    if (device == Device::CPU)
        this->raw = cudaoplib_kernel::create_cpu_tensor_from_cpu_data(dtype_to_enum(T{}), data, shape, need_copy);
    else
    {
        if (!need_copy)
            throw std::runtime_error("FATAL: cudaoplib::Tensor constructor failed: Copy is required when creating a GPU tensor.");
        this->raw = cudaoplib_kernel::create_gpu_tensor_from_cpu_data(dtype_to_enum(T{}), data, shape);
    }
}

template <SupportedDType T>
Tensor<T>::Tensor(T value, Device device)
{
    T* data = new T[1];
    data[0] = value;

    if (device == Device::CPU)
       this->raw = cudaoplib_kernel::create_cpu_tensor_from_cpu_data(dtype_to_enum(T{}), data, {1}, true);
    else
    {
        this->raw = cudaoplib_kernel::create_gpu_tensor_from_cpu_data(dtype_to_enum(T{}), data, {1});
    }

    delete[] data;
}

template <SupportedDType T>
template<std::ranges::range Range> 
Tensor<T>::Tensor(const Range& range, [[maybe_unused]] _ForceToUseRangeConstructor)
{
    auto shape = deduce_shape(range);
    raw = cudaoplib_kernel::create_empty_cpu_tensor(dtype_to_enum(T{}), shape);
    fill_data(range, static_cast<T*>(raw.data));
}

template <SupportedDType T>
Tensor<T>::Tensor(const Tensor<T>& another)
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

template <SupportedDType T>
Tensor<T>::Tensor(Tensor<T>&& another)
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

template <SupportedDType T>
Tensor<T>::~Tensor()
{
    if (this->owns_data())
    {
        if (this->get_device() == Device::CPU)
        {
            cudaoplib_kernel::free_cpu_tensor(this->raw);
        }
        else
        {
            cudaoplib_kernel::free_gpu_tensor(this->raw);
        }
    }
}

template <SupportedDType T>
TensorShape Tensor<T>::binary_broadcast(const TensorShape& another_shape) const
{
    const TensorShape& this_shape = this->raw.shape;

    size_t target_dims = std::max(this_shape.size(), another_shape.size());

    TensorShape broadcast_shape(target_dims, 1);

    for (size_t i = target_dims - 1;;i--)
    {
        if (i < this_shape.size() && i < this_shape.size())
            if (this_shape[i] != another_shape[i] && another_shape[i] > 1 && this_shape[i] > 1)
                throw std::runtime_error("FATAL: cudaoplib::Tensor::binary_broadcast Broadcast failed: dimensions mismatch.");

        if (i < this_shape.size())
            broadcast_shape[i] = std::max(this_shape[i], broadcast_shape[i]);

        if (i < another_shape.size())
            broadcast_shape[i] = std::max(another_shape[i], broadcast_shape[i]);

        if (i == 0)
            break;
    }

    return broadcast_shape;
}

template <SupportedDType T>
Device Tensor<T>::binaray_device(Device another_device) const
{
    if (this->raw.device != another_device)
        throw std::runtime_error("FATAL: cudaoplib::Tensor::binaray_device: tensors are on different devices.");

    return another_device;
}

template <SupportedDType T>
Tensor<T> Tensor<T>::to_device(Device device) const
{
    if (device == this->raw.device)
        throw std::runtime_error("FATAL: cudaoplib::Tensor::to_device: Invalid destination: the same as the source.");
    
    if (device == Device::CPU)
    {
        return Tensor<T>(cudaoplib_kernel::copy_to_cpu_from_gpu(this->raw));
    }
    else
    {
        return Tensor<T>(cudaoplib_kernel::copy_to_gpu_from_cpu(this->raw));
    }
}

template <SupportedDType T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& another) const
{   
    Device device = binaray_device(another.raw.device);
    if (device == Device::CPU)
        throw std::runtime_error("FATAL: operators on CPU unsupported.");

    TensorShape shape = binary_broadcast(another.raw.shape);
    cudaoplib_kernel::Tensor result_raw = cudaoplib_kernel::create_empty_gpu_tensor(dtype_to_enum(T{}), shape);

    cudaoplib_kernel::add(this->raw, another.raw, result_raw);
    return Tensor<T>(result_raw);
}

template <SupportedDType T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T>& input)
{
    auto print_cpu_tensor = [&](const Tensor<T>& tensor) {
        
        std::ios::fmtflags old_flags = stream.flags();
        std::streamsize old_precision = stream.precision();

        const int INDENT_SPACES = 2;
        const int FIXED_WIDTH = 10;
        [[maybe_unused]] const int PRECISION = 4;

        if (tensor.raw.numel == 0)
        {   
            stream << "Tensor()" << std::endl;
        }

        stream << "Tensor(shape=[";
        for (size_t i = 0;i < tensor.raw.shape.size();i++)
        {
            if (i > 0) stream << ", ";
            stream << tensor.raw.shape[i];
        }
        stream << "], dtype=";
        if constexpr (std::is_same_v<T, float>) stream << "float32";
        else if constexpr (std::is_same_v<T, __half>) stream << "float16";
        else if constexpr (std::is_same_v<T, int>) stream << "int32";
        else if constexpr (std::is_same_v<T, int8_t>) stream << "int8";
        else stream << "UNSUPPORTED DTYPE(" << typeid(T).name() << ")";
        stream << ")" << std::endl;

        auto print_recursive = [&](this auto self, size_t current_dim, size_t offset, int indent = 0) {
            std::string prefix(indent * INDENT_SPACES, ' ');

            // baseline condition
            if (current_dim == tensor.raw.shape.size() - 1) {
                stream << prefix << "[";
                for (size_t i = 0;i < tensor.raw.shape[current_dim];i++) {
                    if (i > 0) stream << ", ";
                    size_t index = offset + i * tensor.raw.stride[current_dim];

                    if constexpr (std::is_floating_point_v<T>) {
                        stream << std::fixed << std::setprecision(PRECISION) << std::setw(FIXED_WIDTH)
                        << static_cast<T*>(tensor.raw.data)[index];
                    } else {
                        stream << std::setw(FIXED_WIDTH) << static_cast<T*>(tensor.raw.data)[index];
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
        Tensor<T> cpu_copy = input.to_cpu();
        print_cpu_tensor(cpu_copy);
    }

    return stream;
}

} // namespace cudaoplib