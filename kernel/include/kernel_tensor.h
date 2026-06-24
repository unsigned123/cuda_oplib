#pragma once
#include <vector>
#include <cuda/std/array>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cstdint>


#define MAX_TENSOR_DIM 8

namespace cudaoplib_kernel
{

// Fundamental Types
enum class DType
{
    Float32,
    Float16,
    Int32,
    Int8,
    Bool
};

enum class Device
{
    CPU,
    GPU
};

using TensorShape = cuda::std::array<int64_t, MAX_TENSOR_DIM>;
using TensorStride = cuda::std::array<int64_t, MAX_TENSOR_DIM>;

struct Tensor
{
    void* data = nullptr;
    std::vector<size_t> shape;
    std::vector<int64_t> stride;
    bool owns_data = false;
    bool is_contiguous = true;
    DType dtype;
    Device device;
    size_t numel = 0;

    bool is_gpu() const { return device != Device::CPU; }
    TensorShape pad_shape() const
    {
        TensorShape padded_shape;
        padded_shape.fill(1);

        int offset = MAX_TENSOR_DIM - shape.size();
        for (size_t i = 0;i < shape.size();i++)
        {
            padded_shape[i + offset] = shape[i];
        }
        return padded_shape;
    }

    TensorStride get_stride() const
    {
        TensorStride processed_stride;
        processed_stride.fill(0);

        int offset = MAX_TENSOR_DIM - shape.size();
        for (size_t i = 0;i < shape.size();i++)
        {
            if (shape[i] == 1)
                processed_stride[i + offset] = 0;
            else
                processed_stride[i + offset] = stride[i];
        }
        return processed_stride;
    }
};

// Utilities

inline TensorShape broadcast_shape(const TensorShape& a, const TensorShape& b)
{
    TensorShape broadcast_shape;

    for (int i = 0;i < MAX_TENSOR_DIM;i++)
    {
        if (a[i] == 1 && b[i] > 1)
            broadcast_shape[i] = b[i];
        else if (b[i] == 1 && a[i] > 1)
            broadcast_shape[i] = a[i];
        else if (a[i] == b[i])
            broadcast_shape[i] = a[i];
        else
            throw std::runtime_error("FATAL: cudaoplib_kernel::broadcast_shape failed. Shape mismatch.");
    }

    return broadcast_shape;
}

inline int get_dtype_size(DType dtype)
{
    int size = 0;
    switch(dtype) {
        case DType::Float32: size = sizeof(float); break;
        case DType::Float16: size = sizeof(__half); break;
        case DType::Int32: size = sizeof(int); break;
        case DType::Int8: size = sizeof(int8_t); break;
        case DType::Bool: size = sizeof(bool); break;
        default: throw std::runtime_error("FATAL: cudaoplib_kernel::create_cpu_tensor_copy failed : Unsupported dtype"); break;
    }
    return size;
}

inline size_t calc_stride_and_numel(std::vector<int64_t>& stride, const std::vector<size_t>& shape)
{
    stride.resize(shape.size(), 1);
    
    size_t cumulated = 1, numel = 0;
    for (int i = shape.size() - 1;i >= 0;i--)
    {
        stride[i] = cumulated;
        cumulated *= shape[i];
    }
    numel = cumulated;

    return numel;
}

// Tensor Management
Tensor create_empty_cpu_tensor(DType dtype, const std::vector<size_t>& shape);
Tensor create_empty_gpu_tensor(DType dtype, const std::vector<size_t>& shape);
Tensor create_cpu_tensor_from_cpu_data(DType dtype, void* data, const std::vector<size_t>& shape, bool copy_from_cpu);
Tensor create_gpu_tensor_from_cpu_data(DType dtype, void* data, const std::vector<size_t>& shape);
Tensor create_gpu_tensor_from_gpu_data(DType dtype, void* data, const std::vector<size_t>& shape);

Tensor copy_to_cpu_from_cpu(const Tensor& tensor);
Tensor copy_to_gpu_from_gpu(const Tensor& tensor);

Tensor copy_to_cpu_from_gpu(const Tensor& tensor);
Tensor copy_to_gpu_from_cpu(const Tensor& tensor);

void free_gpu_tensor(Tensor& tensor);
void free_cpu_tensor(Tensor& tensor);

} // namespace cudaoplib_kernel
