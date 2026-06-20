#pragma once
#include <vector>
#include <cuda/std/array>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cstdint>


#define MAX_TENSOR_DIM 8

namespace cudaoplib
{

// Fundamental Types
enum class DType
{
    Float32,
    Float16,
    Int32,
    Int8
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
    DType dtype;
    Device device;
    size_t numel;

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
        processed_stride.fill(1);

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

inline TensorShape boardcast_shape(const TensorShape& a, const TensorShape& b)
{
    TensorShape boardcast_shape;

    for (int i = 0;i < MAX_TENSOR_DIM;i++)
    {
        if (a[i] == 1 && b[i] > 1)
            boardcast_shape[i] = b[i];
        else if (b[i] == 1 && a[i] > 1)
            boardcast_shape[i] = a[i];
        else
            throw std::runtime_error("FATAL: cudaoplib::boardcast_shape failed. Shape mismatch.");
    }

    return boardcast_shape;
}

inline int get_dtype_size(DType dtype)
{
    int size = 0;
    switch(dtype) {
        case DType::Float32: size = sizeof(float); break;
        case DType::Float16: size = sizeof(__half); break;
        case DType::Int32: size = sizeof(int); break;
        case DType::Int8: size = sizeof(int8_t); break;
        default: throw std::runtime_error("FATAL: create_cpu_tensor_copy failed : Unsupported dtype"); break;
    }
    return size;
}


// Tensor Management
Tensor create_cpu_tensor(DType dtype, void* data, std::vector<size_t> shape, bool copy_from_cpu);

Tensor copy_to_cpu(const Tensor& tensor);
Tensor copy_to_gpu(const Tensor& tensor);


}; // namespace cudaoplib
