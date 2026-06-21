#include "cuda_utils.h"
#include "kernel_tensor.h"
#include <memory>
#include <stdexcept>

#include <stdio.h>

namespace cudaoplib_kernel
{

Tensor create_empty_cpu_tensor(DType dtype, const std::vector<size_t>& shape)
{
    int size = get_dtype_size(dtype);

    int numel = 0, cumulated = 1;
    std::vector<int64_t> stride;
    stride.resize(shape.size(), 1);
    
    for (int i = shape.size() - 1;i >= 0;i--)
    {
        stride[i] = cumulated;
        cumulated *= shape[i];
    }
    numel = cumulated;

    void* tensor_buffer = nullptr;
    
    tensor_buffer = operator new(numel * size);
    memset(tensor_buffer, 0, numel * size);

    Tensor tensor;
    tensor.data = tensor_buffer;
    tensor.device = Device::CPU;
    tensor.dtype = dtype;
    tensor.numel = numel;
    tensor.owns_data = true;
    tensor.shape = shape;
    tensor.stride = stride;
    tensor.is_contiguous = true;

    return tensor;
}

Tensor create_empty_gpu_tensor(DType dtype, const std::vector<size_t>& shape)
{
    int size = get_dtype_size(dtype);

    int numel = 0, cumulated = 1;
    std::vector<int64_t> stride;
    stride.resize(shape.size(), 1);
    
    for (int i = shape.size() - 1;i >= 0;i--)
    {
        stride[i] = cumulated;
        cumulated *= shape[i];
    }
    numel = cumulated;

    void* tensor_buffer = nullptr;
    
    CUDA_CHECK(cudaMalloc(&tensor_buffer, size * numel));

    Tensor tensor;
    tensor.data = tensor_buffer;
    tensor.device = Device::GPU;
    tensor.dtype = dtype;
    tensor.numel = numel;
    tensor.owns_data = true;
    tensor.shape = shape;
    tensor.stride = stride;
    tensor.is_contiguous = true;

    return tensor;
}

Tensor create_cpu_tensor_from_cpu_data(DType dtype, void* data, const std::vector<size_t>& shape, bool copy_from_cpu)
{
    int size = get_dtype_size(dtype);

    int numel = 0, cumulated = 1;
    std::vector<int64_t> stride;
    stride.resize(shape.size(), 1);
    
    for (int i = shape.size() - 1;i >= 0;i--)
    {
        stride[i] = cumulated;
        cumulated *= shape[i];
    }
    numel = cumulated;

    void* tensor_buffer = nullptr;
    
    if (copy_from_cpu)
    {
        tensor_buffer = operator new(numel * size);
        memset(tensor_buffer, 0, numel * size);
        memcpy(tensor_buffer, data, numel * size);
    }
    else
        tensor_buffer = data;

    Tensor tensor;
    tensor.data = tensor_buffer;
    tensor.device = Device::CPU;
    tensor.dtype = dtype;
    tensor.numel = numel;
    tensor.owns_data = copy_from_cpu;
    tensor.shape = shape;
    tensor.stride = stride;
    tensor.is_contiguous = true;

    return tensor;
}

Tensor create_gpu_tensor_from_cpu_data(DType dtype, void* data, const std::vector<size_t>& shape)
{
    int size = get_dtype_size(dtype);

    int numel = 0, cumulated = 1;
    std::vector<int64_t> stride;
    stride.resize(shape.size(), 1);
    
    for (int i = shape.size() - 1;i >= 0;i--)
    {
        stride[i] = cumulated;
        cumulated *= shape[i];
    }
    numel = cumulated;

    void* tensor_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(&tensor_buffer, size * numel));
    CUDA_CHECK(cudaMemcpy(tensor_buffer, data, size * numel, cudaMemcpyKind::cudaMemcpyHostToDevice));

    Tensor tensor;
    tensor.data = tensor_buffer;
    tensor.device = Device::GPU;
    tensor.dtype = dtype;
    tensor.numel = numel;
    tensor.owns_data = true;
    tensor.shape = shape;
    tensor.stride = stride;
    tensor.is_contiguous = true;

    return tensor;
}

Tensor copy_to_cpu_from_cpu(const Tensor& tensor)
{
    if (tensor.device != Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_cpu_from_cpu failed: Tensor not on CPU.");

    if (!tensor.is_contiguous)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_cpu_from_cpu failed: Tensor views unsupported.");

    int size = get_dtype_size(tensor.dtype);
    void* cpu_buffer = nullptr;

    cpu_buffer = operator new(tensor.numel * size);
    memset(cpu_buffer, 0, tensor.numel * size);
    memcpy(cpu_buffer, tensor.data, tensor.numel * size);

    Tensor new_tensor;
    new_tensor.data = cpu_buffer;
    new_tensor.device = Device::CPU;
    new_tensor.dtype = tensor.dtype;
    new_tensor.numel = tensor.numel;
    new_tensor.owns_data = true;
    new_tensor.shape = tensor.shape;
    new_tensor.stride = tensor.stride;
    new_tensor.is_contiguous = true;

    return new_tensor;
}

Tensor copy_to_gpu_from_gpu(const Tensor& tensor)
{
    if (tensor.device == Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_gpu_from_gpu failed: Tensor not on GPU.");
    
    if (!tensor.is_contiguous)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_gpu_from_gpu failed: Tensor views unsupported.");

    int size = get_dtype_size(tensor.dtype);
    void* gpu_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(&gpu_buffer, size * tensor.numel));
    CUDA_CHECK(cudaMemcpy(gpu_buffer, tensor.data, size * tensor.numel, cudaMemcpyKind::cudaMemcpyDeviceToDevice));

    Tensor new_tensor;
    new_tensor.data = gpu_buffer;
    new_tensor.device = Device::GPU;
    new_tensor.dtype = tensor.dtype;
    new_tensor.numel = tensor.numel;
    new_tensor.owns_data = true;
    new_tensor.shape = tensor.shape;
    new_tensor.stride = tensor.stride;
    new_tensor.is_contiguous = true;

    return new_tensor;
}

Tensor copy_to_gpu_from_cpu(const Tensor& tensor)
{
    if (tensor.device != Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_gpu_from_cpu failed: Tensor already on GPU.");

    if (!tensor.is_contiguous)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_gpu_from_cpu failed: Tensor views unsupported.");

    int size = get_dtype_size(tensor.dtype);
    void* gpu_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(&gpu_buffer, size * tensor.numel));
    CUDA_CHECK(cudaMemcpy(gpu_buffer, tensor.data, size * tensor.numel, cudaMemcpyKind::cudaMemcpyHostToDevice));

    Tensor new_tensor;
    new_tensor.data = gpu_buffer;
    new_tensor.device = Device::GPU;
    new_tensor.dtype = tensor.dtype;
    new_tensor.numel = tensor.numel;
    new_tensor.owns_data = true;
    new_tensor.shape = tensor.shape;
    new_tensor.stride = tensor.stride;
    new_tensor.is_contiguous = true;

    return new_tensor;
}

Tensor copy_to_cpu_from_gpu(const Tensor& tensor)
{
    if (tensor.device == Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_gpu_from_cpu failed: Tensor already on CPU.");

    if (!tensor.is_contiguous)
        throw std::runtime_error("FATAL: cudaoplib_kernel::copy_to_cpu_from_gpu failed: Tensor views unsupported.");

    int size = get_dtype_size(tensor.dtype);
    void* cpu_buffer = nullptr;

    cpu_buffer = operator new(tensor.numel * size);
    CUDA_CHECK(cudaMemcpy(cpu_buffer, tensor.data, size * tensor.numel, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    Tensor new_tensor;
    new_tensor.data = cpu_buffer;
    new_tensor.device = Device::CPU;
    new_tensor.dtype = tensor.dtype;
    new_tensor.numel = tensor.numel;
    new_tensor.owns_data = true;
    new_tensor.shape = tensor.shape;
    new_tensor.stride = tensor.stride;
    new_tensor.is_contiguous = true;

    return new_tensor;
}

void free_cpu_tensor(Tensor& tensor)
{
    if (!tensor.owns_data)
        throw std::runtime_error("FATAL: cudaoplib_kernel::free_cpu_tensor failed: You don't own the tensor!");

    if (tensor.device != Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib_kernel::free_cpu_tensor failed: Tensor is not on CPU.");

    operator delete(tensor.data);
}

void free_gpu_tensor(Tensor& tensor)
{
    if (!tensor.owns_data)
        throw std::runtime_error("FATAL: cudaoplib_kernel::free_gpu_tensor failed: You don't own the tensor!");

    if (tensor.device == Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib_kernel::free_gpu_tensor failed: Tensor is on CPU.");

    CUDA_CHECK(cudaFree(tensor.data));
}

}; // namespace cudaoplib_kernel
