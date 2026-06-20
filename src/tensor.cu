#include "cuda_utils.h"
#include "tensor.h"
#include <memory>
#include <stdexcept>

namespace cudaoplib
{

Tensor create_cpu_tensor(DType dtype, void* data, std::vector<size_t> shape, bool copy_from_cpu)
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

    return tensor;
}

Tensor copy_to_gpu(const Tensor& tensor)
{
    if (tensor.device == Device::CPU)
        throw std::runtime_error("FATAL: cudaoplib::copy_to_gpu failed: Tensor already on GPU.");

    int size = get_dtype_size(tensor.dtype);
    void* gpu_buffer = nullptr;

    CUDA_CHECK(cudaMalloc((void**)gpu_buffer, size * tensor.numel));
    CUDA_CHECK(cudaMemcpy(gpu_buffer, tensor.data, size * tensor.numel, cudaMemcpyKind::cudaMemcpyHostToDevice));

    Tensor new_tensor;
    new_tensor.data = gpu_buffer;
    new_tensor.device = Device::GPU;
    new_tensor.dtype = tensor.dtype;
    new_tensor.numel = tensor.numel;
    new_tensor.owns_data = true;
    new_tensor.shape = tensor.shape;
    new_tensor.stride = tensor.stride;

    return new_tensor;
}

Tensor copy_to_cpu(const Tensor& tensor)
{
    if (tensor.device == Device::GPU)
        throw std::runtime_error("FATAL: cudaoplib::copy_to_gpu failed: Tensor already on CPU.");

    int size = get_dtype_size(tensor.dtype);
    void* cpu_buffer;

    cpu_buffer = operator new(tensor.numel * size);
    CUDA_CHECK(cudaMemcpy(tensor.data, cpu_buffer, size * tensor.numel, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    Tensor new_tensor;
    new_tensor.data = cpu_buffer;
    new_tensor.device = Device::CPU;
    new_tensor.dtype = tensor.dtype;
    new_tensor.numel = tensor.numel;
    new_tensor.owns_data = true;
    new_tensor.shape = tensor.shape;
    new_tensor.stride = tensor.stride;

    return new_tensor;
}

}; // namespace cudaoplib
