#pragma once
#include <stdio.h>
#include <stdexcept>

#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        fprintf(stderr, "CUDA Error:\n");                      \
        fprintf(stderr, "    File:       %s\n", __FILE__);     \
        fprintf(stderr, "    Line:       %d\n", __LINE__);     \
        fprintf(stderr, "    Error code: %d\n", error_code);   \
        fprintf(stderr, "    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(EXIT_FAILURE);                                      \
    }                                                 \
} while (0)

// cuda_utils.h 或 tensor.h 中
#define CHECK_GPU(tensor)                                               \
    do {                                                                \
        if (!(tensor).is_gpu()) {                                       \
            fprintf(stderr, "Error: Tensor is not on GPU at %s:%d\n",  \
                    __FILE__, __LINE__);                                \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#define DISPATCH_DTYPE(dtype, func, ...)                                \
    switch (dtype) {                                                    \
        case cudaoplib_kernel::DType::Float32: func(float{}, ##__VA_ARGS__); break;    \
        case cudaoplib_kernel::DType::Float16: func(__half{}, ##__VA_ARGS__); break;   \
        case cudaoplib_kernel::DType::Int32:   func(int{}, ##__VA_ARGS__); break;      \
        case cudaoplib_kernel::DType::Int8:    func(int8_t{}, ##__VA_ARGS__); break;   \
        case cudaoplib_kernel::DType::Bool:    func(bool{}, ##__VA_ARGS__); break;      \
        default: throw std::runtime_error("FATAL: DISPATCH_DTYPE failed during kernel launching: Unsupported dtype"); }

//