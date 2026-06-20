#include "cuda_utils.h"
#include "elementwise.h"
#include "tensor.h"

#include <cuda/std/array>

namespace cudaoplib
{

template<typename T> __global__ void add_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    size_t N,
    TensorStride a_strides,
    TensorStride b_strides,
    TensorShape output_shape
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int64_t offset_a = 0, offset_b = 0;
    int64_t remaining = tid;
    for (int64_t i = MAX_TENSOR_DIM - 1;i >= 0;i--)
    {
        int coord = remaining % output_shape[i];
        remaining /= output_shape[i];

        offset_a += a_strides[i] * coord;
        offset_b += b_strides[i] * coord;
    }
    out[tid] = a[offset_a] + b[offset_b];
}

void add(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        add_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<T*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}

}; // namespace cudaoplib
