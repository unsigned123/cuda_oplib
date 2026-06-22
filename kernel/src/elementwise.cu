#include "cuda_utils.h"
#include "elementwise.h"
#include "kernel_tensor.h"

#include <cuda/std/array>
#include <type_traits>

namespace cudaoplib_kernel
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

template<typename T> __global__ void sub_kernel(
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
    out[tid] = a[offset_a] - b[offset_b];
}

template<typename T> __global__ void mul_kernel(
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
    out[tid] = a[offset_a] * b[offset_b];
}

template<typename T> __global__ void div_kernel(
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
    out[tid] = a[offset_a] / b[offset_b];
}

template<typename T> __global__ void mod_kernel(
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
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, __half>)
        out[tid] = fmodf(a[offset_a], b[offset_b]);
    else
        out[tid] = ((a[offset_a] % b[offset_b]) + b[offset_b]) % b[offset_b];
}

template<typename T> __global__ void remainder_kernel(
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
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, __half>)
        out[tid] = remainderf(a[offset_a], b[offset_b]);
    else
        out[tid] = a[offset_a] % b[offset_b];
}

template<typename T> __global__ void eq_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] == b[offset_b];
}

template<typename T> __global__ void neq_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] != b[offset_b];
}

template<typename T> __global__ void lt_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] < b[offset_b];
}

template<typename T> __global__ void le_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] <= b[offset_b];
}

template<typename T> __global__ void gt_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] > b[offset_b];
}

template<typename T> __global__ void ge_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] >= b[offset_b];
}

template<typename T> __global__ void logical_and_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] && b[offset_b];
}

template<typename T> __global__ void logical_or_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    int8_t* __restrict__ out,
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
    out[tid] = a[offset_a] || b[offset_b];
}

template<typename T> __global__ void pow_kernel(
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
    out[tid] = powf(a[offset_a], b[offset_b]);
}

template<typename T> __global__ void maximum_kernel(
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
    out[tid] = (a[offset_a] > b[offset_b]) ? a[offset_a] : b[offset_b];
}

template<typename T> __global__ void minimum_kernel(
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
    out[tid] = (a[offset_a] < b[offset_b]) ? a[offset_a] : b[offset_b];
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

void sub(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        sub_kernel<<<grid_size, block_size>>>(
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

void mul(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        mul_kernel<<<grid_size, block_size>>>(
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

void div(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        div_kernel<<<grid_size, block_size>>>(
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

void mod(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        mod_kernel<<<grid_size, block_size>>>(
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

void remainder(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        remainder_kernel<<<grid_size, block_size>>>(
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

void eq(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        eq_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}
void neq(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        neq_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}

void lt(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        lt_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}

void le(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        le_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}

void gt(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        gt_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}
void ge(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        ge_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}
void logical_and(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        logical_and_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}
void logical_or(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        logical_or_kernel<<<grid_size, block_size>>>(
            static_cast<T*>(a.data),
            static_cast<T*>(b.data),
            static_cast<int8_t*>(out.data),
            out.numel,
            a.get_stride(),
            b.get_stride(),
            out.pad_shape()
        );
    };

    DISPATCH_DTYPE(out.dtype, launcher)
}
void pow(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        pow_kernel<<<grid_size, block_size>>>(
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
void maximum(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        maximum_kernel<<<grid_size, block_size>>>(
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
void minimum(const Tensor& a, const Tensor& b, Tensor& out)
{
    CHECK_GPU(a); CHECK_GPU(b); CHECK_GPU(out);

    auto launcher = [&](auto dummy){
        using T = decltype(dummy);
        int block_size = 256;
        int grid_size = (out.numel + block_size - 1) / block_size;

        minimum_kernel<<<grid_size, block_size>>>(
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


}; // namespace cudaoplib_kernel
