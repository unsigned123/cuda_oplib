#include "cuda_utils.h"
#include "kernel_tensor.h"

#include <cuda/std/array>
#include <type_traits>

#define BLOCK_SIZE 1024
#define TILE_SIZE 32

namespace cudaoplib_kernel
{

template<typename T> __global__ void one_axis_dense_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    size_t n_col
)
{
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / 32;
    constexpr int n_warp = BLOCK_SIZE / 32;

    int y_id = blockIdx.y;

    __shared__ T cache[n_warp];

    T val = (x_id < n_col) ? in[y_id * n_col + x_id] : T{};

    for (int offset = 16;offset > 0;offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0)
        cache[warp_id] = val;
    
    __syncthreads();

    
    for (int offset = n_warp / 2;offset > 0;offset >>= 1)
    {
        if (threadIdx.x < offset)
            cache[threadIdx.x] += cache[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out + y_id, cache[0]);
}

template<typename T> __global__ void one_axis_striding_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out, // (z, n_row, n_col) -each-> (n_row, n_col) -reduce-> (n_col, )
    size_t n_row,
    size_t n_col
)
{
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int n_warp = TILE_SIZE / 32;
    int warp_id = threadIdx.x / 32;

    int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    int z_id = blockIdx.z;

    __shared__ T in_cache[TILE_SIZE][TILE_SIZE + 1];
    __shared__ T warp_out_cache[TILE_SIZE][n_warp];
    __shared__ T tile_out_cache[TILE_SIZE];

    // transpose: (n_row, n_col) => (n_col, n_row) => dense reduce
    in_cache[threadIdx.x][threadIdx.y] = (x_id < n_col && y_id < n_row) ? in[z_id * n_row * n_col + y_id * n_col + x_id] : T{};
    __syncthreads();

    T val = in_cache[threadIdx.y][threadIdx.x];

    for (int offset = 16;offset > 0;offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0)
        warp_out_cache[threadIdx.y][warp_id] = val;
    
    __syncthreads();

    for (int offset = n_warp / 2;offset > 0;offset >>= 1)
    {
        if (threadIdx.x < offset)
            warp_out_cache[threadIdx.y][threadIdx.x] += warp_out_cache[threadIdx.y][threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        tile_out_cache[threadIdx.y] = warp_out_cache[threadIdx.y][0];

    __syncthreads();

    if (threadIdx.x == 0 && y_id < n_col)
        atomicAdd(out + z_id * n_col + y_id, tile_out_cache[threadIdx.y]);
}





void sum(const Tensor& in, Tensor& out, int dim)
{
    CHECK_GPU(in); CHECK_GPU(out);
    if (dim == in.shape.size() - 1)
    {
        unsigned int cumulated = 1;
        for (size_t i = 0;i < in.shape.size() - 1;i++)
            cumulated *= in.shape[i];

        auto launcher = [&](auto dummy){
            using T = decltype(dummy);
            int block_size = BLOCK_SIZE;
            dim3 grid_size((in.shape[in.shape.size() - 1] + BLOCK_SIZE - 1) / BLOCK_SIZE, cumulated);

            one_axis_dense_sum_kernel<<<grid_size, block_size>>>(
                static_cast<T*>(in.data),
                static_cast<T*>(out.data),
                in.shape[in.shape.size() - 1]
            );
        };

        DISPATCH_DTYPE(out.dtype, launcher)
    }
    else
    {
        unsigned int left_cumulated = 1, right_cumulated = 1;
        size_t i = 0;
        for (;i < dim;i++)
            left_cumulated *= in.shape[i];
        
        for (;i < in.shape.size();i++)
            right_cumulated *= in.shape[i];

        auto launcher = [&](auto dummy){
            using T = decltype(dummy);
            dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
            dim3 grid_size((right_cumulated + TILE_SIZE - 1) / TILE_SIZE, (in.shape[dim] + TILE_SIZE - 1) / TILE_SIZE, left_cumulated);

            one_axis_striding_sum_kernel<<<grid_size, block_size>>>(
                static_cast<T*>(in.data),
                static_cast<T*>(out.data),
                in.shape[dim],
                right_cumulated
            );
        };

        DISPATCH_DTYPE(out.dtype, launcher);
    }
}

} // namespace cudaoplib_kernel