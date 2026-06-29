#include "cuda_utils.h"
#include "kernel_tensor.h"

#include <cuda/std/array>
#include <type_traits>

#define BLOCK_SIZE 256
#define TILE_SIZE 32

namespace cudaoplib_kernel
{

template<typename T>
__device__ void atomic_add_narrow(T* addr, T val) {
      auto* byte_addr = reinterpret_cast<int8_t*>(addr);
      int8_t byte_val = static_cast<int8_t>(val);
      uint32_t* base = (uint32_t*)((uintptr_t)byte_addr & ~3);
      int shift = ((uintptr_t)byte_addr & 3) * 8;
      uint32_t mask = 0xFF << shift;

      uint32_t old = *base;
      uint32_t assumed;
      do {
          assumed = old;
          int8_t cur = (assumed & mask) >> shift;
          int8_t sum = cur + byte_val;
          uint32_t updated = (assumed & ~mask) | ((uint32_t)(uint8_t)sum << shift);
          old = atomicCAS(base, assumed, updated);
      } while (old != assumed);
  }

template<typename T> __global__ void one_axis_dense_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    size_t n_col
)
{
    size_t x_id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t warp_id = threadIdx.x / 32;
    constexpr size_t n_warp = BLOCK_SIZE / 32;

    size_t y_id = blockIdx.y;

    __shared__ T cache[n_warp];

    T val = (x_id < n_col) ? in[y_id * n_col + x_id] : T{};

    for (size_t offset = 16;offset > 0;offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0)
        cache[warp_id] = val;
    
    __syncthreads();

    
    for (size_t offset = n_warp / 2;offset > 0;offset >>= 1)
    {
        if (threadIdx.x < offset)
            cache[threadIdx.x] += cache[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
            atomic_add_narrow(out + y_id, cache[0]);
        else
            atomicAdd(out + y_id, cache[0]);
}

template<typename T> __global__ void one_axis_striding_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out, // (z, n_row, n_col) -each-> (n_row, n_col) -reduce-> (n_col, )
    size_t n_row,
    size_t n_col
)
{
    size_t x_id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr size_t n_warp = TILE_SIZE / 32;
    size_t warp_id = threadIdx.x / 32;

    size_t y_id = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z_id = blockIdx.z;

    __shared__ T in_cache[TILE_SIZE][TILE_SIZE + 1];
    __shared__ T warp_out_cache[TILE_SIZE][n_warp];

    // transpose: (n_row, n_col) => (n_col, n_row) => dense reduce
    in_cache[threadIdx.x][threadIdx.y] = (x_id < n_col && y_id < n_row) ? in[z_id * n_row * n_col + y_id * n_col + x_id] : T{};
    __syncthreads();

    // original: (y, x) => (n_row, n_col)
    // transposed(in_cache): (thread_y, thread_x) => (n_col, n_row)
    T val = in_cache[threadIdx.y][threadIdx.x];

    for (size_t offset = 16;offset > 0;offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0)
        warp_out_cache[threadIdx.y][warp_id] = val;
    
    if constexpr (n_warp > 1)
    {
        __syncthreads();

        for (size_t offset = n_warp / 2;offset > 0;offset >>= 1)
        {
            if (threadIdx.x < offset)
                warp_out_cache[threadIdx.y][threadIdx.x] += warp_out_cache[threadIdx.y][threadIdx.x + offset];
            __syncthreads();
        }
    }

    if (threadIdx.x == 0 && blockIdx.x * blockDim.x + threadIdx.y < n_col)
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
            atomic_add_narrow(out + z_id * n_col + blockIdx.x * blockDim.x + threadIdx.y, warp_out_cache[threadIdx.y][0]);
        else
            atomicAdd(out + z_id * n_col + blockIdx.x * blockDim.x + threadIdx.y, warp_out_cache[threadIdx.y][0]);
}

template<typename T> __global__ void one_axis_striding_sum_without_atomic_kernel(
    const T* __restrict__ in,
    T* __restrict__ scratch, // (z, (n_row + TILE_SIZE - 1) / TILE_SIZE, n_col)
    size_t n_row,
    size_t n_col
)
{
    size_t x_id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr size_t n_warp = TILE_SIZE / 32;
    size_t warp_id = threadIdx.x / 32;

    size_t y_id = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z_id = blockIdx.z;

    __shared__ T in_cache[TILE_SIZE][TILE_SIZE + 1];
    __shared__ T warp_out_cache[TILE_SIZE][n_warp];

    // transpose: (n_row, n_col) => (n_col, n_row) => dense reduce
    in_cache[threadIdx.x][threadIdx.y] = (x_id < n_col && y_id < n_row) ? in[z_id * n_row * n_col + y_id * n_col + x_id] : T{};
    __syncthreads();

    // original: (y, x) => (n_row, n_col)
    // transposed(in_cache): (thread_y, thread_x) => (n_col, n_row)
    T val = in_cache[threadIdx.y][threadIdx.x];

    for (size_t offset = 16;offset > 0;offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0)
        warp_out_cache[threadIdx.y][warp_id] = val;
    

    if constexpr (n_warp > 1)
    {
        __syncthreads();

        for (size_t offset = n_warp / 2;offset > 0;offset >>= 1)
        {
            if (threadIdx.x < offset)
                warp_out_cache[threadIdx.y][threadIdx.x] += warp_out_cache[threadIdx.y][threadIdx.x + offset];
            __syncthreads();
        }

    }

    if (threadIdx.x == 0 && blockIdx.x * blockDim.x + threadIdx.y < n_col)
        scratch[z_id * ((n_row + TILE_SIZE - 1) / TILE_SIZE) * n_col + blockIdx.y * n_col + blockIdx.x * blockDim.x + threadIdx.y] = warp_out_cache[threadIdx.y][0];
}


void sum(const Tensor& in, Tensor& out, int dim)
{
    CHECK_GPU(in); CHECK_GPU(out);
    if (dim == static_cast<int>(in.shape.size()) - 1)
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
        int i = 0;
        for (;i < dim;i++)
            left_cumulated *= in.shape[i];
        
        i++;
        
        for (;i < static_cast<int>(in.shape.size());i++)
            right_cumulated *= in.shape[i];


        auto launcher = [&](auto dummy){
            using T = decltype(dummy);
            dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
            dim3 grid_size((right_cumulated + TILE_SIZE - 1) / TILE_SIZE, (in.shape[dim] + TILE_SIZE - 1) / TILE_SIZE, left_cumulated);

            if (in.shape[dim] / TILE_SIZE >= 8)
            {
                size_t required_buf = left_cumulated * (in.shape[dim] + TILE_SIZE - 1) / TILE_SIZE * right_cumulated * sizeof(T);

                get_scratch_buffer_mutex().lock();
                T* scratch = static_cast<T*>(acquire_scratch_buffer(required_buf));
        
                one_axis_striding_sum_without_atomic_kernel<<<grid_size, block_size>>>(
                    static_cast<T*>(in.data),
                    static_cast<T*>(scratch),
                    in.shape[dim],
                    right_cumulated
                );

                size_t new_in_dim = (in.shape[dim] + TILE_SIZE - 1) / TILE_SIZE;

                dim3 new_block_size(TILE_SIZE, TILE_SIZE, 1);
                dim3 new_grid_size((right_cumulated + TILE_SIZE - 1) / TILE_SIZE, (new_in_dim + TILE_SIZE - 1) / TILE_SIZE, left_cumulated);
                one_axis_striding_sum_kernel<<<new_grid_size, new_block_size>>>(
                    static_cast<T*>(scratch),
                    static_cast<T*>(out.data),
                    new_in_dim,
                    right_cumulated
                );

                cudaDeviceSynchronize();
                get_scratch_buffer_mutex().unlock();
                
            }
            else
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