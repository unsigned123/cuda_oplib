#include "cuda_utils.h"
#include "kernel_tensor.h"

#include <cuda/std/array>
#include <type_traits>

#define BLOCK_SIZE 256
#define PARALLEL 16

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



template<bool UseScratch, typename T> __global__ void one_axis_dense_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    T* __restrict__ scratch,
    size_t n_col
)
{
    size_t y_id = blockIdx.x;
    const T* row = in + y_id * n_col;

    // while-loop: one block eats the entire row → gridDim.x=rows, gridDim.y=col_chunks
    size_t x_base = blockIdx.y * BLOCK_SIZE * PARALLEL;
    T local_sum = T{};

    while (x_base < n_col) {
        size_t x_id = x_base + threadIdx.x * PARALLEL;
        T chunk = T{};

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>)
        {
            #pragma unroll
            for (size_t i = 0;i < PARALLEL / 4;i++)
            {
                size_t offset = x_id + i * 4;
                if (offset + 3 < n_col) {
                    float4 v = reinterpret_cast<const float4*>(row + offset)[0];
                    chunk += v.x; chunk += v.y; chunk += v.z; chunk += v.w;
                } else {
                    if (offset + 0 < n_col) chunk += row[offset + 0];
                    if (offset + 1 < n_col) chunk += row[offset + 1];
                    if (offset + 2 < n_col) chunk += row[offset + 2];
                    if (offset + 3 < n_col) chunk += row[offset + 3];
                }
            }
        }
        else
        {
            #pragma unroll
            for (size_t i = 0;i < PARALLEL;i++)
            {
                if (x_id + i < n_col)
                    chunk += row[x_id + i];
            }
        }

        local_sum += chunk;
        x_base += gridDim.x * BLOCK_SIZE * PARALLEL;
    }

    // block-level reduce
    __shared__ T smem;
    if (threadIdx.x == 0) smem = T{};
    __syncthreads();

    #pragma unroll
    for (size_t offset = 16;offset > 0;offset >>= 1)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (threadIdx.x % 32 == 0)
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
            atomic_add_narrow(&smem, local_sum);
        else
            atomicAdd(&smem, local_sum);

    __syncthreads();

    if (threadIdx.x == 0)
        if constexpr (UseScratch)
            scratch[y_id * gridDim.y + blockIdx.y] = smem;
        else
            out[y_id] = smem;
}

/* old (fixed grid per block):
template<typename T> __global__ void one_axis_dense_sum_kernel_old(
    const T* __restrict__ in, T* __restrict__ out, size_t n_col)
{
    size_t x_id = blockIdx.x * BLOCK_SIZE * PARALLEL + threadIdx.x * PARALLEL;
    size_t y_id = blockIdx.y;
    __shared__ T smem;
    T val = T{};
    const T* row = in + y_id * n_col;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>)
    {
        #pragma unroll
        for (size_t i = 0;i < PARALLEL / 4;i++)
        {
            size_t offset = x_id + i * 4;
            if (offset + 3 < n_col) {
                float4 v = reinterpret_cast<const float4*>(row + offset)[0];
                val += v.x; val += v.y; val += v.z; val += v.w;
            } else {
                if (offset + 0 < n_col) val += row[offset + 0];
                if (offset + 1 < n_col) val += row[offset + 1];
                if (offset + 2 < n_col) val += row[offset + 2];
                if (offset + 3 < n_col) val += row[offset + 3];
            }
        }
    }
    else
    {
        #pragma unroll
        for (size_t i = 0;i < PARALLEL;i++)
            if (x_id + i < n_col) val += row[x_id + i];
    }
    if (threadIdx.x == 0) smem = T{};
    __syncthreads();
    for (size_t offset = 16;offset > 0;offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x % 32 == 0)
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
            atomic_add_narrow(&smem, val); else atomicAdd(&smem, val);
    __syncthreads();
    if (threadIdx.x == 0)
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
            atomic_add_narrow(out + y_id, smem); else atomicAdd(out + y_id, smem);
}
*/

template<bool UseScratch, typename T> __global__ void one_axis_striding_sum_kernel(
    const T* __restrict__ in,
    T* __restrict__ out, // (z, n_row, n_col) -each-> (n_row, n_col) -reduce-> (n_col, )
    T* __restrict__ scratch,
    size_t n_row,
    size_t n_col
)
{
    size_t z_id = blockIdx.z;
    T col[TILE_SIZE];
    __shared__ T cache[TILE_SIZE];

    if (threadIdx.x < TILE_SIZE)
        cache[threadIdx.x] = T{};
    __syncthreads();

    #pragma unroll
    for (size_t i = 0;i < TILE_SIZE;i++)
    {
        size_t y_id = blockIdx.y * TILE_SIZE * 8 + (threadIdx.x / TILE_SIZE) * TILE_SIZE + i;
        size_t x_id = blockIdx.x * TILE_SIZE + threadIdx.x % TILE_SIZE;
        col[i] = (x_id < n_col && y_id < n_row) ? in[z_id * n_row * n_col + y_id * n_col + x_id] : T{};
    }

    T result = T{};
    #pragma unroll
    for (size_t i = 0;i < TILE_SIZE;i++)
        result += col[i];

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
        atomic_add_narrow(cache + threadIdx.x % 32, result);
    else
        atomicAdd(cache + threadIdx.x % 32, result);

    __syncthreads();

    // warp 0 writes result; atomicAdd across grid_y blocks
    if (threadIdx.x / 32 == 0 && blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE) < n_col)
    {
        if constexpr (UseScratch)
        {
            scratch[z_id * gridDim.y * n_col + blockIdx.y * n_col + blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE)] = cache[threadIdx.x % 32];
        }
        else
        {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
                atomic_add_narrow(out + z_id * n_col + blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE), cache[threadIdx.x % 32]);
            else
                atomicAdd(out + z_id * n_col + blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE), cache[threadIdx.x % 32]);
        }
    }

}

template<size_t TILE, bool UseScratch, typename T> __global__ void one_axis_striding_sum_kernel_for_few_col(
    const T* __restrict__ in,
    T* __restrict__ out, // (z, n_row, n_col) -each-> (n_row, n_col) -reduce-> (n_col, )
    T* __restrict__ scratch,
    size_t n_row,
    size_t n_col
)
{
    size_t z_id = blockIdx.z;
    T col[TILE_SIZE];
    __shared__ T cache[TILE];

    if (threadIdx.x < TILE)
        cache[threadIdx.x] = T{};
    __syncthreads();

    size_t x_id = blockIdx.x * TILE + threadIdx.x % TILE;
    #pragma unroll
    for (size_t i = 0;i < TILE_SIZE;i++)
    {
        size_t y_id = blockIdx.y * (1024 / TILE) * 8 + (threadIdx.x / TILE_SIZE) * (1024 / TILE) + i * (TILE_SIZE / TILE);
        col[i] = (x_id < n_col && y_id < n_row) ? in[z_id * n_row * n_col + y_id * n_col + x_id] : T{};
    }

    T result = T{};
    #pragma unroll
    for (size_t i = 0;i < TILE_SIZE;i++)
        result += col[i];

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
        atomic_add_narrow(cache + threadIdx.x % TILE, result);
    else
        atomicAdd(cache + threadIdx.x % TILE, result);

    __syncthreads();

    if (threadIdx.x / TILE == 0 && x_id < n_col)
    {
        if constexpr (UseScratch)
        {
            scratch[z_id * gridDim.y * n_col + blockIdx.y * n_col + x_id] = cache[threadIdx.x % TILE];
        }
        else
        {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
                atomic_add_narrow(out + z_id * n_col + x_id, cache[threadIdx.x % TILE]);
            else
                atomicAdd(out + z_id * n_col + x_id, cache[threadIdx.x % TILE]);
        }
    }
}

template<int TILE, typename T> __global__ void one_axis_striding_sum_kernel_for_few_row(
    const T* __restrict__ in,
    T* __restrict__ out,
    size_t n_row,
    size_t n_col
)
{
    size_t z_id = blockIdx.z;
    size_t x_base = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    T results[4] = {T{}, T{}, T{}, T{}};

    #pragma unroll
    for (size_t i = 0;i < TILE;i++)
    {
        size_t y_id = blockIdx.y * TILE + i;

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>)
        {
            if (x_base + 3 < n_col && y_id < n_row) {
                float4 v = reinterpret_cast<const float4*>(in + z_id * n_row * n_col + y_id * n_col + x_base)[0];
                results[0] += v.x; results[1] += v.y; results[2] += v.z; results[3] += v.w;
            } else {
                if (x_base + 0 < n_col && y_id < n_row) results[0] += in[z_id * n_row * n_col + y_id * n_col + x_base + 0];
                if (x_base + 1 < n_col && y_id < n_row) results[1] += in[z_id * n_row * n_col + y_id * n_col + x_base + 1];
                if (x_base + 2 < n_col && y_id < n_row) results[2] += in[z_id * n_row * n_col + y_id * n_col + x_base + 2];
                if (x_base + 3 < n_col && y_id < n_row) results[3] += in[z_id * n_row * n_col + y_id * n_col + x_base + 3];
            }
        }
        else
        {
            if (x_base + 0 < n_col && y_id < n_row) results[0] += in[z_id * n_row * n_col + y_id * n_col + x_base + 0];
            if (x_base + 1 < n_col && y_id < n_row) results[1] += in[z_id * n_row * n_col + y_id * n_col + x_base + 1];
            if (x_base + 2 < n_col && y_id < n_row) results[2] += in[z_id * n_row * n_col + y_id * n_col + x_base + 2];
            if (x_base + 3 < n_col && y_id < n_row) results[3] += in[z_id * n_row * n_col + y_id * n_col + x_base + 3];
        }
    }

    #pragma unroll
    for (size_t j = 0;j < 4;j++)
    {
        if (x_base + j < n_col)
        {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool>)
                atomic_add_narrow(out + z_id * n_col + x_base + j, results[j]);
            else
                atomicAdd(out + z_id * n_col + x_base + j, results[j]);
        }
    }

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

            constexpr size_t max_iterations = 16;
            size_t last_dim = in.shape[in.shape.size() - 1];
            size_t per_iter = BLOCK_SIZE * PARALLEL;

            if (last_dim >= max_iterations * per_iter)
            {
                size_t grid_x = (last_dim + (max_iterations * per_iter) - 1) / (max_iterations * per_iter);
                dim3 grid_size(cumulated, grid_x, 1);  // x=row, y=col_chunk

                get_scratch_buffer_mutex().lock();
                one_axis_dense_sum_kernel<true, T> <<<grid_size, block_size>>>(
                    static_cast<T*>(in.data),
                    static_cast<T*>(nullptr),
                    static_cast<T*>(acquire_scratch_buffer(cumulated * grid_x * sizeof(T))),
                    last_dim
                );

                dim3 second_grid(cumulated, 1, 1);
                one_axis_dense_sum_kernel<false> <<<second_grid, block_size>>>(
                    static_cast<T*>(get_scratch_buffer()),
                    static_cast<T*>(out.data),
                    static_cast<T*>(nullptr), grid_x
                );
                get_scratch_buffer_mutex().unlock();
            }
            else
            {
                dim3 grid_size(cumulated, 1, 1);

                one_axis_dense_sum_kernel<false, T> <<<grid_size, block_size>>>(
                    static_cast<T*>(in.data),
                    static_cast<T*>(out.data),
                    static_cast<T*>(nullptr),
                    in.shape[in.shape.size() - 1]
                );
            }
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
            constexpr size_t max_atomic_conflicts = 16;
            constexpr size_t min_reduce_dim_length = 256;
            constexpr size_t min_right_cumulated = 32;
            size_t reduce_dim = in.shape[dim];
            size_t per_block = TILE_SIZE * 8;

            if (reduce_dim < min_reduce_dim_length)
            {
                // Select nearest power-of-2 TILE ≤ reduce_dim (minimum 2)
                auto launch_few = [&](int tile) {
                    #define LAUNCH_TILE(N) \
                        dim3 block_size(N * 8, 1, 1); \
                        dim3 g((right_cumulated + N * 8 * 4 - 1) / (N * 8 * 4), \
                           (reduce_dim + N - 1) / N, left_cumulated); \
                        one_axis_striding_sum_kernel_for_few_row<N, T><<<g, block_size>>>( \
                            static_cast<T*>(in.data), static_cast<T*>(out.data), \
                            reduce_dim, right_cumulated);
                    if (tile <= 2)      { LAUNCH_TILE(2) }
                    else if (tile <= 4) { LAUNCH_TILE(4) }
                    else if (tile <= 8) { LAUNCH_TILE(8) }
                    else if (tile <= 16){ LAUNCH_TILE(16) }
                    else                { LAUNCH_TILE(32) }
                    #undef LAUNCH_TILE
                };
                launch_few(static_cast<int>(reduce_dim));
            }
            else if (right_cumulated < min_right_cumulated)
            {
                auto launch_few = [&](int tile) {
                    #define LAUNCH_TILE(N) \
                        dim3 block_size(TILE_SIZE * 8, 1, 1); \
                        dim3 g((right_cumulated + N - 1) / N, \
                           (reduce_dim + (TILE_SIZE * (256 / N) * 4) - 1) / (TILE_SIZE * (256 / N) * 4), left_cumulated); \
                        one_axis_striding_sum_kernel_for_few_col<N, false, T><<<g, block_size>>>( \
                            static_cast<T*>(in.data), static_cast<T*>(out.data), static_cast<T*>(nullptr),\
                            reduce_dim, right_cumulated);
                    if (tile <= 2)      { LAUNCH_TILE(2) }
                    else if (tile <= 4) { LAUNCH_TILE(4) }
                    else if (tile <= 8) { LAUNCH_TILE(8) }
                    else if (tile <= 16){ LAUNCH_TILE(16) }
                    else                { LAUNCH_TILE(32) }
                    #undef LAUNCH_TILE
                };
                launch_few(static_cast<int>(right_cumulated));
            }
            else if (reduce_dim >= max_atomic_conflicts * per_block)
            {
                size_t grid_y = (reduce_dim + TILE_SIZE * 8 - 1) / (TILE_SIZE * 8);
                dim3 block_size(TILE_SIZE * 8, 1, 1);
                dim3 pass1_grid((right_cumulated + TILE_SIZE - 1) / TILE_SIZE, grid_y, left_cumulated);

                size_t scratch_elems = left_cumulated * grid_y * right_cumulated;
                get_scratch_buffer_mutex().lock();
                void* scratch = acquire_scratch_buffer(scratch_elems * sizeof(T));
                one_axis_striding_sum_kernel<true, T> <<<pass1_grid, block_size>>>(
                    static_cast<T*>(in.data),
                    static_cast<T*>(nullptr),
                    static_cast<T*>(scratch),
                    reduce_dim, right_cumulated
                );

                size_t pass2_grid_y = (grid_y + TILE_SIZE * 8 - 1) / (TILE_SIZE * 8);
                dim3 pass2_grid((right_cumulated + TILE_SIZE - 1) / TILE_SIZE, pass2_grid_y, left_cumulated);

                one_axis_striding_sum_kernel<false, T> <<<pass2_grid, block_size>>>(
                    static_cast<T*>(get_scratch_buffer()),
                    static_cast<T*>(out.data),
                    static_cast<T*>(nullptr),
                    grid_y, right_cumulated
                );
                get_scratch_buffer_mutex().unlock();

            }
            else
            {
                dim3 block_size(TILE_SIZE * 8, 1, 1);
                dim3 grid_size((right_cumulated + TILE_SIZE - 1) / TILE_SIZE, (in.shape[dim] + TILE_SIZE * 8 - 1) / (TILE_SIZE * 8), left_cumulated);

                one_axis_striding_sum_kernel<false, T> <<<grid_size, block_size>>>(
                    static_cast<T*>(in.data),
                    static_cast<T*>(out.data),
                    static_cast<T*>(nullptr),
                    in.shape[dim],
                    right_cumulated);
            }
        };

        DISPATCH_DTYPE(out.dtype, launcher);
    }
}

} // namespace cudaoplib_kernel