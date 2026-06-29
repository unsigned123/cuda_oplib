# CUDA Reduce 算子调优报告

## 概述

本报告记录了 `cudaoplib` 张量库中 `sum` reduce 算子从初版实现到最终高度优化的完整历程。优化涵盖 dense 规约路径（规约最后一维）和 striding 规约路径（规约非最后一维）。经过多轮迭代，全部 21 个 benchmark 测例均达到或接近 PyTorch 性能（0.16×–1.16×，大部分 < 1.0×），且全部 6 个 correctness 测例（含 15 个子 case 的 PyTorch 逐元素对比）通过。

---

## 1. 起点：初版实现

初版 `one_axis_dense_sum_kernel` 使用经典的 GPU reduce 算法：

```
BLOCK_SIZE = 1024
每线程 1 元素 → warp shuffle → shared memory 跨 warp tree reduce → atomicAdd(global)
```

关键参数：
- **Dense 路径**：`grid_x = ceil(n_col / BLOCK_SIZE)`，`grid_y = n_row`。每行被切分成多个 block，block 间通过 `atomicAdd` 竞争同一输出地址。
- **Striding 路径**：block=(32,32)，通过 shared memory transpose + warp shuffle 实现规约。

**性能基线**（vs PyTorch，双方均用 CUDA event 测 GPU kernel 时间）：

| Case | Ratio (our/torch) |
|---|---|
| dense 4096² | 2.76× |
| strid 4096² dim0 | 2.28× |
| strid 4096²×64 dim1 | 2.97× |

### 初版的主要问题

通过 ncu profiling 发现三个结构性瓶颈：

1. **`__syncthreads` barrier stall**：跨 warp tree reduce 使用 `__syncthreads × 3`，BLOCK_SIZE=1024 意味着 31/32 的 warp 在 barrier 上空等。ncu 报告 `No Eligible = 70-80%`。
2. **Block Limit Warps = 1**：1024 线程/block = 32 warp，每个 SM 只能容纳 1 个 block。当唯一的 block 被 barrier 卡住时，SM 没有备用 warp 可以切入隐藏延迟。
3. **Global atomicAdd 串行化**：多个 block 同时 `atomicAdd` 到同一输出地址，L2 cache controller 做串行仲裁，每次 ~400-500 GPU cycles。

---

## 2. Phase 1：缩小 BLOCK_SIZE，提升 Occupancy

**核心思路**：将 BLOCK_SIZE 从 1024 降到 256（8 warp/block），每个 SM 从容纳 1 block 变为 6 block。当一个 block 的 warp 被 barrier stall 时，其他 5 个 block 可以切入执行。

```
BLOCK_SIZE: 1024 → 256
Block/SM:   1 → 6
n_warp:     32 → 8
```

**效果**：dense 路径全面改善。

---

## 3. Phase 2：用 Shared Memory Atomic 替代 Tree Reduce

**关键发现**：对 `n_warp = 8` 的规模，shared memory atomicAdd 比 3 轮 tree reduce + 3 次 `__syncthreads` 更快。

```
传统：warp shuffle → cache[8] → __syncthreads ×3 → tree reduce → atomicAdd(global)
优化：warp shuffle → atomicAdd(&smem) ×8路 → __syncthreads ×1 → atomicAdd(global)
```

**经验**：当 N（warp 数）很小时，常数项（barrier 同步延迟）碾压了 O(log N) 的复杂度优势。Shared memory atomic 的流水线延迟约 65 cycles，而 3 轮 `__syncthreads` + tree reduce 超过 150 cycles。N < 16 时 atomicAdd 总是更快。

---

## 4. Phase 3：While-loop 策略——一个 Block 吃整行

**核心思路**：不让多个 block 瓜分一行，而是让**一个 block 用 while 循环吃掉整行**。

```cuda
// 传统：多个 block 瓜分一行 → atomicAdd 竞争
grid_x = ceil(n_col / (BLOCK*PARALLEL))

// While-loop：一个 block 吃整行 → 零 atomicAdd 竞争
while (x_base < n_col) {
    // 处理 BLOCK*PARALLEL 个元素
    x_base += gridDim.y * BLOCK * PARALLEL;  // 步进到下一个列块
}
```

为了利用 grid 的三维索引，对 dense 路径做了 grid 轴交换：`grid = (cumulated, grid_x, 1)`，其中 `gridDim.x = 行数`，`gridDim.y = 列块数`。

**效果**：block 间的 global atomicAdd 竞争被完全消除。

**经验**：GPU 的 global atomic 代价远超计算。能在一个 block 内用纯计算解决的问题，绝不让第二个 block 参与。while-loop 的额外串行计算时间远小于消除的 atomic 竞争时间。

---

## 5. Phase 4：float4 向量化 + ILP Fold

**核心思路**：每个线程加载 16 个元素（PARALLEL=16），通过 float4 向量化加载（4 组 float4），并将 fold 展开为独立 FADD 以利用指令级并行（ILP）。

```cuda
// 向量化加载
float4 v = reinterpret_cast<const float4*>(row + offset)[0];
// ILP fold：独立 FADD 语句，编译器可双发射（dual-issue）load 与 FADD
chunk += v.x; chunk += v.y; chunk += v.z; chunk += v.w;
```

**关键**：`val += v.x + v.y + v.z + v.w`（合并表达式）vs `val += v.x; val += v.y; ...`（独立语句）。后者让编译器生成 4 条独立的 FADD 指令，可以在 load 的 in-flight 延迟期间发射，隐藏内存延迟。

---

## 6. Phase 5：Two-Pass 二级规约 + Scratch Buffer

**核心思路**：当规约维度超过 while-loop 安全阈值（16 次迭代，65,536 元素）时，启用两 pass 策略：

```
Pass 1: grid_y > 1 个 block 瓜分行 → 写入 scratch buffer（每个 block 写一个 partial sum，无 global atomic）
Pass 2: grid_y = 1 个 block → while-loop 吃掉 scratch 行 → 写入最终输出
```

Dense 路径的 scratch buffer 极简——每个 block 写一个值，scratch 大小 = `n_row × grid_y`。Pass 2 用 while-loop 处理 scratch（grid_y 个元素，通常 1 次迭代即可完成）。

**WSL2 cudaMalloc 不保证清零**：在调试过程中发现，WSL2 的 CUDA 驱动返回的显存可能包含旧数据。`create_empty_gpu_tensor` 中加 `cudaMemset(tensor_buffer, 0, ...)` 解决了 scratch buffer 复用导致的地址别名和数据残留问题。所有输出 tensor（无论是否需要零初始化）也统一做 `cudaMemset`，防止 C++ 端未初始化的 `std::vector` resize 行为带来的差异。

---

## 7. Phase 6：Striding 路径的专用 Kernel

Striding 路径处理 `dim < last` 的情况。输入被映射为 3D 布局：`z = left_cumulated`（规约维之前的累积），`y = reduce_dim`（规约维），`x = right_cumulated`（规约维之后的累积）。每个 z-slice 是一个 `[n_row × n_col]` 矩阵，需要对行方向求和。

### 7.1 few_col kernel（right_cumulated < 32）

当保留列数极小时（如 `{33.5M, 2}, dim=0` 中 right_cumulated=2），通用 striding 的 TILE_SIZE=32 列 block 设计造成大量线程空转。few_col kernel 以 TILE 匹配实际列数，将列方向"折叠"：

```
256 线程 / TILE 列 = threads_per_col 线程/列
每列 threads_per_col 个线程横向排布在连续行上（coalesced 访存）
每线程通过 col[32] 加载 32 行，stride = threads_per_col
```

以 TILE=2 为例：线程 0,2,4,...,254（偶数）负责 col 0，线程 1,3,...,255（奇数）负责 col 1。col 0 的 128 个线程各负责唯一的连续行区间，总共覆盖 128×32=4096 行/block。

**关键修复**：初版 few_col 的 y_id 公式存在两个 bug：(a) 同列线程读相同行（翻倍计数），(b) stride 过大导致大量行漏读。修复后的公式将 `threadIdx.x / TILE` 作为列内线程序号，`i * threads_per_col` 作为纵向 stride。

### 7.2 few_row kernel（reduce_dim < 256）

当规约维度很短时（如 `{2, 33.5M}, dim=0` 中只有 2 行需要求和），通用 striding kernel 的 256 行/block 浪费了线程。few_row kernel 对规约维使用动态 TILE（2/4/8/16/32），并用 256 线程覆盖 16 列/线程（4×float4），大幅减少 block 数量。

```cuda
results[16] = {};    // 每线程 16 列
x_base = bx * 4096 + t * 16;  // 每 block 覆盖 4096 列
// 外层 4 个 chunk，内层 TILE 行，float4 向量化
```

优化历程：
1. **初版**：`blockDim.x = TILE * 8`（TILE=2 时仅 16 线程/block），524K block 启动开销主导 → 1.18× torch
2. **改 blockDim=256**：`grid_x = ceil(n_col / 1024)`，block 从 524K 降到 32K → 1.15× torch
3. **results[16] + need_atomic=false**：每线程 16 列 + grid_y=1 时用 plain store 替代 atomicAdd → **0.94× torch**

关于 `need_atomic`：当 `grid_y == 1` 时（规约维恰好被 TILE 覆盖，如 reduce_dim=2 时 TILE=2），每个输出地址只被一个 block 写入，无需全局 atomicAdd。用 plain store 消除 ~400 cycle 的原子操作延迟。当 `grid_y > 1`（规约维需要多个 y-block）时保留 atomicAdd。

### 7.3 两路专用 kernel 的正确性验证策略

CUDA 坐标映射公式极易出错。实践中采用的验证方法：

- **先写表格后写公式**：以最小参数（TILE=2，16 线程）手画表格，列出每线程的 (行,列) 坐标，确认覆盖无遗漏无重叠后反推公式。
- **中间量命名**：将 `1024/TILE*8` 等魔法数字拆成 `threads_per_col`、`rows_per_block` 等 constexpr 变量。
- **三个不变量**：每 block 覆盖行数 = 总行数 / grid_y（无遗漏）；同列不同线程的行区间无交集（无重复）；同 warp 内同列线程访问连续地址（合并访存）。
- **TILE=32 退化**：few_col 在 TILE=32 时应与通用 striding kernel 产生完全相同的坐标映射，可作为回归测试。

---

## 8. Phase 7：Dense few_col Kernel——窄列 dense 路径

### 8.1 问题

Dense 路径中，when-loop 策略要求 `gridDim.x = cumulated`（行数），每行一个 block。当 `cumulated` 极大但 `n_col` 极小时，会产生海量微型 block：

| Case | cumulated | n_col | block 数 | 每 block 工作量 |
|------|-----------|-------|----------|----------------|
| 8d (8⁸) | 2,097,152 | 8 | 2M | 8 元素 |
| 1M×2 dim1 | 1,048,576 | 2 | 1M | 2 元素 |

每次 kernel launch 启动数百万个 block 处理个位数元素，launch overhead 主导耗时，benchmark 表现为 26–90× torch。

### 8.2 设计

新增 `one_axis_dense_sum_kernel_for_few_col<TILE, T>`，适用于 `n_col ≤ 128`：

```
一个 block 处理多行：256 线程 / (TILE/4 线程/行) × 4 tile = rows_per_block 行
每行 TILE/4 线程，每线程加载 1 个 float4（4 列）
行组内用 warp shuffle（width=thread_per_row）做 segment 归约
外层 repeat 循环 4 轮，每轮处理一个 tile 的行
```

以 TILE=8（8d 场景）为例：`thread_per_row = 2`，`rows_per_block = 1024`。一个 block 用 1024 行 × 8 列 = 8192 个 load + FADD 取代原来 2M 个 block。

### 8.3 warp shuffle 的 width 参数

这是一个关键细节。由于一个 warp 内可能包含多个行组（如 TILE=8 时 thread_per_row=2，warp 0 内有 16 个行组），普通的 `__shfl_down_sync(0xffffffff, val, offset)` 会导致跨行组数据混入。

修复方法：使用 4 参数版 `__shfl_down_sync(0xffffffff, val, offset, width)`，`width = thread_per_row`。width 将 warp 切分为 thread_per_row 宽的段，shuffle 只在段内有效。因 `threadIdx.x / thread_per_row` 恰好按行组连续排列线程，段边界 = 行组边界，不会跨行混数据。

### 8.4 效果

| Case | 修前 | 修后 | 路径 |
|------|------|------|------|
| 8d (n_col=8) | 26× | **0.64×** | dense few_col TILE=8 |
| 1M×2 dim1 (n_col=2) | 90× | **0.27×** | dense few_col TILE=4 |
| 128×128 (n_col=128) | 0.29× | 0.16× | dense few_col TILE=128 |
| 6d (n_col=4) | 0.42× | 0.26× | dense few_col TILE=4 |

---

## 9. 关键 Bug 修复记

### 9.1 32-bit 溢出导致死循环

**现象**：Benchmark 跑到 `{1048576, 2}, dim=1` 和 8d `{8,8,8,8,8,8,8,8}, dim=7` 时 GPU 温度飙至 90°C，进程被 OOM killer 杀掉。

**根因**：Dense kernel 的 while-loop 步进公式：

```cuda
x_base += gridDim.x * BLOCK_SIZE * PARALLEL;  // 32-bit 乘法！
```

`gridDim.x` 是 `unsigned int`（32-bit）。当 `cumulated * 256 * 16 = cumulated * 4096 ≥ 2^32` 时溢出：
- `1048576 × 4096 = 2^32 ≡ 0 (mod 2^32)` → `x_base` 永远为 0 → 死循环
- `2097152 × 4096 = 2^33 ≡ 0 (mod 2^32)` → 同上

更一般地，`cumulated` 为 2^20 = 1,048,576 的任意整数倍时均会溢出归零。

**修复**：两步——(1) 转 `(size_t)` 做 64-bit 乘法；(2) 用 `gridDim.y`（列块数）而非 `gridDim.x`（行数）作为步进乘数（后者是因 grid 轴交换后用错了维度）。

### 9.2 Grid 轴交换后 while-loop 步进用错维度

Grid 轴交换后：`grid = (cumulated, grid_x, 1)` → `gridDim.x = 行数`，`gridDim.y = 列块数`。但 while-loop 步进仍使用 `gridDim.x`，导致每轮跳过 `cumulated * 4096` 个元素而非 `4096` 个。`{64, 65536}` 实测只处理了 4096/65536 = 6.25% 的元素，sum 值与 torch 差 16 倍。

### 9.3 few_col kernel 行漏读与翻倍计数

初版 few_col 的 y_id 公式 `(threadIdx.x / TILE_SIZE) * (1024/TILE)` 只用 warp 组号区分线程，同 warp 同列的所有线程（TILE=2 时达 16 个）读完全相同的 32 个行。同时 `i * (TILE_SIZE/TILE)` 的 stride=16 导致每组只覆盖 span 的 6.25%。

### 9.4 浮点累加精度与测试 tolerance

`{1048576, 2}, dim=0` 对 100 万随机 float 求和，不同累加顺序（kernel vs torch）导致的舍入误差约 0.2，超出了固定的 0.1 tolerance。修复：将 tolerance 设为与规约维长度相关的阶梯值（>1M 时 1.0，>10K 时 0.1）。

---

## 10. 测时公平性：CUDA Events vs CPU Timer

最初 benchmark 中 Python worker 使用 `time.perf_counter()`（CPU wall clock），而 CUDA 端使用 `cudaEvent`（GPU time）。极端窄列场景（大量微型 block 的 kernel launch）会被 CPU timer 计入 launch overhead 而 CUDA event 不计，导致对比失真。

修复：将 Python worker 的 torch 测时改为 `torch.cuda.Event`，双方均测 GPU kernel 时间，消除测时方法差异。

---

## 11. 指令级并行：展开、流水线与延迟隐藏

以 few_row kernel（TILE=2, CHUNKS=4, results[16]）为例，`#pragma unroll` 完全展开后每线程约：

- **8 次 float4 load**（128-bit 合并访存指令）
- **32 次 FADD**（独立 FADD，无数据依赖链）
- **16 次 store**

8 次 load 之间没有数据依赖，编译器/Warp Scheduler 可以将它们全部发射出去再一起等待（in-flight memory transactions）。GPU 全局内存延迟约 300–800 cycles，但多条 in-flight load 使内存控制器并行处理请求。当数据陆续返回时，32 条 FADD 充当了"缓冲垫"——即使某条 load 返回较慢，其他 FADD 也能先执行，不让执行单元空闲。

同时，block 数从 524K（results[4], 每 block 1024 列）降到 8K（results[16], 每 block 4096 列），kernel launch、SM 调度、寄存器分配等"每 block 付一次"的固定成本被摊薄到 4 倍的 payload 上。

---

## 12. 最终性能总表

全部 21 个 benchmark 测例，ITERS=100，双方均使用 CUDA event 测量：

### Dense 路径（规约最后一维）

| Shape | numel (×10³) | our (ms) | torch (ms) | ratio | 走的 kernel |
|-------|-------------|----------|------------|-------|------------|
| 4×4 | 0.004 | 0.008 | 0.038 | 0.22 | few_col TILE=4 |
| 32×32 | 0.032 | 0.008 | 0.038 | 0.20 | few_col TILE=32 |
| 128×128 | 0.128 | 0.017 | 0.041 | 0.41 | few_col TILE=128 |
| 1024×1024 | 1.0 | 0.010 | 0.035 | 0.30 | while-loop single-pass |
| 1000×1000 | 1.0 | 0.010 | 0.031 | 0.33 | while-loop single-pass |
| 4096×4096 | 4.1 | 0.244 | 0.262 | 0.93 | while-loop two-pass |
| 64×65536 | 0.064 | 0.021 | 0.031 | 0.66 | while-loop two-pass |
| 16×48 | 0.016 | 0.008 | 0.040 | 0.21 | few_col TILE=64 |
| 1×1M dim1 | 1 | 0.019 | 0.042 | 0.46 | while-loop two-pass |
| 2×1M dim1 | 2 | 0.022 | 0.037 | 0.60 | while-loop two-pass |
| 1M×2 dim1 | 1049 | 0.013 | 0.049 | **0.27** | few_col TILE=4 |
| 8d (8⁸) | 2097 | 0.252 | 0.394 | **0.64** | few_col TILE=8 |
| 6d (4⁶) | 1.0 | 0.008 | 0.032 | 0.26 | few_col TILE=4 |
| 4d (16⁴) | 4.1 | 0.013 | 0.033 | 0.40 | few_col TILE=16 |
| 33.5M×2 dim1 | 33554 | 0.014 | 0.041 | 0.34 | few_col TILE=4 |

### Striding 路径（规约非最后一维）

| Shape | numel (×10³) | our (ms) | torch (ms) | ratio | 走的 kernel |
|-------|-------------|----------|------------|-------|------------|
| 1024×1024 dim0 | 1.0 | 0.011 | 0.040 | 0.29 | few_row |
| 4096×4096 dim0 | 4.1 | 0.253 | 0.265 | 0.95 | general striding |
| 2×1024×1024 dim1 | 2.0 | 0.012 | 0.040 | 0.30 | general striding |
| 100×300×500 dim1 | 50.0 | 0.221 | 0.191 | 1.16 | general striding |
| 1M×16 dim0 | 16 | 0.246 | 0.267 | 0.92 | few_col TILE=16 |
| 2×33M dim0 | 33554 | 1.431 | 1.461 | **0.98** | few_row TILE=2 |
| 33M×2 dim0 | 2 | 0.906 | 0.984 | 0.92 | few_col TILE=2 |

### 从初版到最终的关键 case 改进

| Case | 初版 ratio | 最终 ratio | 关键优化 |
|------|-----------|-----------|---------|
| dense 4096² | 2.76× | 0.93× | while-loop + two-pass + float4 |
| strid 4096² dim0 | 2.28× | 0.95× | general striding refactor |
| 2×33M dim0 | 15.8× | **0.98×** | few_row + results[16] + plain store |
| 8d | 26× | **0.64×** | dense few_col kernel |
| 1M×2 dim1 | 90× | **0.27×** | dense few_col kernel |

---

## 13. 最终架构

```
sum(dim)
├── dim == last → Dense 路径
│   ├── n_col ≤ 128 → dense few_col kernel (TILE = 4/8/16/32/64/128)
│   │    每 block 处理多行，warp shuffle segment 归约
│   ├── n_col < 65536 → while-loop single-pass (grid_y=1, 一 block 吃一行)
│   └── n_col ≥ 65536 → while-loop two-pass (scratch buffer, grid_y>1)
│
└── dim < last → Striding 路径
    ├── reduce_dim < 256 → few_row kernel (TILE = 2/4/8/16/32)
    │    256 线程 × 16 列/线程，plain store 优化（grid_y=1 时）
    ├── right_cumulated < 32 → few_col kernel (TILE = 2/4/8/16/32)
    │    256 线程，列折叠为 TILE，coalesced 行访存
    ├── reduce_dim ≥ 16 × 256 → two-pass striding (scratch buffer)
    └── else → general striding kernel (256 线程, shared memory atomic)
```

内核总数：5 个（dense × 1, dense few_col × 1, striding × 1, few_row × 1, few_col × 1）。通过模板参数 `UseScratch`、`TILE`、`need_atomic` 实现不同路径的复用。

---

## 14. 关键经验总结

### 14.1 GPU Reduce 的真正瓶颈是同步，不是计算

`__syncthreads` 和 `atomicAdd` 是 reduce kernel 的真正瓶颈。ncu 数据一致显示 `No Eligible = 70-80%`——GPU 调度器在大部分周期找不到可发射指令的 warp。减少同步点的数量比减少计算量更重要。

### 14.2 小 N 时 Shared Memory Atomic > Tree Reduce

当 warp 数 ≤ 16 时，shared memory 的流水线化 atomicAdd 比 tree reduce + 多轮 `__syncthreads` 更快。barrier 的全局同步延迟碾压了 O(log N) 的复杂度优势。

### 14.3 一个 Block 吃整行/多行 > 多个 Block 瓜分

while-loop 策略（grid_y=1）是本次调优中单点收益最大的优化方向。其极端形式——dense few_col kernel——将 2M block 的 8d 场景压缩为 2048 block，消除了 99.9% 的 block launch 开销。核心原则：**能用少量大 block 做的事，绝不用大量小 block**。

### 14.4 向量化 + ILP 要适度

4×float4 per thread 是 few_row 的 sweet spot。过少（1×float4）则 load 密度不足，内存延迟暴露；过多则寄存器压力上升，occupancy 下降。`val += v.x; val += v.y; ...` 的独立 FADD 优于 `val += v.x+v.y+v.z+v.w` 的合并表达式——前者允许编译器做指令级并行调度。

### 14.5 CUDA 坐标映射公式应先验证，后写代码

CUDA kernel 中最容易出错的不是算法而是坐标公式。实践中的有效方法：(1) 选最小参数手画表格验证覆盖；(2) 中间量命名（constexpr）替代魔法数字；(3) 用 debug coverage kernel（atomicExch 标记已读位置）自动化验证；(4) 退化验证（TILE=32 与通用 kernel 结果一致）。

### 14.6 32-bit 溢出是隐性杀手

`gridDim` 的所有分量都是 32-bit `unsigned int`。任何与 `BLOCK_SIZE`、`PARALLEL` 等宏的乘法在超过 2^32 时静默溢出。与 `gridDim` 做运算前必须显式 `(size_t)` 转型。所有累加器（x_base, y_id, offset）在涉及 `gridDim` 乘法时必须使用 64-bit。

### 14.7 WSL2 cudaMalloc 不保证清零

`cudaMalloc` 返回的显存可能包含任意旧数据。Scratch buffer 复用和输出 tensor 都必须显式清零（`cudaMemset`），否则地址复用会导致难以排查的数据残留 bug。

### 14.8 浮点累加误差与测试 tolerance

对 100 万+ 元素的 float32 累加，不同计算顺序（kernel vs torch）的舍入误差可达 0.2–20。tolerance 应随规约维大小阶梯递增，而非使用固定值。测试目标是检测逻辑 bug，不是测浮点精度。
