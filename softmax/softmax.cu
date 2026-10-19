// 这个实现假设 cols <= BLOCKSIZE，即一个 block 能容纳一整行。如果 cols 更大，需要更复杂的多阶段 reduction。
template <int BLOCKSIZE>
__global__ void softmax_kernel_cuda(float *input, float *output, int rows, int cols) {
    int tid = threadIdx.x
    int bid = blockIdx.x

    if(bid >= rows) return; // 每行都要进行 softmax，每个 block 负责一行

    // Shared memory 用于 reduction
    __shared__ float s_max[BLOCKSIZE];
    __shared__ float s_sum[BLOCKSIZE];

    // === 第一步：求最大值 ===
    int idx = bid * cols + tid;
    float val = (tid < cols) ? input[idx] : -INFINITY;
    s_max[tid] = val;
    __syncthreads();

    for(int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < cols) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float row_max = s_max[0];

    // === 第二步：计算 exp(x - max) 并求和
    val = (tid < cols) ? expf(val - row_max) : 0.0f;
    s_sum[tid] = val;
    __syncthreads();

    for(int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < cols) {
            s_sum[tid] = s_sum[tid] + s_sum[tid + s];
        }
        __syncthreads();
    }

    float row_sum = s_sum[0];

    // === 第三步：归一化并写回 ===
    if(tid < cols) {
        output[idx] = val / row_sum;
    }
}