// 让每个线程计算一个 8*8 的网格
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tiled2d_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // Block 索引
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // Thread 索引
    // 每个线程负责 C 的一个 TM × TN 的子块，整个 block 覆盖 BM × BN 的数据 tile
    // 因此每个 block 的一行应当有 BN / TN 个线程。但由于在启动时让 block 简化成一维了，所以需要下面的行列切分
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // 二维 tile (block tile) 的大小（注意这指的是数据数，不是线程数）
    const uint total_results_block_tile = BM * BN;
    // 一个 block tile 需要的线程数量
    const uint number_threads_block_tile = total_results_block_tile / (TM * TN);
    assert(number_threads_block_tile == blockDim.x);   // 启动时把 block 简化成一维的了

    // 计算该 Thread 负责加载的共享内存索引
    // smem_A 的列数为 BK
    const uint inner_row_A = threadIdx.x / BK;  // inner_row_A 的范围很可能远小于需要加载的总行数 BM（通常后者是前者的倍数），原因是一个 block 的线程数量可能小于一个 block tile 的数据元素数量，因此需要用 for 循环，每次加载一部分行
    const uint inner_col_A = threadIdx.x % BK;
    // 计算每个 block 一次加载的行数
    const uint stride_A = number_threads_block_tile / BK;

    // smem_B 的列数为 BN
    const uint inner_row_B = threadIdx.x / BN;  // inner_row_B 情况基本同 inner_row_A
    const uint inner_col_B = threadIdx.x % BN;
    const uint stride_B = number_threads_block_tile / BN;  // 也是计算每个 block 一次加载的行数

    // 共享内存
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];

    // 线程结果和寄存器变量
    // 这只用来存单个线程的计算结果
    float thread_results[TM * TN] = {0.0};

    // 把 A、B、C 指针移动到 tile 左上角
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    // 外层循环
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load matrix A and B into shared memory
        for (uint load_offset = 0; load_offset < BM; load_offset += stride_A) { // 一个 block 一次循环加载 stride_A 行
            smem_A[(inner_row_A + load_offset) * BK + inner_col_A] =
                 A[(inner_row_A + load_offset) * K + inner_col_A];
        }
        for (uint load_offset = 0; load_offset < BK; load_offset += stride_B) { // 一个 block 一次循环加载 stride_B 行
            smem_B[(inner_row_B + load_offset) * BN + inner_col_B] =
                 B[(inner_row_B + load_offset) * N + inner_col_B];
        }
        __syncthreads();

        // 将 smem_A 和 smem_B 下次要加载的起始点分别沿 K 方向推进到下一个位置（注意 smem_A 的 K 方向是列方向，而 smem_B 的 K 方向是行方向）
        A += BK;
        B += BK * N;

        // 计算当前 block tile（大小 BM * BN）矩阵乘法的结果
        // 由于线程并行，每个线程实际上只负责大小为 TM * TN 的矩阵结果（线程在 block tile 中定位属于自己的 TM * TN，用的是 thread_row 和 thread_col）
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            float reg_m[TM];
            float reg_n[TN];
            // 加载矩阵 A 和矩阵 B 到寄存器中
            // 下面的循环加载了一个 TM * 1 的列向量
            // block 中一行有 BN / TN 个线程，它们的 thread_row 相同，因此加载的 TM * 1 的列向量完全相同
            // 而 block 中的不同行的 thread_row 不同，thread_row 有 BM / TM 个取值。当 dot_idx = 0 时，正好能拼出 smem_A 的第 0 列（形状 BM * 1），只不过代码里没有拼，因为是并行的，不同线程分别算一部分列向量与一部分行向量的外积，累加之后和整个列向量与整个行向量的外积是一样的。类似地，当 dot_idx = 1 时可以拼出第 1 列，以此类推
            for (uint i = 0; i < TM; i++) {
                reg_m[i] = smem_A[(thread_row * TM + i) * BK + dot_idx];
            }
            // 下面的循环加载了一个 1 * TN 的行向量
            // block 中一列有 BM / TM 个线程，它们的 thread_col 相同，因此加载的 1 * TN 的行向量完全相同
            // 而 block 中的不同列的 thread_col 不同，thread_col 有 BN / TN 个取值。当 dot_idx = 0 时，正好能拼出 smem_B 的第 0 行（形状 1 * BN），只不过代码里没有拼，因为是并行的，不同线程分别算一部分列向量与一部分行向量的外积，累加之后和整个列向量与整个行向量的外积是一样的。类似地，当 dot_idx = 1 时可以拼出第 1 行，以此类推
            for (uint i = 0; i < TN; i++) {
                reg_n[i] = smem_B[dot_idx * BN + thread_col * TN + i];
            }
            // 列向量 reg_m 和行向量 reg_n 的外积（即矩阵乘法），结果是 TM * TN 的矩阵，正是每个线程的计算结果
            // 这里得知道一个结论：向量外积之和 = 矩阵乘法结果。即把以下的算出来相加：
            // smem_A 的第 0 列 与 smem_B 的第 0 行的外积（dot_idx = 0 时算出）
            // smem_A 的第 1 列 与 smem_B 的第 1 行的外积（dot_idx = 1 时算出）
            // ……………………
            // smem_A 的第 BK-1 列 与 smem_B 的第 BK-1 行的外积（dot_idx = BK-1 时算出）
            // 以上 BK 个大小为 BM * BN 的矩阵相加，就是 smem_A 与 smem_B 相乘的结果。
            // 另外，外层循环 bkIdx 也会一直将移动着的 smem_A 与 smem_B 的乘法结果累加到 thread_results，直到外层循环结束，得到最终结果的对应 BM * BN 大小的部分（各 block 并行，因此得到最终结果 C）
            for (uint reg_idx_m = 0; reg_idx_m < TM; ++reg_idx_m) {
                for (uint reg_idx_n = 0; reg_idx_n < TN; ++reg_idx_n) {
                    thread_results[reg_idx_m * TN + reg_idx_n] +=
                        reg_m[reg_idx_m] * reg_n[reg_idx_n];
                }
            }
        }
        __syncthreads();
    }

    // 写回全局内存
    for (uint reg_idx_m = 0; reg_idx_m < TM; ++reg_idx_m) {
        for (uint reg_idx_n = 0; reg_idx_n < TN; ++reg_idx_n) {
            C[(thread_row * TM + reg_idx_m) * N + thread_col * TN + reg_idx_n] =
                thread_results[reg_idx_m * TN + reg_idx_n];
        }
    }
}

static void sgemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s M N K\n", argv[0]);
        return 1;
    }
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);

    const size_t bytesA = sizeof(float) * (size_t)M * K;
    const size_t bytesB = sizeof(float) * (size_t)K * N;
    const size_t bytesC = sizeof(float) * (size_t)M * N;

    float *h_A     = (float *)malloc(bytesA);
    float *h_B     = (float *)malloc(bytesB);
    float *h_C     = (float *)malloc(bytesC);
    float *h_C_ref = (float *)malloc(bytesC);

    srand(0);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // 二维 Thread Tile 参数（与教程一致：TM = TN = BK = 8）
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    constexpr int num_threads = BM * BN / (TM * TN);  // = 256

    dim3 blockDim(num_threads, 1, 1);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_tiled2d_kernel<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    sgemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double err = fabs((double)h_C[i] - (double)h_C_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max abs error vs CPU: %e\n", max_err);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}