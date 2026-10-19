#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using uint = unsigned int;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(elem) (reinterpret_cast<float4 *>(&(elem))[0])

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vectorize_kernel(
    int M, int N, int K, float *A, float *B, float *C) {
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    const int block_row_thread = BN / TN;  // block 中，一行有几个线程
    const int block_col_thread = BM / TM;  // block 中，一列有几个线程
    // 一个线程负责计算 block 中 TM*TN 个元素
    const int thread_num = block_row_thread * block_col_thread;

    // 当前线程对应 thread tile（即 TM * TN）的左上角元素在 block tile（即 BM * BN） 中的位置
    const int thread_col = (threadIdx.x % block_row_thread) * TN;
    const int thread_row = (threadIdx.x / block_row_thread) * TM;

    // 每行把 4 个 float 作为一个内存块，当前线程搬运 A 的 inner_row_a 行中，从 inner_col_a 开始的连续 4 个 float
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4) * 4;

    // 每行把 4 个 float 作为一个内存块，当前线程搬运 B 的 inner_row_b 行中，从 inner_col_b 开始的连续 4 个 float
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4) * 4;

    // 每个线程搬运4个浮点数，完成搬运至 smem_a 需要所有线程搬运 ldg_a_num 轮
    const int ldg_a_num = BK * BM / thread_num / 4;
    // 每个线程搬运4个浮点数，完成搬运至 smem_b 需要所有线程搬运 ldg_b_num 轮
    const int ldg_b_num = BK * BN / thread_num / 4;

    // 一共 BM 行，搬运 ldg_a_num 轮，每轮搬运 stride_a 行
    const int stride_a = BM / ldg_a_num;
    // 一共 BN 行，搬运 ldg_b_num 轮，每轮搬运 stride_b 行
    const int stride_b = BK / ldg_b_num;

    __shared__ float smem_a[BK * BM];  // 注意 smem_a 转置了
    __shared__ float smem_b[BK * BN];

    float thread_results[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];
    float ldg_reg_a[BM * BK / thread_num];  // 存放单个线程取到的数

    // 计算当前线程负责计算的矩阵的起始位置（即把 A、B、C 指针移动到 tile 左上角）
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {
        for (int i = 0; i < BM; i += stride_a) {  // 一次 i 循环，并行加载 stride_a 行（单个线程只加载一行中的 4 个 float）
            int ldg_index = i / stride_a * 4;  // 上界为 BM * BK / thread_num（取不到，i = BM 时才能取到）
                                               // ldg_reg_a 存的就是单个线程取到的数（包含多次循环）（即不含其他线程取到的数），ldg_index 就是 ldg_reg_a 的当前下标（4 的倍数）
            FETCH_FLOAT4(ldg_reg_a[ldg_index]) = FETCH_FLOAT4(A[OFFSET(i + inner_row_a, inner_col_a, K)]);
            // smem_a 转置存，其中 ldg_reg_a 做中间缓存，目的是读取时可以按FLOAT4读取
            smem_a[OFFSET(inner_col_a, i + inner_row_a, BM)] = ldg_reg_a[ldg_index];
            smem_a[OFFSET(inner_col_a + 1, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 1];
            smem_a[OFFSET(inner_col_a + 2, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 2];
            smem_a[OFFSET(inner_col_a + 3, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 3];
        }

        for (int i = 0; i < BK; i += stride_b) {
            FETCH_FLOAT4(smem_b[OFFSET(inner_row_b + i, inner_col_b, BN)]) =
                FETCH_FLOAT4(B[OFFSET(inner_row_b + i, inner_col_b, N)]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(reg_a[m]) =
                    FETCH_FLOAT4(smem_a[OFFSET(dot_idx, thread_row + m, BM)]);
            }
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(reg_b[n])
                    = FETCH_FLOAT4(smem_b[OFFSET(dot_idx, thread_col + n, BN)]);
            }

            for (int m = 0; m < TM; m ++) {
                for (int n = 0; n < TN; n ++) {
                    thread_results[m * TN + n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n+=4) {
            FETCH_FLOAT4(C[OFFSET(thread_row + m, thread_col + n, N)]) = FETCH_FLOAT4(thread_results[m * TN + n]);
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

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int thread_num = (BN / TN) * (BM / TM);

    if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr,
                "M and N must be positive multiples of %d; K must be a positive multiple of %d.\n",
                BM, BK);
        return 1;
    }

    const size_t bytesA = sizeof(float) * (size_t)M * K;
    const size_t bytesB = sizeof(float) * (size_t)K * N;
    const size_t bytesC = sizeof(float) * (size_t)M * N;

    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_C_ref = (float *)malloc(bytesC);
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    srand(0);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(thread_num, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(M, N, K, d_A, d_B, d_C);
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
        const double err = fabs((double)h_C[i] - (double)h_C_ref[i]);
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
