#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using uint = unsigned int;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER, int TM, int TN>
__global__ void sgemm_bank_conflicts_kernel(
    int M, int N, int K, float *A, float *B, float *C) {
    constexpr uint WARPSIZE = 32;
    constexpr uint SMEM_A_STRIDE = BM + 4;

    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;
    const uint thread_num = blockDim.x;

    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4) * 4;
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4) * 4;

    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;
    const int stride_a = BM / ldg_a_num;
    const int stride_b = BK / ldg_b_num;

    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // A 保持转置后的 [K][M] 逻辑布局。M 方向 padding 4 个 float，
    // 使转置写入时 K=0 与 K=4 的线程不再落入同一个 bank，同时保持 float4 对齐。
    __shared__ float smem_a[2][BK * SMEM_A_STRIDE];
    __shared__ float smem_b[2][BK * BN];

    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_a[WMITER * TM];
    float reg_b[WNITER * TN];
    int write_index = 0;

    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {
        for (uint offset = 0; offset < BM; offset += stride_a) {
            const float4 tmp = FETCH_FLOAT4(A[OFFSET(offset + inner_row_a, inner_col_a, K)]);
            smem_a[write_index][OFFSET(inner_col_a, offset + inner_row_a, SMEM_A_STRIDE)] = tmp.x;
            smem_a[write_index][OFFSET(inner_col_a + 1, offset + inner_row_a, SMEM_A_STRIDE)] = tmp.y;
            smem_a[write_index][OFFSET(inner_col_a + 2, offset + inner_row_a, SMEM_A_STRIDE)] = tmp.z;
            smem_a[write_index][OFFSET(inner_col_a + 3, offset + inner_row_a, SMEM_A_STRIDE)] = tmp.w;
        }

        for (uint offset = 0; offset < BK; offset += stride_b) {
            FETCH_FLOAT4(smem_b[write_index][OFFSET(inner_row_b + offset, inner_col_b, BN)]) =
                FETCH_FLOAT4(B[OFFSET(inner_row_b + offset, inner_col_b, N)]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx) {
                for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(reg_a[warp_sub_row_idx * TM + m]) =
                        FETCH_FLOAT4(smem_a[write_index][OFFSET(
                            dot_idx,
                            warp_row * WM + warp_sub_row_idx * WSUBM +
                                thread_row_in_warp * TM + m,
                            SMEM_A_STRIDE)]);
                }
            }

            for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER;
                 ++warp_sub_col_idx) {
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(reg_b[warp_sub_col_idx * TN + n]) =
                        FETCH_FLOAT4(smem_b[write_index][OFFSET(
                            dot_idx,
                            warp_col * WN + warp_sub_col_idx * WSUBN +
                                thread_col_in_warp * TN + n,
                            BN)]);
                }
            }

            for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER;
                 ++warp_sub_row_idx) {
                for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER;
                     ++warp_sub_col_idx) {
                    for (int m = 0; m < TM; ++m) {
                        for (int n = 0; n < TN; ++n) {
                            thread_results[(warp_sub_row_idx * TM + m) * (WNITER * TN) +
                                           warp_sub_col_idx * TN + n] +=
                                reg_a[warp_sub_row_idx * TM + m] *
                                reg_b[warp_sub_col_idx * TN + n];
                        }
                    }
                }
            }
        }

        // 下一轮写另一个 shared-memory buffer，因此这里不需要第二次同步。
        write_index = 1 - write_index;
    }

    C += warp_row * WM * N + warp_col * WN;
    for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx) {
        for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER;
             ++warp_sub_col_idx) {
            float *C_interim = C + (warp_sub_row_idx * WSUBM) * N +
                               warp_sub_col_idx * WSUBN;
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(C_interim[OFFSET(m + thread_row_in_warp * TM,
                                                  n + thread_col_in_warp * TN, N)]) =
                        FETCH_FLOAT4(thread_results[(warp_sub_row_idx * TM + m) *
                                                    (WNITER * TN) +
                                                    warp_sub_col_idx * TN + n]);
                }
            }
        }
    }
}

static void sgemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
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
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WMITER = 2;
    constexpr int WNITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 8;
    constexpr int WARPSIZE = 32;
    constexpr int num_warps = (BM / WM) * (BN / WN);
    constexpr int thread_num = num_warps * WARPSIZE;

    if (M <= 0 || N <= 0 || K <= 0 || M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr,
                "M and N must be positive multiples of %d; K must be a positive multiple of %d.\n",
                BM, BK);
        return 1;
    }

    float *h_A = (float *)malloc(sizeof(float) * (size_t)M * K);
    float *h_B = (float *)malloc(sizeof(float) * (size_t)K * N);
    float *h_C = (float *)malloc(sizeof(float) * (size_t)M * N);
    float *h_C_ref = (float *)malloc(sizeof(float) * (size_t)M * N);
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    srand(0);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * (size_t)K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * (size_t)M * N));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(float) * (size_t)M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(float) * (size_t)K * N, cudaMemcpyHostToDevice));

    dim3 block(thread_num, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sgemm_bank_conflicts_kernel<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * (size_t)M * N,
                          cudaMemcpyDeviceToHost));
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
