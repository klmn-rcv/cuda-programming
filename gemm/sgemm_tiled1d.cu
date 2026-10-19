#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Fair 1D comparison variant:
// BM=BN=128, BK=8, 256 threads per block, and a 64x1 per-thread tile.
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_tiled1d_fair_kernel(
    const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int THREADS = (BM / TM) * BN;
    static_assert(BM % TM == 0, "BM must be divisible by TM");

    assert(blockDim.x == THREADS);

    const int thread_row = threadIdx.x / BN;
    const int thread_col = threadIdx.x % BN;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float thread_results[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        for (int idx = threadIdx.x; idx < BM * BK; idx += THREADS) {
            const int row = idx / BK;
            const int col = idx % BK;
            const int global_row = block_row + row;
            const int global_col = bk + col;
            As[idx] = (global_row < M && global_col < K)
                          ? A[global_row * K + global_col]
                          : 0.0f;
        }

        for (int idx = threadIdx.x; idx < BK * BN; idx += THREADS) {
            const int row = idx / BN;
            const int col = idx % BN;
            const int global_row = bk + row;
            const int global_col = block_col + col;
            Bs[idx] = (global_row < K && global_col < N)
                          ? B[global_row * N + global_col]
                          : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            const float b = Bs[dot * BN + thread_col];
            #pragma unroll
            for (int row = 0; row < TM; ++row) {
                thread_results[row] +=
                    As[(thread_row * TM + row) * BK + dot] * b;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < TM; ++row) {
        const int global_row = block_row + thread_row * TM + row;
        const int global_col = block_col + thread_col;
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = thread_results[row];
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

    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
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

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 64;
    constexpr int THREADS = (BM / TM) * BN;

    dim3 block(THREADS, 1, 1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sgemm_tiled1d_fair_kernel<BM, BN, BK, TM>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
