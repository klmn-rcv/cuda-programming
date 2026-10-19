#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void sgemm_naive_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // col in C
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // row in C
    if (x < N && y < M) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[y * K + i] * B[i * N + x];
        }
        C[y * N + x] = sum;
    }
}

// 一个简单易读的 CPU 参考实现，用于验证结果
static void sgemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // ---------------- 问题规模 ----------------
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t bytesA = sizeof(float) * (size_t)M * K;
    const size_t bytesB = sizeof(float) * (size_t)K * N;
    const size_t bytesC = sizeof(float) * (size_t)M * N;

    // ---------------- 主机内存 ----------------
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_C_ref = (float *)malloc(bytesC);

    // 用一些可复现的小随机数初始化
    srand(0);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    // ---------------- 设备内存 ----------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // ---------------- 配置 grid / block ----------------
    const int BLOCK = 32;
    dim3 gridDim(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK), 1);
    dim3 blockDim(BLOCK, BLOCK, 1);  // 32 * 32 = 1024 threads per block

    // ---------------- 启动内核 ----------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel time: %.3f ms\n", ms);

    // ---------------- 取回结果并验证 ----------------
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    sgemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double err = fabs((double)h_C[i] - (double)h_C_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max abs error vs CPU: %e\n", max_err);

    // ---------------- 清理 ----------------
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