#include <cstdio>

template <int BLOCKSIZE>
__global__ void reduce_naive_kernel(int *arr, int *out, int len) {
    __shared__ int sdata[BLOCKSIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int i = bid * bdim + tid;   // 全局线程 id

    // 将数据拷贝到共享内存
    if(i < len) {
        sdata[tid] = arr[i];
    }

    __syncthreads();

    // 每个线程计算 log2(bdim)-1 个轮回
    // 比如 bdim = 8, 则每个线程计算 2 个轮回
    for(int s = 1; s < bdim; s *= 2) {  // s：归约步长，从 1 开始倍增
        if(tid % (2 * s) == 0 && i + s < len) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 每个 block 的第一个线程将结果写入到 out 中
    if(tid == 0) {
        out[bid] = sdata[0];
    }
}

const int len = 1000;

int main() {
    int *arr = new int[len];
    int *out = new int[len];
    int *d_arr, *d_out;     // device 上的地址

    // 初始化数组
    for(int i = 0; i < len; i++) {
        arr[i] = i;
    }

    // 分配内存
    cudaMalloc((void **)&d_arr, sizeof(int) * len);
    cudaMalloc((void **)&d_out, sizeof(int) * len);

    // 拷贝数据到显存
    cudaMemcpy(d_arr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);

    // 计算 block 和 grid 的大小
    constexpr int blocksize = 256;
    int gridsize = (len + blocksize - 1) / blocksize;

    // 调用 kernel 函数
    reduce_naive_kernel<blocksize><<<gridsize, blocksize>>>(d_arr, d_out, len); // 内核启动是异步的，CPU 不会等待 GPU 执行完内核

    // 拷贝数据到内存
    cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);  // cudaMemcpy 是同步的，这让原本需要调用的 cudaDeviceSynchronize 可以省略

    // 计算结果
    int sum = 0;
    // 注意是gridsize，不是blocksize
    // 因为每个block的第一个线程都会把自己的值写入到out中
    // gridsize是block的数量（结合图理解）
    for(int i = 0; i < gridsize; i++) {
        sum += out[i];
    }
    printf("sum = %d\n", sum);

    // 核对结果
    int sum2 = 0;
    for(int i = 0; i < len; i++) {
        sum2 += arr[i];
    }

    if(sum == sum2) {
        printf("success\n");
    } else {
        printf("failed\n");
    }

    // 释放内存
    cudaFree(d_arr);
    cudaFree(d_out);
    delete[] arr;
    delete[] out;

    return 0;
}