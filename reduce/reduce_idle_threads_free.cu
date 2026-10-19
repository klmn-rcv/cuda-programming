#include <cstdio>

template <int BLOCKSIZE>
__global__ void reduce_naive_kernel(int *arr, int *out, int len) {
    // IDLE threads 修改前：
    /* __shared__ int sdata[BLOCKSIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int i = bid * bdim + tid;   // 全局线程 id

    // 将数据拷贝到共享内存
    if(i < len) {
        sdata[tid] = arr[i];
    } */
    
    // IDLE threads 修改后：
    // 本质：本来第一轮加法计算（s=1）只有一半线程活跃，所以想在一开始加载数据的时候（此时所有线程活跃），就进行一轮加法计算，提高线程利用率
    // 这样的效果是，一个 block 的线程可以处理 2 个 block 大小的数据，因此启动 kernel 时，原本的 gridsize 可以减小一半，节省资源
    // 这同时反映了当程序中有大量的空闲线程时，我们可以采取的两种优化手段：1. 让空闲的线程也做一些工作；2. 减少线程的数量
    __shared__ int sdata[BLOCKSIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    // 注意这里是 bdim * 2，因为我们要让一个线程干两个线程的活
    int i = bid * bdim * 2 + tid;   // 全局 id（因为第一轮相加是 2 个相邻的 block 两两相加，所以这里 bid 每增加 1，要跳过 2 个 block 大小）

    // 将数据拷贝到共享内存
    if(i < len) {
        sdata[tid] = arr[i] + arr[i + bdim];
    }

    __syncthreads();

    /* 不使用交错寻址：
    s = 1 时，只有 tid 为偶数的线程干活：0, 2, 4, 6, ...
    s = 2 时，只有 tid 是 4 的倍数的线程干活：0, 4, 8, 12, ...
    s = 4 时，只有 tid 是 8 的倍数的线程干活：0, 8, 16, ...
    这种模式下，同一个 warp 里会有的线程干活、有的线程空闲。

    for(int s = 1; s < bdim; s *= 2) {  // s 是归约步长
        if(tid % (2 * s) == 0 && i + s < len) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    } */
    
    /* 使用交错寻址：让活跃线程集中成连续区间
    第1轮	s = 1	tid = 0 ~ 127	warp 0~3 全活跃，warp 4~7 全空闲
    第2轮	s = 2	tid = 0 ~ 63	warp 0~1 全活跃，其余空闲
    第3轮	s = 4	tid = 0 ~ 31	warp 0 全活跃，其余空闲
    第4轮	s = 8	tid = 0 ~ 15	warp 0 中前 16 个线程活跃，后 16 个空闲，开始发散
    可见成功地消除了前 3 次迭代中的 warp divergence。另外还消除了取模操作。
    交错寻址的本质就是：让连续的线程尽可能保持一样的行为。

    但，这也造成了 bank conflict：
    在第一次迭代中，0 号线程需要加载 shared memory 的 0 号和 1 号地址，然后写回 0 号地址。
    同时，0 号 warp 的 16 号线程需要加载 shared memory 的 32 和 33 号地址，并写回 32 号地址。
    因此，在一个 warp 内同时访问了一个 bank 的不同内存地址（因为 0 号和 32 号地址都在 bank0 中），导致发生了 2 路的 Bank Conflict。
    （注意：只有同一个 warp 内的不同线程访问同一个 bank 才叫 bank conflict；不同 warp 的线程不算。）
    类似地，在第二次迭代过程中，会产生 4 路 bank conflict，越来越严重。
    for(int s = 1; s < bdim; s *= 2) {  // s 是归约步长
        int index = 2 * s * tid;
        if((index + s < bdim) && (bdim * bid + s < len)) { // index + s < bdim：防止共享内存sdata越界；bdim * bid + s < len：处理最后一个block数据不足的情况（数据数小于归约步长s）
                                                           // 简单说就是，第一个条件可能剪去每个block内最后几个thread；第二个条件可能剪去最后一个block
            sdata[index] += sdata[index + s];
            __syncthreads();
        }
    } */
    
    // 使用顺序寻址：避免 bank conflict
    // 即，不存在“同一个 warp 内的**不同线程**，在某条指令处同时访问了同一个bank”
    for(int s = blockDim.x/2; s > 0; s >>= 1) {  // s 是归约步长
        if (tid < s){
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
    // int gridsize = (len + blocksize - 1) / blocksize;
    int gridsize = (len + blocksize * 2 - 1) / (blocksize * 2); // 由于之前的 IDLE threads 优化，这里 block 数可以减少一半

    // 调用 kernel 函数
    reduce_naive_kernel<blocksize><<<gridsize, blocksize>>>(d_arr, d_out, len);

    // 拷贝数据到内存
    cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

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