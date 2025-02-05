#include <cuda_runtime.h>
#include <iostream>
#include "macro.h"
#include "Raw10ToRaw8.h"
// CUDA kernel for RAW10 to RAW8 conversion
template<typename T> //数据格式Tensorwrapper->shape[1,2464,3264]  2464行，3264列 HW排列
__global__ void raw10_to_raw8_kernel(T* raw10, int rows, int cols) {
    // 获取当前线程的索引
    int thread_id = threadIdx.x; // 线程在block中的索引
    int block_id = blockIdx.x; // block在grid中的索引

    // 计算当前处理的是哪一行
    int r_id = block_id ; // 总行数 (2464)

    // 计算行偏移
    int start_index = (r_id * cols) ;

    // 每个线程处理一行中的3274个数据
    for (int i = thread_id; i < cols; i += blockDim.x) {//blockDim.x=256，一次处理256个数据，循环直到处理完
        // 将RAW10格式数据右移2位转换为RAW8格式数据
        raw10[start_index + i] /=4.0f;
    }
}

// Host function to call the kernel
template<typename T>
void launch_raw10_to_raw8_cuda(TensorWrapper<T>* d_raw10, int rows, int cols) {
    int total_blocks = rows;  // 总block数,一个block处理一行数据

    // 每个block的线程数
    int threadsPerBlock = 256; // 8个warp

    // 调用CUDA kernel
    raw10_to_raw8_kernel<<<total_blocks, threadsPerBlock>>>(d_raw10->data,
                                                            rows, 
                                                            cols);

    // 同步确保kernel执行完毕
    cudaDeviceSynchronize();
}
// zhaziqwe: 显式实例化模版函数，由于cuda的语法规则，cuda语法的语句不能存在.cpp文件里，因此只能在此实例化
template void launch_raw10_to_raw8_cuda(TensorWrapper<float>* d_raw10,         
                                   int rows,
                                   int cols);