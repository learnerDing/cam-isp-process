
#include <iostream>
#include "AWB.h"
// #include "cuda_debug_utils.cuh"
#include "macro.h"

// 用于每个block内的并行求和
template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if (laneid == 0) {
        warpsum[wid] = val;
    }
    __syncthreads();
    T sum = tid < warpnum ? warpsum[tid] : (T)0;
    sum = warpReduceSum<T>(sum);
    return sum;
}

// 计算每个通道的均值并应用增益
template <typename T>
__global__ void AWB_compute_and_apply_gain(T* img_data, int rows, int cols) {
    float channel_sum[3]={0};
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int num_pixels = rows * cols;
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;

    // 每个线程累加一部分像素的RGB值
    for (int i = idx; i < num_pixels; i += stride) {
        r_sum += img_data[i * 3];     // R通道
        g_sum += img_data[i * 3 + 1]; // G通道
        b_sum += img_data[i * 3 + 2]; // B通道
    }

    // 归约得到整个block的sum
    r_sum = blockReduceSum<float>(r_sum);
    g_sum = blockReduceSum<float>(g_sum);
    b_sum = blockReduceSum<float>(b_sum);

    // 保存到全局内存，只使用block内的0号线程
    if (threadIdx.x == 0) {
        atomicAdd(&channel_sum[0], r_sum);//原子加 channel_sum[0]+=r_sum
        atomicAdd(&channel_sum[1], g_sum);
        atomicAdd(&channel_sum[2], b_sum);
    }

    // 计算均值和增益并应用到像素中
    __syncthreads(); // 确保每个线程都完成归约
        // 这些值应该是共享内存中的增益

    __shared__ float r_gain, b_gain;//存在shared_memory中
    if (threadIdx.x == 0) {
        float r_avg = channel_sum[0] / num_pixels;
        float g_avg = channel_sum[1] / num_pixels;
        float b_avg = channel_sum[2] / num_pixels;
        r_gain = g_avg / r_avg;
        b_gain = g_avg / b_avg;

}

    __syncthreads();


    if (idx < num_pixels) {
        for(int i=idx ; i<num_pixels;i+=stride){
            img_data[i * 3] = min(255.0f, img_data[i * 3] * r_gain);        // R通道
            img_data[i * 3 + 2] = min(255.0f, img_data[i * 3 + 2] * b_gain); // B通道
    }
    }
}

// 主函数：调用AWB
template <typename T>
void launchAWB(TensorWrapper<T>* d_img_tensor, int rows, int cols) {
    int num_pixels = rows * cols;
    int threads_per_block = 256;
    int num_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    // 用于存储各个通道的累加值
    float* d_channel_sum;
    cudaMalloc(&d_channel_sum, 3 * sizeof(float));
    cudaMemset(d_channel_sum, 0, 3 * sizeof(float));
    
    //算子融合
    AWB_compute_and_apply_gain<T><<<num_blocks, threads_per_block>>>(d_img_tensor->data, rows, cols);
    
    //此时d_img_tensor->data里面的值已经修改过了
    // 释放资源
    cudaFree(d_channel_sum);
}

// 特化模板函数
template void launchAWB(TensorWrapper<float>* d_img_tensor, int rows, int cols);