#include <iostream>
#include "CCMcuda.h"
// #include "src/utils/cuda_debug_utils.cuh"
#include "macro.h"


template <typename T>
__global__ void CCMGemm(T* img_data, T* CCM_mat, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;

    if (idx >= num_pixels) return;

    // 每个像素在不同通道上的索引
    int r_idx = idx;
    int g_idx = idx + num_pixels;
    int b_idx = idx + 2 * num_pixels;

    // 读取当前像素的RGB值
    T r = img_data[r_idx];
    T g = img_data[g_idx];
    T b = img_data[b_idx];

    // CCM矩阵乘法，假设CCM矩阵是3x3的
    T new_r = CCM_mat[0] * r + CCM_mat[1] * g + CCM_mat[2] * b;
    T new_g = CCM_mat[3] * r + CCM_mat[4] * g + CCM_mat[5] * b;
    T new_b = CCM_mat[6] * r + CCM_mat[7] * g + CCM_mat[8] * b;

    // 将校正后的结果写回
    img_data[r_idx] = new_r;
    img_data[g_idx] = new_g;
    img_data[b_idx] = new_b;
}
/*主函数：调用CCM
d_img_tensor 输入图片的TensorWrapper类，位于GPU
图片数据格式[Channnel,rows,cols]
CCM_mat_tensor CCM矩阵的TensorWrapper类，位于GPU

*/ 

template <typename T>
void launchCCMGemm(TensorWrapper<T>* d_img_tensor, TensorWrapper<T>* CCM_mat_tensor ,int rows, int cols) {
    int num_pixels = rows * cols;
    int threads_per_block = 256;
    int num_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    
    CCMGemm<T><<<num_blocks, threads_per_block>>>(d_img_tensor->data, CCM_mat_tensor->data, rows, cols);

}

// 特化模板函数
template void launchCCMGemm(TensorWrapper<float>* d_img_tensor, TensorWrapper<float>* CCM_mat_tensor,int rows, int cols);