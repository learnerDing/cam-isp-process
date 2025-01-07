#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "Gammacuda.h"
// #include "src/utils/cuda_debug_utils.cuh"
#include "macro.h"

// CUDA 核函数：对图像数据进行 Gamma 矫正
template <typename T>
__global__ void GammaCorrectionKernel(T* img_data, const T* lut, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = rows * cols;

    if (idx >= num_pixels) return;

    // 每个像素在不同通道上的索引
    int r_idx = idx; 
    int g_idx = idx + num_pixels; //做偏移
    int b_idx = idx + 2 * num_pixels;

    // 读取当前像素的 RGB 值
    T r = img_data[r_idx];
    T g = img_data[g_idx];
    T b = img_data[b_idx];

    // 使用 LUT 进行 Gamma 矫正
    img_data[r_idx] = lut[r];
    img_data[g_idx] = lut[g];
    img_data[b_idx] = lut[b];
}

/*
主函数：调用 Gamma Correction
d_img_tensor   输入图片的 TensorWrapper 类，位于 GPU
d_lut_tensor   LUT 的 TensorWrapper 类，位于 GPU
int rows, int cols 图片宽高
*/
template <typename T>
void launchGammaCorrection(TensorWrapper<T>* d_img_tensor, TensorWrapper<T>* d_lut_tensor, int rows, int cols) {
    int num_pixels = rows * cols;
    int threads_per_block = 256;
    int num_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    // 调用 GammaCorrectionKernel 核函数
    GammaCorrectionKernel<<<num_blocks, threads_per_block>>>(d_img_tensor->data, d_lut_tensor->data, rows, cols);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch GammaCorrectionKernel: " << cudaGetErrorString(err) << std::endl;
    }

    // 等待核函数执行完毕
    cudaDeviceSynchronize();
}

void launchGammaCorrection(TensorWrapper<float>* d_img_tensor, TensorWrapper<float>* d_lut_tensor, int rows, int cols);