#include "../include/Tensor.h"
#include <cuda_runtime.h>
#include <iostream> // 引入iostream以便输出错误信息

// 辅助函数：检查 CUDA 错误
inline void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(error) << std::endl;
        // 可以选择是否终止程序，这里仅输出错误信息
        // exit(EXIT_FAILURE); // 如果希望在遇到错误时终止程序，可以取消注释
    }
}

// RGGB Bayer模式的双线性插值核函数
__global__ void debayer_rggb_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    const int x = idx % width;
    const int y = idx / width;

    // 边界检查（处理边缘像素）
    const int x_clamped = max(0, min(x, width - 1));
    const int y_clamped = max(0, min(y, height - 1));

    // 根据Bayer模式确定当前位置颜色
    const bool is_red_row = (y % 2) == 0;
    const bool is_red_col = (x % 2) == 0;

    float r, g, b;

    if (is_red_row) {
        if (is_red_col) { // R位置
            r = input[y * width + x];
            // 插值G（水平和垂直邻域）
            g = (input[y * width + max(0, x-1)] +
                 input[y * width + min(width-1, x+1)] +
                 input[max(0, y-1) * width + x] +
                 input[min(height-1, y+1) * width + x]) / 4.0f;
            // 插值B（四角邻域）
            b = (input[max(0, y-1) * width + max(0, x-1)] +
                 input[max(0, y-1) * width + min(width-1, x+1)] +
                 input[min(height-1, y+1) * width + max(0, x-1)] +
                 input[min(height-1, y+1) * width + min(width-1, x+1)]) / 4.0f;
        } else { // G位置（红色行）
            g = input[y * width + x];
            // 插值R（左右邻域）
            r = (input[y * width + max(0, x-1)] +
                 input[y * width + min(width-1, x+1)]) / 2.0f;
            // 插值B（上下邻域）
            b = (input[max(0, y-1) * width + x] +
                 input[min(height-1, y+1) * width + x]) / 2.0f;
        }
    } else {
        if (is_red_col) { // G位置（蓝色行）
            g = input[y * width + x];
            // 插值B（左右邻域）
            b = (input[y * width + max(0, x-1)] +
                 input[y * width + min(width-1, x+1)]) / 2.0f;
            // 插值R（上下邻域）
            r = (input[max(0, y-1) * width + x] +
                 input[min(height-1, y+1) * width + x]) / 2.0f;
        } else { // B位置
            b = input[y * width + x];
            // 插值G（水平和垂直邻域）
            g = (input[y * width + max(0, x-1)] +
                 input[y * width + min(width-1, x+1)] +
                 input[max(0, y-1) * width + x] +
                 input[min(height-1, y+1) * width + x]) / 4.0f;
            // 插值R（四角邻域）
            r = (input[max(0, y-1) * width + max(0, x-1)] +
                 input[max(0, y-1) * width + min(width-1, x+1)] +
                 input[min(height-1, y+1) * width + max(0, x-1)] +
                 input[min(height-1, y+1) * width + min(width-1, x+1)]) / 4.0f;
        }
    }

    // 写入CHW布局的输出（通道顺序为BGR）
    const int chw_offset = height * width;
    output[0 * chw_offset + y * width + x] = b; // Blue通道
    output[1 * chw_offset + y * width + x] = g; // Green通道
    output[2 * chw_offset + y * width + x] = r; // Red通道
}

// CUDA函数封装
void launchDebayer(const Tensor& input, Tensor& output) {
    assert(input.device() == DeviceType::GPU && "Input must be on GPU");
    assert(input.shape().size() == 3 && input.shape()[0] == 1 && "Input must be CHW [1, H, W]");
    assert(output.shape().size() == 3 && output.shape()[0] == 3 && "Output must be CHW [3, H, W]");

    const int width = input.width_;
    const int height = input.height_;
    const int total_pixels = width * height;

    // 计算线程配置
    const int block_size = 256; // 每block 256线程
    const int grid_size = (total_pixels + block_size - 1) / block_size;

    // 启动核函数并检查错误
    debayer_rggb_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        width,
        height
    );
    checkCudaError(cudaGetLastError(), "Kernel launch failed"); // 检查核函数启动错误

    // 同步设备并检查错误
    cudaDeviceSynchronize(); // 确保核函数执行完成
    checkCudaError(cudaGetLastError(), "cudaDeviceSynchronize failed"); // 检查同步错误
}