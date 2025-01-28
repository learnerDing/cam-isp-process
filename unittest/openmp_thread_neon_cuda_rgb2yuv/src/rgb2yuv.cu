// src/rgb2yuv.cu
#include "rgb2yuv.cuh"
#include <cuda_runtime.h>
//不需要使用模板，因为一般的嵌入式开发板只支持fp32
// 计算Y分量的核函数
__global__ void rgb2yuv_y_kernel(const float* __restrict__ rgb,
                                 float* __restrict__ y,
                                 int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coord = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y_coord >= height) return;

    const int R = y_coord * width + x;
    const int G = width * height + y_coord * width + x;
    const int B = 2 * width * height + y_coord * width + x;

    const float fr = rgb[R];
    const float fg = rgb[G];
    const float fb = rgb[B];

    y[y_coord * width + x] = 0.299f * fr + 0.587f * fg + 0.114f * fb;
}

// 计算UV分量的核函数（下采样到420）
__global__ void rgb2yuv_uv_kernel(const float* __restrict__ rgb,
                                  float* __restrict__ yuv,
                                  int width, int height) {
    const int u_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int u_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int uv_width = width / 2;
    const int uv_height = height / 2;

    if (u_x >= uv_width || u_y >= uv_height) return;

    // 原图起始位置
    const int orig_x = u_x * 2;
    const int orig_y = u_y * 2;

    float u_sum = 0, v_sum = 0;
    int count = 0;

    // 遍历2x2块
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int x = orig_x + dx;
            const int y = orig_y + dy;
            
            if (x >= width || y >= height) continue;

            const int R = y * width + x;
            const int G = width * height + y * width + x;
            const int B = 2 * width * height + y * width + x;

            const float fr = rgb[R];
            const float fg = rgb[G];
            const float fb = rgb[B];

            // 计算U和V值
            u_sum += -0.169f * fr - 0.331f * fg + 0.5f * fb + 128;
            v_sum += 0.5f * fr - 0.419f * fg - 0.081f * fb + 128;
            count++;
        }
    }

    // 计算平均值
    const float u_avg = u_sum / count;
    const float v_avg = v_sum / count;

    // YUV420P内存布局：Y + U + V
    const int y_size = width * height;
    const int uv_size = uv_width * uv_height;
    const int u_index = y_size + u_y * uv_width + u_x;
    const int v_index = y_size + uv_size + u_y * uv_width + u_x;

    yuv[u_index] = u_avg;
    yuv[v_index] = v_avg;
}

void launch_rgb2yuv(Tensor* d_rgb, Tensor* d_yuv, int width, int height) {
    // 新增类型和设备断言
    assert(d_rgb->dtype() == DataType::FLOAT32 && "Input tensor must be FLOAT32");
    assert(d_yuv->dtype() == DataType::FLOAT32 && "Output tensor must be FLOAT32");
    assert(d_rgb->device() == DeviceType::GPU && "Input tensor must be on GPU");
    assert(d_yuv->device() == DeviceType::GPU && "Output tensor must be on GPU");

    // 显式获取浮点指针
    float* rgb_data = static_cast<float*>(d_rgb->data());
    float* yuv_data = static_cast<float*>(d_yuv->data());

    // 处理Y分量
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1)/block.x, 
             (height + block.y - 1)/block.y);
    rgb2yuv_y_kernel<<<grid, block>>>(rgb_data, yuv_data, width, height);

    // 处理UV分量
    const int uv_width = width / 2;
    const int uv_height = height / 2;
    dim3 uv_block(16, 8);
    dim3 uv_grid((uv_width + uv_block.x - 1)/uv_block.x,
                (uv_height + uv_block.y - 1)/uv_block.y);
    rgb2yuv_uv_kernel<<<uv_grid, uv_block>>>(rgb_data, yuv_data, width, height);

    cudaDeviceSynchronize();
}
