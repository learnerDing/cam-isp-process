// bgr2yuvtest.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// 计算Y分量的核函数
__global__ void bgr2yuv_y_kernel(const float* bgr, float* y, int width, int height) {
    const int y_row = blockIdx.x;
    if (y_row >= height) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int x = tid; x < width; x += stride) {
        // BGR内存布局：B通道在前，接着G通道，最后R通道
        const int B = y_row * width + x;
        const int G = width * height + y_row * width + x;
        const int R = 2 * width * height + y_row * width + x;

        const float fb = bgr[B];
        const float fg = bgr[G];
        const float fr = bgr[R];

        y[y_row * width + x] = 0.299f * fr + 0.587f * fg + 0.114f * fb;
    }
}

// 计算UV分量的核函数
__global__ void bgr2yuv_uv_kernel(const float* bgr, float* yuv, int width, int height) {
    const int uv_row = blockIdx.x;
    const int uv_height = height / 2;
    const int uv_width = width / 2;
    
    if (uv_row >= uv_height) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int u_x = tid; u_x < uv_width; u_x += stride) {
        const int orig_x = u_x * 2;
        const int orig_y = uv_row * 2;

        float u_sum = 0.0f, v_sum = 0.0f;
        int count = 0;

        // 处理2x2区域
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int x = orig_x + dx;
                const int y = orig_y + dy;
                
                if (x >= width || y >= height) continue;

                const int B = y * width + x;
                const int G = width * height + y * width + x;
                const int R = 2 * width * height + y * width + x;

                const float fb = bgr[B];
                const float fg = bgr[G];
                const float fr = bgr[R];

                u_sum += -0.169f * fr - 0.331f * fg + 0.5f * fb;
                v_sum += 0.5f * fr - 0.419f * fg - 0.081f * fb;
                count++;
            }
        }

        const float u_avg = u_sum / count;
        const float v_avg = v_sum / count;

        // YUV420内存布局：Y全分辨率在前，U半分辨率，V半分辨率最后
        const int y_size = width * height;
        const int uv_size = uv_width * uv_height;
        const int u_index = y_size + uv_row * uv_width + u_x;
        const int v_index = y_size + uv_size + uv_row * uv_width + u_x;

        yuv[u_index] = u_avg;
        yuv[v_index] = v_avg;
    }
}

int main() {
    const int W = 4, H = 4;  // 测试用4x4图像
    const int BGR_SIZE = W * H * 3;
    const int Y_SIZE = W * H;
    const int UV_SIZE = (W/2) * (H/2);
    const int YUV_SIZE = Y_SIZE + 2 * UV_SIZE;

    // 初始化全蓝色图像（B=1, G=0, R=0）
    float* h_bgr = new float[BGR_SIZE]{0};
    float* h_yuv = new float[YUV_SIZE]{0};
    
    // 设置B通道（最后W*H个元素）
    float* b_channel = h_bgr;
    std::fill(b_channel, b_channel + W * H, 1.0f);

    // GPU内存分配
    float *d_bgr, *d_yuv;
    cudaMalloc(&d_bgr, BGR_SIZE * sizeof(float));
    cudaMalloc(&d_yuv, YUV_SIZE * sizeof(float));

    // 拷贝数据到GPU
    cudaMemcpy(d_bgr, h_bgr, BGR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 执行核函数
    dim3 y_block(256);
    dim3 y_grid(H);
    bgr2yuv_y_kernel<<<y_grid, y_block>>>(d_bgr, d_yuv, W, H);

    dim3 uv_block(256);
    dim3 uv_grid(H/2);
    bgr2yuv_uv_kernel<<<uv_grid, uv_block>>>(d_bgr, d_yuv, W, H);

    // 拷贝结果回CPU
    cudaMemcpy(h_yuv, d_yuv, YUV_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印前20个YUV数值
    std::cout << "First 20 YUV values:" << std::endl;
    for (int i = 0; i < 24; ++i) {
        if (i < Y_SIZE) {
            std::cout << "Y[" << i << "]: " << h_yuv[i] << std::endl;
        } else {
            int u_index = i - Y_SIZE;
            std::cout << "U[" << u_index << "]: " << h_yuv[i] << std::endl;
        }
    }

    // 清理资源
    delete[] h_bgr;
    delete[] h_yuv;
    cudaFree(d_bgr);
    cudaFree(d_yuv);

    return 0;
}