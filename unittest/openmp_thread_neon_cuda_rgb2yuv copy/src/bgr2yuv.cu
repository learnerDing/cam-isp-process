// src/bgr2yuv.cu
#include "bgr2yuv.cuh"
#include <cuda_runtime.h>
//不需要使用模板，因为一般的嵌入式开发板只支持fp32
// 计算Y分量的核函数

// 计算Y分量的核函数（修正通道顺序）
// 计算Y分量的核函数（一维block，每个block处理一行）
__global__ void bgr2yuv_y_kernel(const float*  bgr,
                                 float*  y,
                                 int width, int height) {
    const int y_row = blockIdx.x;  // 每个block处理一行，此blcokIdx代表现在处理的是图像的多少行
    if (y_row >= height) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;//stride = 256

    for (int x = tid; x < width; x += stride) {//x处于0-255
        // 计算通道索引
        const int B = y_row * width + x;                // B通道偏移
        const int G = width * height + y_row * width + x; // G通道
        const int R = 2 * width * height + y_row * width + x; // R通道

        const float fb = bgr[B];
        const float fg = bgr[G];
        const float fr = bgr[R];

        y[y_row * width + x] = 0.299f * fr + 0.587f * fg + 0.114f * fb;
    }
}

// 计算UV分量的核函数（一维block，每个block处理一行UV）
__global__ void bgr2yuv_uv_kernel(const float*  bgr,
                                  float*  yuv,
                                  int width, int height) {
    const int uv_row = blockIdx.x;  // 每个block处理一行UV
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

        const int y_size = width * height;
        const int uv_size = uv_width * uv_height;
        const int u_index = y_size + uv_row * uv_width + u_x;
        const int v_index = y_size + uv_size + uv_row * uv_width + u_x;

        yuv[u_index] = u_avg;
        yuv[v_index] = v_avg;
    }
}

//d_bgr数据为[3,h,w]的bgr排列的数据，而d_yuv的数据为[1,1,1.5*h*w]的yuv420p数据
void launch_bgr2yuv(Tensor* d_bgr, Tensor* d_yuv, int width, int height) {
    // 新增类型和设备断言
    assert(d_bgr->dtype() == DataType::FLOAT32 && "Input tensor must be FLOAT32");
    assert(d_yuv->dtype() == DataType::FLOAT32 && "Output tensor must be FLOAT32");
    assert(d_bgr->device() == DeviceType::GPU && "Input tensor must be on GPU");
    assert(d_yuv->device() == DeviceType::GPU && "Output tensor must be on GPU");

    // 显式获取浮点指针
    float* bgr_data = static_cast<float*>(d_bgr->data());
    float* yuv_data = static_cast<float*>(d_yuv->data());

    // 处理Y分量：一维grid（height行），一维block（256线程）
    dim3 block(256);  // 32的倍数
    dim3 grid(height); //加入图像是1024*1024，则grid=1024
    printf("block=256 grid=%d\n",height);
    bgr2yuv_y_kernel<<<grid, block>>>(bgr_data, yuv_data, width, height);

    // 处理UV分量：一维grid（uv_height行），一维block（256线程）
    const int uv_height = height / 2;
    dim3 uv_block(256);
    dim3 uv_grid(uv_height);
    bgr2yuv_uv_kernel<<<uv_grid, uv_block>>>(bgr_data, yuv_data, width, height);

    cudaDeviceSynchronize();
}
