// src/rgb2yuv.cu
#include "rgb2yuv.cuh"
#include <cuda_runtime.h>

template <typename T>
__global__ void rgb2yuv_kernel(const T* __restrict__ rgb,
                               T* __restrict__ yuv,
                               int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= width || y >= height) return;

    // CHW内存布局计算
    const int R = y*width + x;              // Red通道索引
    const int G = height*width + y*width +x; // Green通道索引
    const int B = 2*height*width + y*width +x; // Blue通道索引

    const T fr = rgb[R];
    const T fg = rgb[G];
    const T fb = rgb[B];

    // YUV转换公式
    yuv[R] =  0.299f*fr + 0.587f*fg + 0.114f*fb;      // Y
    yuv[G] = -0.169f*fr - 0.331f*fg + 0.5f*fb + 128;  // U
    yuv[B] =  0.5f*fr - 0.419f*fg - 0.081f*fb + 128;  // V
}

template<typename T>
void launch_rgb2yuv(Tensor* d_rgb, Tensor* d_yuv, int width, int height) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1)/block.x, 
              (height + block.y - 1)/block.y);

    rgb2yuv_kernel<T><<<grid, block>>>(
        static_cast<T*>(d_rgb->data()),
        static_cast<T*>(d_yuv->data()),
        width,
        height
    );
    cudaDeviceSynchronize();
}

template void launch_rgb2yuv<float>(Tensor* d_rgb, Tensor* d_yuv, int rows, int cols);