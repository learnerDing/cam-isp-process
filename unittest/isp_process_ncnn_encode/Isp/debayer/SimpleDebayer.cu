#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 核函数声明
__global__ void debayer_rggb_kernel(const float* input, float* output, int width, int height);

int main() {
    const int width = 4;
    const int height = 4;
    const int input_size = width * height;
    const int output_size = 3 * width * height;

    // 生成RGGB模式的测试数据（4x4）
    float h_input[] = {
        1.0, 0.5, 1.0, 0.5,   // Row 0: R G R G
        0.5, 0.0, 0.5, 0.0,   // Row 1: G B G B
        1.0, 0.5, 1.0, 0.5,   // Row 2: R G R G
        0.5, 0.0, 0.5, 0.0    // Row 3: G B G B
    };
    float* h_output = new float[output_size];

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 block(256);
    dim3 grid((width * height + block.x - 1) / block.x); // 计算足够的网格数
    debayer_rggb_kernel<<<grid, block>>>(d_input, d_output, width, height);

    // 同步并检查错误
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输入和输出
    printf("Input Bayer (4x4 RGGB):\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.1f ", h_input[y * width + x]);
        }
        printf("\n");
    }

    printf("\nOutput BGR (CHW layout):\n");
    const char* channels[] = {"Blue", "Green", "Red"};
    for (int c = 0; c < 3; c++) {
        printf("%s Channel:\n", channels[c]);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                printf("%.2f ", h_output[c * width * height + y * width + x]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // 释放资源
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

// RGGB Bayer模式的双线性插值核函数
__global__ void debayer_rggb_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    //线程与数据关系，一个线程处理一个图像像素点
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