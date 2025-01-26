// main.cpp
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "include/rgb2yuv.h"
#include <arm_neon.h>
#define USE_CUDA
// 编译时通过-D定义这些宏来选择实现方式
// #define USE_OPENMP
// #define USE_NEON
// #define USE_CUDA
void rgb2yuv_openmp(const cv::Mat& rgb, cv::Mat& yuv) {
    CV_Assert(rgb.type() == CV_8UC3);
    yuv.create(rgb.size(), CV_8UC3);

    #pragma omp parallel for
for(int r = 0; r < rgb.rows; ++r) {
        const uchar* ptr_rgb = rgb.ptr<uchar>(r);
        uchar* ptr_yuv = yuv.ptr<uchar>(r);
        
        for(int c = 0; c < rgb.cols; ++c) {
            uchar B = ptr_rgb[3*c];
            uchar G = ptr_rgb[3*c+1];
            uchar R = ptr_rgb[3*c+2];
            
            // YUV转换公式
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);       // Y
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);  // U
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);   // V
        }
    }
}
// NEON实现 (可放在单独文件)



void rgb2yuv_neon(const cv::Mat& rgb, cv::Mat& yuv) {
    CV_Assert(rgb.type() == CV_8UC3);
    yuv.create(rgb.size(), CV_8UC3);

    for(int r = 0; r < rgb.rows; ++r) {
        const uint8_t* ptr_rgb = rgb.ptr<uint8_t>(r);
        uint8_t* ptr_yuv = yuv.ptr<uint8_t>(r);
        
        int c = 0;
        for(; c <= rgb.cols - 8; c += 8) {
            // 加载8个像素的RGB数据
            uint8x8x3_t rgb_pixels = vld3_u8(ptr_rgb + c*3);
            
            // 转换为16位进行计算
            uint16x8_t R = vmovl_u8(rgb_pixels.val[2]);
            uint16x8_t G = vmovl_u8(rgb_pixels.val[1]);
            uint16x8_t B = vmovl_u8(rgb_pixels.val[0]);
            
            // Y分量计算
            uint16x8_t Y = vaddq_u16(
                vaddq_u16(vmulq_n_u16(R, 76), vmulq_n_u16(G, 150)),
                vmulq_n_u16(B, 29)
            );
            
            // ...类似计算U和V分量...
            
            // 存储结果
            uint8x8x3_t yuv_pixels;
            yuv_pixels.val[0] = vshrn_n_u16(Y, 8); // 等效于 >>8
            // ...设置其他通道...
            vst3_u8(ptr_yuv + c*3, yuv_pixels);
        }
        // 处理剩余像素...
    }
}
// CPU基础实现
void rgb2yuv_cpu(const cv::Mat& rgb, cv::Mat& yuv) {
    CV_Assert(rgb.type() == CV_8UC3);
    yuv.create(rgb.size(), CV_8UC3);

    for(int r = 0; r < rgb.rows; ++r) {
        const uchar* ptr_rgb = rgb.ptr<uchar>(r);
        uchar* ptr_yuv = yuv.ptr<uchar>(r);
        
        for(int c = 0; c < rgb.cols; ++c) {
            uchar B = ptr_rgb[3*c];
            uchar G = ptr_rgb[3*c+1];
            uchar R = ptr_rgb[3*c+2];
            
            // YUV转换公式
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);       // Y
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);  // U
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);   // V
        }
    }
}

int main() {
    const int W = 1920;
    const int H = 1080;
    const int TRIALS = 100;
    
    // 生成测试图像
    cv::Mat rgb = cv::Mat(H, W, CV_8UC3);
    cv::randu(rgb, 0, 255);
    cv::Mat yuv;

    auto t_start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < TRIALS; ++i) {
#ifdef USE_CUDA
        // CUDA版本需要特殊处理内存
        cv::cuda::GpuMat d_rgb(rgb);
        cv::cuda::GpuMat d_yuv;
        rgb2yuv_cuda(d_rgb, d_yuv);
        d_yuv.download(yuv);
#else
# ifdef USE_OPENMP
        rgb2yuv_openmp(rgb, yuv);
# elif defined(USE_NEON)
        rgb2yuv_neon(rgb, yuv);
# else
        rgb2yuv_cpu(rgb, yuv);
# endif
#endif
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    std::cout << "Average time: " << duration / (float)TRIALS << "ms" << std::endl;
    return 0;
}