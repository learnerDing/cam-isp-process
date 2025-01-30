// main.cpp
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
//#define USE_CUDA
// #define USE_OPENMP
// #define USE_NEON
#define USE_OPENCV
#ifdef USE_OPENMP
void bgr2yuv_openmp(const cv::Mat& bgr, cv::Mat& yuv) {
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    #pragma omp parallel for
for(int r = 0; r < bgr.rows; ++r) {
        const uchar* ptr_bgr = bgr.ptr<uchar>(r);
        uchar* ptr_yuv = yuv.ptr<uchar>(r);
        
        for(int c = 0; c < bgr.cols; ++c) {
            uchar B = ptr_bgr[3*c];
            uchar G = ptr_bgr[3*c+1];
            uchar R = ptr_bgr[3*c+2];
            
            // YUV转换公式
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);       // Y
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);  // U
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);   // V
        }
    }
}
#endif
// OpenMP实现
#ifdef USE_OPENMP
void bgr2yuv_openmp(const cv::Mat& bgr, cv::Mat& yuv) {
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    #pragma omp parallel for
    for(int r = 0; r < bgr.rows; ++r) {
        const uchar* ptr_bgr = bgr.ptr<uchar>(r);
        uchar* ptr_yuv = yuv.ptr<uchar>(r);
        
        for(int c = 0; c < bgr.cols; ++c) {
            uchar B = ptr_bgr[3*c];
            uchar G = ptr_bgr[3*c+1];
            uchar R = ptr_bgr[3*c+2];
            
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);       // Y
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);  // U
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);   // V
        }
    }
}
#endif

// NEON实现
#ifdef USE_NEON
void bgr2yuv_neon(const cv::Mat& bgr, cv::Mat& yuv) {
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    for(int r = 0; r < bgr.rows; ++r) {
        const uint8_t* ptr_bgr = bgr.ptr<uint8_t>(r);
        uint8_t* ptr_yuv = yuv.ptr<uint8_t>(r);
        
        int c = 0;
        for(; c <= bgr.cols - 8; c += 8) {
            uint8x8x3_t bgr_pixels = vld3_u8(ptr_bgr + c*3);
            
            // 扩展为16位无符号整数
            uint16x8_t R = vmovl_u8(bgr_pixels.val[2]);
            uint16x8_t G = vmovl_u8(bgr_pixels.val[1]);
            uint16x8_t B = vmovl_u8(bgr_pixels.val[0]);

            // Y分量计算 (0.299*R + 0.587*G + 0.114*B) 缩放256倍
            uint16x8_t Y = vaddq_u16(vaddq_u16(
                vmulq_n_u16(R, 76),    // 0.299*256 ≈ 76
                vmulq_n_u16(G, 150)),  // 0.587*256 ≈ 150
                vmulq_n_u16(B, 29)     // 0.114*256 ≈ 29
            );

            // U分量计算 (-0.169*R - 0.331*G + 0.5*B + 128) 缩放256倍
            int16x8_t R_s16 = vreinterpretq_s16_u16(R);
            int16x8_t G_s16 = vreinterpretq_s16_u16(G);
            int16x8_t B_s16 = vreinterpretq_s16_u16(B);
            
            int16x8_t U = vaddq_s16(
                vaddq_s16(
                    vmulq_n_s16(R_s16, -43),   // -0.169*256 ≈ -43
                    vmulq_n_s16(G_s16, -85)    // -0.331*256 ≈ -85
                ),
                vaddq_s16(
                    vmulq_n_s16(B_s16, 128),   // 0.5*256 = 128
                    vdupq_n_s16(128 << 8)      // 128*256
                )
            );

            // V分量计算 (0.5*R - 0.419*G - 0.081*B + 128) 缩放256倍
            int16x8_t V = vaddq_s16(
                vaddq_s16(
                    vmulq_n_s16(R_s16, 128),   // 0.5*256 = 128
                    vmulq_n_s16(G_s16, -107)   // -0.419*256 ≈ -107
                ),
                vaddq_s16(
                    vmulq_n_s16(B_s16, -21),   // -0.081*256 ≈ -21
                    vdupq_n_s16(128 << 8)      // 128*256
                )
            );

            // 右移8位并转换回uint8
            uint8x8_t Y_u8 = vshrn_n_u16(Y, 8);
            uint8x8_t U_u8 = vqmovun_s16(vshrq_n_s16(U, 8));
            uint8x8_t V_u8 = vqmovun_s16(vshrq_n_s16(V, 8));

            // 存储结果
            uint8x8x3_t yuv_pixels;
            yuv_pixels.val[0] = Y_u8;
            yuv_pixels.val[1] = U_u8;
            yuv_pixels.val[2] = V_u8;
            vst3_u8(ptr_yuv + c*3, yuv_pixels);
        }

        // 处理剩余像素
        for(; c < bgr.cols; ++c) {
            uchar B = ptr_bgr[3*c];
            uchar G = ptr_bgr[3*c+1];
            uchar R = ptr_bgr[3*c+2];
            
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);
        }
    }
}
#endif

// OpenCV实现
#ifdef USE_OPENCV
void bgr2yuv_opencv(const cv::Mat& bgr, cv::Mat& yuv) {
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);
}
#endif

// CPU基础实现
#ifdef USE_CPU
void bgr2yuv_cpu(const cv::Mat& bgr, cv::Mat& yuv) {
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    for(int r = 0; r < bgr.rows; ++r) {
        const uchar* ptr_bgr = bgr.ptr<uchar>(r);
        uchar* ptr_yuv = yuv.ptr<uchar>(r);
        
        for(int c = 0; c < bgr.cols; ++c) {
            uchar B = ptr_bgr[3*c];
            uchar G = ptr_bgr[3*c+1];
            uchar R = ptr_bgr[3*c+2];
            
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);
        }
    }
}
#endif

int main() {
    const int W = 1920;
    const int H = 1080;
    const int TRIALS = 100;
    
    // 生成测试图像
    cv::Mat bgr = cv::Mat(H, W, CV_8UC3);
    cv::randu(bgr, 0, 255);
    cv::Mat yuv;

    auto t_start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < TRIALS; ++i) {
        #if defined(USE_OPENMP)
            bgr2yuv_openmp(bgr, yuv);
        #elif defined(USE_NEON)
            bgr2yuv_neon(bgr, yuv);
        #elif defined(USE_OPENCV)
            bgr2yuv_opencv(bgr, yuv);
        #elif defined(USE_CPU)
            bgr2yuv_cpu(bgr, yuv);
        #else
            #error "No implementation selected"
        #endif
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "Average time: " << duration / static_cast<float>(TRIALS) << " ms" << std::endl;
    
    return 0;
}