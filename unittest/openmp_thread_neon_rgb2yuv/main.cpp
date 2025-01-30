#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

// OpenMP实现
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
            
            ptr_yuv[3*c]   = (uchar)(0.299*R + 0.587*G + 0.114*B);
            ptr_yuv[3*c+1] = (uchar)(-0.169*R - 0.331*G + 0.5*B + 128);
            ptr_yuv[3*c+2] = (uchar)(0.5*R - 0.419*G - 0.081*B + 128);
        }
    }
}

// NEON实现
void bgr2yuv_neon(const cv::Mat& bgr, cv::Mat& yuv) {
    CV_Assert(bgr.type() == CV_8UC3);
    yuv.create(bgr.size(), CV_8UC3);

    for(int r = 0; r < bgr.rows; ++r) {
        const uint8_t* ptr_bgr = bgr.ptr<uint8_t>(r);
        uint8_t* ptr_yuv = yuv.ptr<uint8_t>(r);
        
        int c = 0;
        for(; c <= bgr.cols - 8; c += 8) {
            uint8x8x3_t bgr_pixels = vld3_u8(ptr_bgr + c*3);
            
            uint16x8_t R = vmovl_u8(bgr_pixels.val[2]);
            uint16x8_t G = vmovl_u8(bgr_pixels.val[1]);
            uint16x8_t B = vmovl_u8(bgr_pixels.val[0]);

            uint16x8_t Y = vaddq_u16(vaddq_u16(
                vmulq_n_u16(R, 76),
                vmulq_n_u16(G, 150)),
                vmulq_n_u16(B, 29)
            );

            int16x8_t R_s16 = vreinterpretq_s16_u16(R);
            int16x8_t G_s16 = vreinterpretq_s16_u16(G);
            int16x8_t B_s16 = vreinterpretq_s16_u16(B);
            
            int16x8_t U = vaddq_s16(
                vaddq_s16(
                    vmulq_n_s16(R_s16, -43),
                    vmulq_n_s16(G_s16, -85)
                ),
                vaddq_s16(
                    vmulq_n_s16(B_s16, 128),
                    vdupq_n_s16(128 << 8)
                )
            );

            int16x8_t V = vaddq_s16(
                vaddq_s16(
                    vmulq_n_s16(R_s16, 128),
                    vmulq_n_s16(G_s16, -107)
                ),
                vaddq_s16(
                    vmulq_n_s16(B_s16, -21),
                    vdupq_n_s16(128 << 8)
                )
            );

            uint8x8_t Y_u8 = vshrn_n_u16(Y, 8);
            uint8x8_t U_u8 = vqmovun_s16(vshrq_n_s16(U, 8));
            uint8x8_t V_u8 = vqmovun_s16(vshrq_n_s16(V, 8));

            uint8x8x3_t yuv_pixels;
            yuv_pixels.val[0] = Y_u8;
            yuv_pixels.val[1] = U_u8;
            yuv_pixels.val[2] = V_u8;
            vst3_u8(ptr_yuv + c*3, yuv_pixels);
        }

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

// OpenCV实现
void bgr2yuv_opencv(const cv::Mat& bgr, cv::Mat& yuv) {
    cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);
}

// CPU基础实现
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

// 计时宏
#define BENCHMARK(func) \
do { \
    auto start = std::chrono::high_resolution_clock::now(); \
    for (int i = 0; i < TRIALS; ++i) { \
        func(bgr, yuv); \
    } \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
    std::cout << #func << " average time: " << duration / static_cast<float>(TRIALS) << " ms" << std::endl; \
} while(0)

int main() {
    const int W = 1920;
    const int H = 1080;
    const int TRIALS = 100;
    
    cv::Mat bgr = cv::Mat(H, W, CV_8UC3);
    cv::randu(bgr, 0, 255);
    cv::Mat yuv;

    BENCHMARK(bgr2yuv_opencv);
    BENCHMARK(bgr2yuv_openmp);
    BENCHMARK(bgr2yuv_neon);
    BENCHMARK(bgr2yuv_cpu);

    return 0;
}