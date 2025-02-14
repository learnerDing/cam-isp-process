#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>
#include <arm_neon.h>
// 定义debayer灰度世界算法、颜色校正矩阵和 Gamma 校正函数
void applyAWB(cv::Mat& image);
void applyCCM(cv::Mat& image);
void applyGamma(cv::Mat& image);

float clamp(float value, float min, float max) {
    return std::min(std::max(value, min), max);
}

int main() {
    // 创建一个 1920x1080 的三通道假图像数据
    cv::Mat image(1080, 1920, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); // 随机填充像素值

    // 定义运行次数
    const int iterations = 100;

    // 测试 applyAWB 函数
    auto startAWB = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cv::Mat temp = image.clone(); // 每次使用原始图像的副本
        applyAWB(temp);
    }
    auto endAWB = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeAWB = endAWB - startAWB;
    double avgTimeAWB = timeAWB.count() / iterations;

    // 测试 applyCCM 函数
    auto startCCM = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cv::Mat temp = image.clone(); // 每次使用原始图像的副本
        applyCCM(temp);
    }
    auto endCCM = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeCCM = endCCM - startCCM;
    double avgTimeCCM = timeCCM.count() / iterations;

    // 测试 applyGamma 函数
    auto startGamma = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cv::Mat temp = image.clone(); // 每次使用原始图像的副本
        applyGamma(temp);
    }
    auto endGamma = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeGamma = endGamma - startGamma;
    double avgTimeGamma = timeGamma.count() / iterations;

    // 输出结果
    std::cout << "Average time for applyAWB: " << avgTimeAWB * 1000 << " ms" << std::endl;
    std::cout << "Average time for applyCCM: " << avgTimeCCM * 1000 << " ms" << std::endl;
    std::cout << "Average time for applyGamma: " << avgTimeGamma * 1000 << " ms" << std::endl;

    return 0;
}

// 灰度世界算法
void applyAWB(cv::Mat& image) {
    assert(image.channels() == 3); // 确保是三通道图像

    // 计算 BGR 分量均值
    cv::Scalar mean = cv::mean(image);

    // 计算增益
    float K = (mean[0] + mean[1] + mean[2]) / 3.0f;
    float gain_B = K / mean[0];
    float gain_G = K / mean[1];
    float gain_R = K / mean[2];

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // 调整每个通道的值
    channels[0] *= gain_B; // Blue
    channels[1] *= gain_G; // Green
    channels[2] *= gain_R; // Red

    // 合并通道
    cv::merge(channels, image);
}

// 颜色校正矩阵
void applyCCM(cv::Mat& image) {
    const float32x4_t ccmmat_row0 = {2.34403f, 0.00823594f, -0.0795542f, 0.0f};
    const float32x4_t ccmmat_row1 = {-1.18042f, 1.44385f, 0.0806464f, 0.0f};
    const float32x4_t ccmmat_row2 = {-0.296824f, -0.556513f, 0.909063f, 0.0f};

    #pragma omp parallel for
    for (int r = 0; r < image.rows; ++r) {
        uchar* row_ptr = image.ptr<uchar>(r);
        for (int c = 0; c < image.cols; c += 4) { // 每次处理 4 个像素
            uint8x8x3_t pixels = vld3_u8(row_ptr + c * 3); // 加载 3 通道像素

            // 将像素转换为浮点数
            float32x4_t b = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(pixels.val[0]))));
            float32x4_t g = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(pixels.val[1]))));
            float32x4_t r = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(pixels.val[2]))));

            // 应用颜色校正矩阵
            float32x4_t new_r = vfmaq_f32(vmulq_f32(ccmmat_row0, r), vmulq_f32(ccmmat_row1, g), vmulq_f32(ccmmat_row2, b));
            float32x4_t new_g = vfmaq_f32(vmulq_f32(ccmmat_row0, r), vmulq_f32(ccmmat_row1, g), vmulq_f32(ccmmat_row2, b));
            float32x4_t new_b = vfmaq_f32(vmulq_f32(ccmmat_row0, r), vmulq_f32(ccmmat_row1, g), vmulq_f32(ccmmat_row2, b));

            // 钳位到 [0, 255]
            new_r = vminq_f32(vmaxq_f32(new_r, vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f));
            new_g = vminq_f32(vmaxq_f32(new_g, vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f));
            new_b = vminq_f32(vmaxq_f32(new_b, vdupq_n_f32(0.0f)), vdupq_n_f32(255.0f));

            // 转换回 8 位整数
            uint8x8x3_t result;
            result.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(new_b)), vdup_n_u16(0)));
            result.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(new_g)), vdup_n_u16(0)));
            result.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(new_r)), vdup_n_u16(0)));

            // 存储结果
            vst3_u8(row_ptr + c * 3, result);
        }
    }
}

// Gamma 校正
void applyGamma(cv::Mat& image) {
    const float gamma = 0.3f;
    unsigned char LUT[256];
    for (int i = 0; i < 256; ++i) {
        LUT[i] = static_cast<uchar>(std::pow(i / 255.0f, gamma) * 255.0f);
    }

    #pragma omp parallel for
    for (int r = 0; r < image.rows; ++r) {
        uchar* row_ptr = image.ptr<uchar>(r);
        int num_pixels = image.cols * image.channels();
        for (int c = 0; c < num_pixels; c += 16) { // 每次处理 16 个像素
            uint8x16_t pixels = vld1q_u8(row_ptr + c);
            uint8x16_t result = vqtbl1q_u8(vld1q_u8(LUT), pixels); // 使用 NEON 表查找
            vst1q_u8(row_ptr + c, result);
        }
    }
}
