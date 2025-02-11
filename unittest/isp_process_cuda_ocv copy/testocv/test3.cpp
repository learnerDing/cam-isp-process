#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <arm_neon.h>
#include <omp.h>
// 声明函数
void applyAWB(cv::Mat& image);
void applyCCM(cv::Mat& image);
void applyGamma(cv::Mat& image);

float clamp(float value, float min, float max) {
    return std::min(std::max(value, min), max);
}
/*优化后
Average Time (ms):
Debayer: 4.4138
AWB:     26.3621
CCM:     17.6816
Gamma:   2.63032
Total:   51.0883
*/
int main() {
    // 读取RAW8图像文件
    const std::string rawFilename = "input.raw"; // 替换为实际文件路径
    cv::Mat rawImage(1080, 1920, CV_8UC1); // 单通道Bayer图像

    FILE* file = fopen(rawFilename.c_str(), "rb");
    if (!file) {
        std::cerr << "无法打开文件: " << rawFilename << std::endl;
        return -1;
    }

    size_t bytesRead = fread(rawImage.data, 1, 1920 * 1080, file);
    fclose(file);

    if (bytesRead != 1920 * 1080) {
        std::cerr << "文件大小错误。预期 " << 1920*1080 << " 字节，实际 " << bytesRead << " 字节。" << std::endl;
        return -1;
    }

    const int iterations = 100; // 运行次数
    double totalDebayer = 0.0, totalAWB = 0.0, totalCCM = 0.0, totalGamma = 0.0, totalAll = 0.0;

    for (int i = 0; i < iterations; ++i) {
        // 总计时开始
        auto startTotal = std::chrono::high_resolution_clock::now();
        
        // 1. Debayer处理
        auto startDebayer = std::chrono::high_resolution_clock::now();
        cv::Mat rgbImage;
        cv::cvtColor(rawImage, rgbImage, cv::COLOR_BayerRG2RGB); // 根据实际Bayer模式调整
        auto endDebayer = std::chrono::high_resolution_clock::now();
        totalDebayer += std::chrono::duration<double>(endDebayer - startDebayer).count();

        // 2. 自动白平衡
        auto startAWB = std::chrono::high_resolution_clock::now();
        applyAWB(rgbImage);
        auto endAWB = std::chrono::high_resolution_clock::now();
        totalAWB += std::chrono::duration<double>(endAWB - startAWB).count();

        // 3. 颜色校正矩阵
        auto startCCM = std::chrono::high_resolution_clock::now();
        applyCCM(rgbImage);
        auto endCCM = std::chrono::high_resolution_clock::now();
        totalCCM += std::chrono::duration<double>(endCCM - startCCM).count();

        // 4. Gamma校正
        auto startGamma = std::chrono::high_resolution_clock::now();
        applyGamma(rgbImage);
        auto endGamma = std::chrono::high_resolution_clock::now();
        totalGamma += std::chrono::duration<double>(endGamma - startGamma).count();

        // 总计时结束
        auto endTotal = std::chrono::high_resolution_clock::now();
        totalAll += std::chrono::duration<double>(endTotal - startTotal).count();
    }

    // 输出平均时间（毫秒）
    std::cout << "Average Time (ms):" << std::endl;
    std::cout << "Debayer: " << (totalDebayer / iterations) * 1000 << std::endl;
    std::cout << "AWB:     " << (totalAWB / iterations) * 1000 << std::endl;
    std::cout << "CCM:     " << (totalCCM / iterations) * 1000 << std::endl;
    std::cout << "Gamma:   " << (totalGamma / iterations) * 1000 << std::endl;
    std::cout << "Total:   " << (totalAll / iterations) * 1000 << std::endl;

    return 0;
}

// 灰度世界白平衡（已修复溢出问题）
void applyAWB(cv::Mat& image) {
    assert(image.channels() == 3);

    cv::Scalar mean = cv::mean(image);
    float K = (mean[0] + mean[1] + mean[2]) / 3.0f;
    float gains[3] = {K / mean[0], K / mean[1], K / mean[2]};

    cv::Mat floatImg;
    image.convertTo(floatImg, CV_32FC3);

    int rows = floatImg.rows;
    int cols = floatImg.cols;
    const int neonStep = 4;

    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        float* ptr = floatImg.ptr<float>(r);
        int c = 0;
        
        // NEON向量化处理
        float32x4_t gainB = vdupq_n_f32(gains[0]);
        float32x4_t gainG = vdupq_n_f32(gains[1]);
        float32x4_t gainR = vdupq_n_f32(gains[2]);
        
        for (; c <= (cols - neonStep)*3; c += neonStep*3) {
            float32x4x3_t pixels = vld3q_f32(ptr + c);
            
            pixels.val[0] = vmulq_f32(pixels.val[0], gainB);
            pixels.val[1] = vmulq_f32(pixels.val[1], gainG);
            pixels.val[2] = vmulq_f32(pixels.val[2], gainR);
            
            vst3q_f32(ptr + c, pixels);
        }
        
        // 处理剩余像素
        for (; c < cols*3; c += 3) {
            ptr[c] *= gains[0];
            ptr[c+1] *= gains[1];
            ptr[c+2] *= gains[2];
        }
    }

    floatImg.convertTo(image, CV_8UC3);
}

// 优化后的颜色校正矩阵（使用矩阵运算）
void applyCCM(cv::Mat& image) {
    const cv::Matx33f CCM(
        2.34403f,  -1.18042f, -0.296824f,
        0.00823594f, 1.44385f, -0.556513f,
        -0.0795542f, 0.0806464f, 0.909063f
    );

    cv::Mat floatImg;
    image.convertTo(floatImg, CV_32FC3);

    const float* ccm = CCM.val;
    const int neonStep = 4;
    int rows = floatImg.rows;
    int cols = floatImg.cols;

    // 将矩阵元素加载到寄存器
    float32x4_t ccm0 = vdupq_n_f32(ccm[0]);
    float32x4_t ccm1 = vdupq_n_f32(ccm[1]);
    float32x4_t ccm2 = vdupq_n_f32(ccm[2]);
    float32x4_t ccm3 = vdupq_n_f32(ccm[3]);
    float32x4_t ccm4 = vdupq_n_f32(ccm[4]);
    float32x4_t ccm5 = vdupq_n_f32(ccm[5]);
    float32x4_t ccm6 = vdupq_n_f32(ccm[6]);
    float32x4_t ccm7 = vdupq_n_f32(ccm[7]);
    float32x4_t ccm8 = vdupq_n_f32(ccm[8]);

    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        float* ptr = floatImg.ptr<float>(r);
        int c = 0;
        
        for (; c <= (cols - neonStep)*3; c += neonStep*3) {
            float32x4x3_t rgb = vld3q_f32(ptr + c);
            
            // 计算新通道值
            float32x4_t newR = vmlaq_f32(
                vmlaq_f32(vmulq_f32(rgb.val[0], ccm0), rgb.val[1], ccm1), 
                rgb.val[2], ccm2);
            float32x4_t newG = vmlaq_f32(
                vmlaq_f32(vmulq_f32(rgb.val[0], ccm3), rgb.val[1], ccm4), 
                rgb.val[2], ccm5);
            float32x4_t newB = vmlaq_f32(
                vmlaq_f32(vmulq_f32(rgb.val[0], ccm6), rgb.val[1], ccm7), 
                rgb.val[2], ccm8);
            
            rgb.val[0] = newR;
            rgb.val[1] = newG;
            rgb.val[2] = newB;
            
            vst3q_f32(ptr + c, rgb);
        }
        
        // 处理剩余像素
        for (; c < cols*3; c += 3) {
            float* p = ptr + c;
            float r = p[0], g = p[1], b = p[2];
            p[0] = r*ccm[0] + g*ccm[1] + b*ccm[2];
            p[1] = r*ccm[3] + g*ccm[4] + b*ccm[5];
            p[2] = r*ccm[6] + g*ccm[7] + b*ccm[8];
        }
    }

    floatImg.convertTo(image, CV_8UC3);
}

// Gamma校正（使用查找表优化）
void applyGamma(cv::Mat& image) {
    const float gamma = 0.3f;
    static uchar LUT[256];
    static std::once_flag flag;
    std::call_once(flag, [&]{
        for (int i = 0; i < 256; ++i)
            LUT[i] = cv::saturate_cast<uchar>(pow(i/255.0f, gamma)*255.0f);
    });

    int total = image.total() * image.channels();
    uchar* data = image.data;

    #pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        data[i] = LUT[data[i]];
    }
}