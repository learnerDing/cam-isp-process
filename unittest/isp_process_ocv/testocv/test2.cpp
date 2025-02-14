#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
//debayer+AWB+CCM+Gamma未优化大概150ms
// 声明函数
void applyAWB(cv::Mat& image);
void applyCCM(cv::Mat& image);
void applyGamma(cv::Mat& image);

float clamp(float value, float min, float max) {
    return std::min(std::max(value, min), max);
}

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
    float gain_B = K / mean[0];
    float gain_G = K / mean[1];
    float gain_R = K / mean[2];

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // 转换为浮点并应用增益
    cv::Mat b_float, g_float, r_float;
    channels[0].convertTo(b_float, CV_32F);
    channels[1].convertTo(g_float, CV_32F);
    channels[2].convertTo(r_float, CV_32F);

    b_float *= gain_B;
    g_float *= gain_G;
    r_float *= gain_R;

    // 饱和转换回CV_8U
    b_float.convertTo(channels[0], CV_8U);
    g_float.convertTo(channels[1], CV_8U);
    r_float.convertTo(channels[2], CV_8U);

    cv::merge(channels, image);
}

// 优化后的颜色校正矩阵（使用矩阵运算）
void applyCCM(cv::Mat& image) {
    const cv::Matx33f CCM(
        2.34403f,  -1.18042f, -0.296824f,
        0.00823594f, 1.44385f, -0.556513f,
        -0.0795542f, 0.0806464f, 0.909063f
    );

    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32FC3);

    // 应用矩阵乘法
    for (int r = 0; r < floatImage.rows; ++r) {
        cv::Vec3f* ptr = floatImage.ptr<cv::Vec3f>(r);
        for (int c = 0; c < floatImage.cols; ++c) {
            cv::Vec3f pixel = ptr[c];
            ptr[c] = CCM * pixel;
        }
    }

    // 截断并转换回CV_8U
    floatImage = cv::max(floatImage, 0.0f);
    floatImage = cv::min(floatImage, 255.0f);
    floatImage.convertTo(image, CV_8UC3);
}

// Gamma校正（使用查找表优化）
void applyGamma(cv::Mat& image) {
    const float gamma = 0.3f;
    static unsigned char LUT[256];
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < 256; ++i) {
            LUT[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0f, gamma) * 255.0f);
        }
        initialized = true;
    }

    if (image.isContinuous()) {
        image.reshape(1, image.total() * image.channels()).forEach<uchar>([](uchar &p, const int*) {
            p = LUT[p];
        });
    } else {
        for (int r = 0; r < image.rows; ++r) {
            uchar* ptr = image.ptr<uchar>(r);
            for (int c = 0; c < image.cols * image.channels(); ++c) {
                ptr[c] = LUT[ptr[c]];
            }
        }
    }
}