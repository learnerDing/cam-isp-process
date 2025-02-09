#include "../include/isp_opencv_pipeline.h"

ISPOpenCVPipeline::ISPOpenCVPipeline(FrameQueue<cv::Mat>& frameQueue)
    : m_frameQueue(frameQueue) {}

ISPOpenCVPipeline::~ISPOpenCVPipeline() {}

void ISPOpenCVPipeline::run() {
    while (true) {
        std::shared_ptr<cv::Mat> frame;
        if (!m_frameQueue.getFrame(frame)) {
            break; // 停止条件
        }

        // 执行 Debayer 处理
        cv::Mat debayeredImage = applyDebayer(*frame);

        // 预留其他模块处理
        applyAWB(debayeredImage);
        applyCCM(debayeredImage);
        applyGamma(debayeredImage);

        // TODO: 将结果保存或传递到下一个阶段
    }
}

cv::Mat ISPOpenCVPipeline::applyDebayer(const cv::Mat& bayerImage) {
    cv::Mat rgbImage;
    cv::cvtColor(bayerImage, rgbImage, cv::COLOR_BayerRG2RGB);
    return rgbImage;
}
//灰度世界算法
void ISPOpenCVPipeline::applyAWB(cv::Mat& image) {
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
float clamp(float value, float min, float max) {
    return std::min(std::max(value, min), max);
}

void ISPOpenCVPipeline::applyCCM(cv::Mat& image) {
    const float ccmmat[3][3] = {
        {2.34403f, 0.00823594f, -0.0795542f},
        {-1.18042f, 1.44385f, 0.0806464f},
        {-0.296824f, -0.556513f, 0.909063f}
    };
    cv::Mat CCM(3, 3, CV_32FC1, (void*)ccmmat);

    for (int r = 0; r < image.rows; ++r) {
        for (int c = 0; c < image.cols; ++c) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);
            float blue = pixel[0], green = pixel[1], red = pixel[2];

            // 应用颜色校正矩阵
            float new_r = CCM.at<float>(0, 0) * red + CCM.at<float>(1, 0) * green + CCM.at<float>(2, 0) * blue;
            float new_g = CCM.at<float>(0, 1) * red + CCM.at<float>(1, 1) * green + CCM.at<float>(2, 1) * blue;
            float new_b = CCM.at<float>(0, 2) * red + CCM.at<float>(1, 2) * green + CCM.at<float>(2, 2) * blue;

            // 确保输出值在 [0, 255] 范围内
            new_r = clamp(new_r, 0.0f, 255.0f);
            new_g = clamp(new_g, 0.0f, 255.0f);
            new_b = clamp(new_b, 0.0f, 255.0f);

            // 更新像素值
            image.at<cv::Vec3b>(r, c) = cv::Vec3b(static_cast<uchar>(new_b), static_cast<uchar>(new_g), static_cast<uchar>(new_r));
        }
    }
}

void ISPOpenCVPipeline::applyGamma(cv::Mat& image) {
    const float gamma = 0.3f; // 默认 Gamma 值
    unsigned char LUT[256];
    for (int i = 0; i < 256; ++i) {
        LUT[i] = static_cast<uchar>(std::pow(i / 255.0f, gamma) * 255.0f);
    }

    if (image.channels() == 1) { // 单通道
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                image.at<uchar>(r, c) = LUT[image.at<uchar>(r, c)];
            }
        }
    } else { // 多通道
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                cv::Vec3b& pixel = image.at<cv::Vec3b>(r, c);
                pixel[0] = LUT[pixel[0]]; // Blue
                pixel[1] = LUT[pixel[1]]; // Green
                pixel[2] = LUT[pixel[2]]; // Red
            }
        }
    }
}