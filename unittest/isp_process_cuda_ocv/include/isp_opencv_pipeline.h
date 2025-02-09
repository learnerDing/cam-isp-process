#ifndef ISP_OPENCV_PIPELINE_H
#define ISP_OPENCV_PIPELINE_H

#include "Tensor.h"
#include "FrameQueue.h"
#include "Thread.h"
#include <opencv2/opencv.hpp>

class ISPOpenCVPipeline : public Thread {
public:
    explicit ISPOpenCVPipeline(FrameQueue<cv::Mat>& frameQueue);
    ~ISPOpenCVPipeline() override;

protected:
    void run() override;

private:
    FrameQueue<cv::Mat>& m_frameQueue;

    // 预留其他模块接口
    cv::Mat applyDebayer(const cv::Mat& bayerImage); // OpenCV Bayer 插值
    void applyAWB(cv::Mat& image); // 自动白平衡
    void applyCCM(cv::Mat& image); // 色彩校正矩阵
    void applyGamma(cv::Mat& image); // Gamma 校正
};

#endif // ISP_OPENCV_PIPELINE_H
