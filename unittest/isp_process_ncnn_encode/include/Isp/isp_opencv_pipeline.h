//This is isp_opencv_pipeline.h
#ifndef ISP_OPENCV_PIPELINE_H
#define ISP_OPENCV_PIPELINE_H

#include "../util/Tensor.h"
#include "../util/FrameQueue.h"
#include "../util/Thread.h"
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include <omp.h>
#include <vector>
#include <memory>
#include "../util/MatbgrToAVFrameyuv.h"

extern "C" {
#include <libavutil/frame.h>
}
class ISPOpenCVPipeline : public Thread {
public:
    explicit ISPOpenCVPipeline(FrameQueue<cv::Mat>& frameQueue);
    ~ISPOpenCVPipeline() override;
    // 新增输出队列管理接口
    void addInferOutputQueue(FrameQueue<cv::Mat>* queue);//向后对接推理线程
    void addEncodeOutputQueue(FrameQueue<AVFrame>* queue);//向后对接编码线程
protected:
    void run() override;

private:
    FrameQueue<cv::Mat>& m_inputQueue;
    std::vector<FrameQueue<cv::Mat>*> m_inferQueues;    // 推理队列列表(支持多个)
    std::vector<FrameQueue<AVFrame>*> m_encodeQueues;   // 编码队列列表(支持多个)

    // 新增转换函数 From cv::Mat bgr to AVFrame yuv420p
    std::shared_ptr<AVFrame> convertToAVFrame(const cv::Mat& mat);
    cv::Mat applyDebayer(const cv::Mat& bayerImage); // OpenCV Bayer 插值
    void applyAWB(cv::Mat& image); // 自动白平衡
    void applyCCM(cv::Mat& image); // 色彩校正矩阵
    void applyGamma(cv::Mat& image); // Gamma 校正
};

#endif // ISP_OPENCV_PIPELINE_H
