#include "InferenceThread.h"
#include "yolov5.h" // 包含检测函数和Object声明
#include <opencv2/opencv.hpp>

InferenceThread::InferenceThread(FrameQueue<cv::Mat>* frameQueue)
    : m_frameQueue(frameQueue) {}

void InferenceThread::run() {
    while (true) {
        std::shared_ptr<cv::Mat> frame;
        // 从队列获取帧，失败时退出循环
        if (!m_frameQueue->getFrame(frame)) {
            break;
        }
        processFrame(*frame);
    }
}

void InferenceThread::processFrame(const cv::Mat& frame) {
    std::vector<Object> objects;
    // 执行YOLOv5检测
    if (detect_yolov5(frame, objects) == 0) {
        // 绘制检测结果到帧（注意：imshow应在主线程调用）
        draw_objects(frame, objects);
    }
}