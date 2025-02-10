// include/PreviewThread.h
#pragma once
#include "FrameQueue.h"
#include "Thread.h"
#include <opencv2/opencv.hpp>

class PreviewThread : public Thread {
public:
    PreviewThread(FrameQueue<cv::Mat>& previewQueue, int fps = 5)
        : m_previewQueue(previewQueue), m_fps(fps) {}
        
    void run() override {
        const int delay = 1000 / m_fps;
        while (true) {
        // 双重终止条件判断 如果previewQueue已经停止并且队列已经空了则停止线程
        if (!m_previewQueue.isRunning() && m_previewQueue.empty()) {
            break;
        }
        std::shared_ptr<cv::Mat> frame;
        if (m_previewQueue.getFrame(frame, 50)) {
            // ...处理显示逻辑...
            cv::resize(*frame, m_displayFrame, cv::Size(), 0.5, 0.5); // 缩小分辨率
            cv::imshow("YOLOv5 Preview", m_displayFrame);
            cv::waitKey(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    }

private:
    FrameQueue<cv::Mat>& m_previewQueue;
    cv::Mat m_displayFrame;
    int m_fps;
};
