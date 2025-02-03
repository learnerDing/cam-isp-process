#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include "Thread.h"
#include "FrameQueue.h"
#include <vector>

// 前向声明Object结构体
struct Object;

class InferenceThread : public Thread {
public:
    InferenceThread(FrameQueue<cv::Mat>* frameQueue);
    void run() override;

private:
    FrameQueue<cv::Mat>* m_frameQueue; // 帧队列指针
    void processFrame(const cv::Mat& frame); // 处理单帧
};

#endif // INFERENCETHREAD_H