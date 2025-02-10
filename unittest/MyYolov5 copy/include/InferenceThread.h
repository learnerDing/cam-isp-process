//This is InferenceThread.h
#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include "Thread.h"
#include "FrameQueue.h"
#include "yolov5.h"
#include <ncnn/net.h>

class InferenceThread : public Thread {
public:
    InferenceThread(FrameQueue<cv::Mat>& frameQueue);
    ~InferenceThread();
    void run() override;

private:
    ncnn::Net yolov5_net;
    FrameQueue<cv::Mat>* m_frameQueue;
    
    void init_model();
    void processFrame(const cv::Mat& frame);
};

#endif // INFERENCETHREAD_H