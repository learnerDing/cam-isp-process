//This is InferenceThread.h
#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include "../util/Thread.h"
#include "PreviewThread.h"
#include "../util/FrameQueue.h"
#include "yolov5.h"
#include <ncnn/net.h>

class InferenceThread : public Thread {
public:
    InferenceThread(FrameQueue<cv::Mat>& frameQueue, 
                    FrameQueue<cv::Mat>* previewQueue);
    void run() override;
    virtual ~InferenceThread();  // 声明析构函数
private:
    ncnn::Net yolov5_net;
    FrameQueue<cv::Mat>* m_frameQueue;
    FrameQueue<cv::Mat>* m_previewQueue; // 新增预览队列指针
    
    void init_model();
    void processFrame(const cv::Mat& frame);
};

#endif // INFERENCETHREAD_H