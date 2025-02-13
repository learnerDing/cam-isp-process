//This is Raw8Provider.h
#pragma once

#include "../../include/Thread.h"
#include "../../include/FrameQueue.h"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <memory>
#include <vector>
#include <string>

class Raw8Provider : public Thread {
public:
    explicit Raw8Provider(FrameQueue<cv::Mat>& frameQueue);
    ~Raw8Provider() override;

    void stop();

protected:
    void run() override;

private:
    FrameQueue<cv::Mat>& m_frameQueue;//缓冲区队列
    std::vector<std::shared_ptr<cv::Mat>> m_images;//五张raw图像
    std::atomic<bool> m_stop;

    void loadRawImages();
};
