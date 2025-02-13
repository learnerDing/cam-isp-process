//This is FrameQueue.h
// FrameQueue.h
#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>  // 用于调试输出

extern "C" {
#include <libavutil/frame.h>
}

template <typename T>
class FrameQueue {
public:
    explicit FrameQueue(size_t maxSize = 30);
    ~FrameQueue();

    void addFrame(std::shared_ptr<T> frame);
    bool getFrame(std::shared_ptr<T>& frame);
    void stop();

private:
    std::queue<std::shared_ptr<T>> m_frameQueue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    bool m_running;
    size_t m_maxSize;
};

#endif // FRAMEQUEUE_H