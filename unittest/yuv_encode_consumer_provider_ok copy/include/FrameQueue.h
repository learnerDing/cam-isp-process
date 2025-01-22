//This is FrameQueue.h
// FrameQueue.h
#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory> // 包含智能指针头文件

extern "C" {
#include <libavutil/frame.h>
}

template <typename T>
class FrameQueue {
public:
    FrameQueue();
    ~FrameQueue();

    void addFrame(std::shared_ptr<T> frame); // 入队
    bool getFrame(std::shared_ptr<T>& frame); // 出队
    void stop();

private:
    std::queue<std::shared_ptr<T>> m_frameQueue; // 使用 shared_ptr 管理帧
    std::mutex m_mutex; // 互斥锁
    std::condition_variable m_cond; // 条件变量
    bool m_running; // 是否运行
};

#endif // FRAMEQUEUE_H