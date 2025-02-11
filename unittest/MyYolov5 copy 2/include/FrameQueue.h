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
    explicit FrameQueue(size_t maxSize = 30, size_t discard_num = 1);
    ~FrameQueue();

    void addFrame(std::shared_ptr<T> frame);
    bool getFrame(std::shared_ptr<T>& frame,uint32_t timeout_ms=500);//引入超时机制
    void stop();
     // D指导建议新增运行状态查询
    bool isRunning() const { return m_running; }
    bool isEmpty() const;
private:
    std::queue<std::shared_ptr<T>> m_frameQueue;
    mutable std::mutex m_mutex;//互斥锁申明为mutable才能在const方法函数中修改锁的状态
    std::condition_variable m_cond;
    bool m_running;
    size_t m_maxSize;
    size_t m_discard_num; // 新增成员变量，控制丢弃帧数
};

#endif // FRAMEQUEUE_H