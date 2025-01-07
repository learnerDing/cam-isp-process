#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
extern "C" {
#include <libavutil/frame.h>
}
//struct AVFrame;
template <typename T>
class FrameQueue {
public:
    FrameQueue();
    ~FrameQueue();

    void addFrame(T* frame);
    bool getFrame(T** frame);
    void stop();

private:
    std::queue<T*> m_frameQueue; // 帧队列
    std::mutex m_mutex; // 互斥锁
    std::condition_variable m_cond; // 条件变量
    bool m_running; // 是否运行
};

#endif // FRAMEQUEUE_H