#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <libavutil/frame.h>

class FrameQueue {
public:
    FrameQueue();
    ~FrameQueue();

    void addFrame(AVFrame* frame);
    bool getFrame(AVFrame** frame);
    void stop();

private:
    std::queue<AVFrame*> m_frameQueue; // 帧队列
    std::mutex m_mutex; // 互斥锁
    std::condition_variable m_cond; // 条件变量
    bool m_running; // 是否运行
};

#endif // FRAMEQUEUE_H