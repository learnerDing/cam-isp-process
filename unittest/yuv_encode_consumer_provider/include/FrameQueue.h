#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
extern "C" {
#include <libavutil/frame.h>
}
// 定义帧类型枚举
enum FrameType {
    FRAME_TYPE_AVFRAME, // AVFrame 类型
    FRAME_TYPE_OTHER    // 其他类型
};
template <typename T>
class FrameQueue {
public:
    FrameQueue();
    ~FrameQueue();

    void addFrame(T* frame);
    bool getFrame(T** frame);
    void stop();
    // 提供一个公共方法，返回帧的类型
    FrameType getFrameType() const;
private:
    std::queue<T*> m_frameQueue; // 帧队列
    std::mutex m_mutex; // 互斥锁
    std::condition_variable m_cond; // 条件变量
    bool m_running; // 是否运行
};
// 实现 getFrameType 方法
template <typename T>
FrameType FrameQueue<T>::getFrameType() const {
    if constexpr (std::is_same<T, AVFrame>::value) {
        return FRAME_TYPE_AVFRAME; // 如果是 AVFrame 类型
    } else {
        return FRAME_TYPE_OTHER; // 其他类型
    }
}
#endif // FRAMEQUEUE_H