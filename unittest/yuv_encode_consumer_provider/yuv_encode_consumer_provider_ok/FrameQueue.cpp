#include "FrameQueue.h"

// 构造函数
template <typename T>
FrameQueue<T>::FrameQueue() : m_running(true) {}

// 析构函数
template <typename T>
FrameQueue<T>::~FrameQueue() {
    stop(); // 确保在销毁之前停止
}

// 添加帧到队列
template <typename T>
void FrameQueue<T>::addFrame(T* frame) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        //m_frameQueue.push(frame); // 直接添加帧指针,导致第二帧的pts变成负无穷出现bug
        T* newFrame = av_frame_clone(frame); // 必须要深拷贝 AVFrame
        m_frameQueue.push(newFrame); // 将深拷贝的帧推入队列
    }
    m_cond.notify_one(); // 唤醒等待的消费者
}

// 从队列获取帧
template <typename T>
bool FrameQueue<T>::getFrame(T** frame) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // 等待帧的到来
    m_cond.wait(lock, [this] { return !m_frameQueue.empty() || !m_running; });

    // 如果流停止且没有帧可取
    if (!m_running && m_frameQueue.empty()) {
        return false;
    }

    *frame = m_frameQueue.front(); // 获取队首帧
    m_frameQueue.pop();
    return true; // 成功获取帧
}

// 停止流
template <typename T>
void FrameQueue<T>::stop() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running = false; // 停止流
    }
    m_cond.notify_all(); // 唤醒所有等待的消费者
}

// 显式实例化模板类
template class FrameQueue<AVFrame>;
//template class FrameQueue<cv::Mat>;//支持opencv