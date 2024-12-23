#include "stream.h"

// 构造函数
Stream::Stream() : m_running(true) {}

// 析构函数
Stream::~Stream() {
    stop(); // 确保在销毁之前停止
}

// 添加帧到队列
void Stream::addFrame(const cv::Mat& frame) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_frameQueue.push(frame.clone()); // 深拷贝帧，防止外部修改
    }
    m_cond.notify_one(); // 唤醒等待的消费者
}

// 从队列获取帧
bool Stream::getFrame(cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // 等待帧的到来
    m_cond.wait(lock, [this] { return !m_frameQueue.empty() || !m_running; });

    // 如果流停止且没有帧可取
    if (!m_running && m_frameQueue.empty()) {
        return false;
    }

    frame = m_frameQueue.front(); // 获取队首帧
    m_frameQueue.pop();
    return true; // 成功获取帧
}

// 停止流
void Stream::stop() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running = false; // 停止流
    }
    m_cond.notify_all(); // 唤醒所有等待的消费者
}