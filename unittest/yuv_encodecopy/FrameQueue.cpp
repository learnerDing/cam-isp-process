// This is FrameQueue.cpp
// FrameQueue.cpp
#include "FrameQueue.h"
#define  ENABLE_QUEUE_DEBUG
template <typename T>
FrameQueue<T>::FrameQueue(size_t maxSize) 
    : m_running(true),
      m_maxSize(maxSize > 0 ? maxSize : 1) {}  // 确保最小容量为1

template <typename T>
FrameQueue<T>::~FrameQueue() {
    stop();
}

template <typename T>
void FrameQueue<T>::addFrame(std::shared_ptr<T> frame) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 自动丢弃旧帧直到有足够空间
        while (m_frameQueue.size() >= m_maxSize) {
            m_frameQueue.pop();
        }
        
        m_frameQueue.push(frame);
    }
    m_cond.notify_one();
}

template <typename T>
bool FrameQueue<T>::getFrame(std::shared_ptr<T>& frame) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cond.wait(lock, [this] { return !m_frameQueue.empty() || !m_running; });

    if (!m_running && m_frameQueue.empty()) {
        return false;
    }

    frame = m_frameQueue.front();
    m_frameQueue.pop();

// 调试输出开关
#ifdef ENABLE_QUEUE_DEBUG
    std::cout << "当前队列大小: " << m_frameQueue.size() 
              << "/" << m_maxSize << std::endl;
#endif

    return true;
}

template <typename T>
void FrameQueue<T>::stop() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running = false;
    }
    m_cond.notify_all();
}

// 显式实例化
template class FrameQueue<cv::Mat>;
template class FrameQueue<AVFrame>;