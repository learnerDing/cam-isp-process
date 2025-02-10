// This is FrameQueue.cpp
// FrameQueue.cpp
#include "FrameQueue.h"
#define  ENABLE_QUEUE_DEBUG
template <typename T>
FrameQueue<T>::FrameQueue(size_t maxSize, size_t discard_num) 
    : m_running(true),
      m_maxSize(maxSize > 0 ? maxSize : 1),
      m_discard_num((discard_num > 0) ? discard_num : 1) {}  // 确保最小丢弃数为1


template <typename T>
FrameQueue<T>::~FrameQueue() {
    stop();
}

template <typename T>
void FrameQueue<T>::addFrame(std::shared_ptr<T> frame) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // 当队列满时，丢弃m_discard_num帧
        if (m_frameQueue.size() >= m_maxSize) {
            size_t toDiscard = m_discard_num;
            while (toDiscard-- > 0 && !m_frameQueue.empty()) {
                m_frameQueue.pop();
            }
        }
        
        m_frameQueue.push(frame);
    }
    m_cond.notify_one();
}

// template <typename T>
// bool FrameQueue<T>::getFrame(std::shared_ptr<T>& frame) {
//     std::unique_lock<std::mutex> lock(m_mutex);
//     m_cond.wait(lock, [this] { return !m_frameQueue.empty() || !m_running; });

//     if (!m_running && m_frameQueue.empty()) {
//         return false;
//     }

//     frame = m_frameQueue.front();
//     m_frameQueue.pop();

// // 调试输出开关
// #ifdef ENABLE_QUEUE_DEBUG
//     std::cout << "当前队列大小: " << m_frameQueue.size() 
//               << "/" << m_maxSize << std::endl;
// #endif

//     return true;
// }

//D指导建议我将getFrame加入超时机制
template <typename T>
bool FrameQueue<T>::getFrame(std::shared_ptr<T>& frame, uint32_t timeout_ms ) {
    std::unique_lock<std::mutex> lock(m_mutex);
    
    // 使用wait_for替换wait
    const auto status = m_cond.wait_for(lock, 
        std::chrono::milliseconds(timeout_ms),
        [this] { return !m_frameQueue.empty() || !m_running; });

    // 处理等待结果
    if (!status || (!m_running && m_frameQueue.empty())) {
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