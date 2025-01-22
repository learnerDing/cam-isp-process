//以下是FrameQueue.cpp
#include "../include/FrameQueue.h"
#include <memory> // 包含智能指针头文件

// 构造函数
template <typename T>
FrameQueue<T>::FrameQueue(size_t maxSize) : m_running(true), m_maxSize(maxSize) {}

// 析构函数
template <typename T>
FrameQueue<T>::~FrameQueue() {
    stop(); // 确保在销毁之前停止
}

// // 添加帧到队列，队列如果满了则等待。
// template <typename T>
// void FrameQueue<T>::addFrame(std::shared_ptr<T> frame) {
//     {
//         std::unique_lock<std::mutex> lock(m_mutex);
//         // 等待队列有空闲空间
//         m_cond.wait(lock, [this] { return m_frameQueue.size() < m_maxSize || !m_running; });

//         if (!m_running) {
//             return; // 如果已经停止，直接返回
//         }

//         // 使用智能指针管理深拷贝的帧
//         std::shared_ptr<T> newFrame = deepCopy(frame);
//         m_frameQueue.push(newFrame); // 将深拷贝的帧推入队列
//     }
//     m_cond.notify_one(); // 唤醒等待的消费者
// }

//队列如果满了则丢弃掉最旧的一帧图像
template <typename T>
void FrameQueue<T>::addFrame(std::shared_ptr<T> frame) {
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_running) {
            return; // 如果已经停止，直接返回
        }

        // 如果队列已满，丢弃最旧的帧
        if (m_frameQueue.size() >= m_maxSize) {
            m_frameQueue.pop();
        }

        // 使用智能指针管理深拷贝的帧
        std::shared_ptr<T> newFrame = deepCopy(frame);
        m_frameQueue.push(newFrame); // 将深拷贝的帧推入队列
    }
    m_cond.notify_one(); // 唤醒等待的消费者
}
// 从队列获取帧
// template <typename T>
// bool FrameQueue<T>::getFrame(std::shared_ptr<T>& frame) {
//     std::unique_lock<std::mutex> lock(m_mutex);
//     // 等待帧的到来
//     m_cond.wait(lock, [this] { return !m_frameQueue.empty() || !m_running; });

//     // 如果流停止且没有帧可取
//     if (!m_running && m_frameQueue.empty()) {
//         return false;
//     }

//     frame = m_frameQueue.front(); // 获取队首帧
//     m_frameQueue.pop();
//     m_cond.notify_one(); // 唤醒可能等待的生产者
//     return true; // 成功获取帧
// }
//从队列获取帧,增加超时机制
template <typename T>
bool FrameQueue<T>::getFrame(std::shared_ptr<T>& frame, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // 等待帧的到来，增加超时机制
    if (!m_cond.wait_for(lock, timeout, [this] { return !m_frameQueue.empty() || !m_running; })) {
        return false; // 超时返回
    }

    // 如果流停止且没有帧可取
    if (!m_running && m_frameQueue.empty()) {
        return false;
    }

    frame = m_frameQueue.front(); // 获取队首帧
    m_frameQueue.pop();
    m_cond.notify_one(); // 唤醒可能等待的生产者
    return true; // 成功获取帧
}
// 停止流
template <typename T>
void FrameQueue<T>::stop() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running = false; // 停止流
    }
    m_cond.notify_all(); // 唤醒所有等待的消费者和生产者
}

// 深拷贝函数模板
template <typename T>
std::shared_ptr<T> FrameQueue<T>::deepCopy(const std::shared_ptr<T>& frame) {
    // 默认情况下，假设类型T有拷贝构造函数
    return std::make_shared<T>(*frame);
}

// 特化深拷贝函数模板为AVFrame
template <>
std::shared_ptr<AVFrame> FrameQueue<AVFrame>::deepCopy(const std::shared_ptr<AVFrame>& frame) {
    // 使用av_frame_clone进行深拷贝
    AVFrame* newFrame = av_frame_clone(frame.get());
    if (!newFrame) {
        throw std::runtime_error("Failed to clone AVFrame");
    }
    return std::shared_ptr<AVFrame>(newFrame, [](AVFrame* ptr) { av_frame_free(&ptr); });
}

// 特化深拷贝函数模板为cv::Mat
template <>
std::shared_ptr<cv::Mat> FrameQueue<cv::Mat>::deepCopy(const std::shared_ptr<cv::Mat>& frame) {
    // 使用cv::Mat的clone方法进行深拷贝
    return std::make_shared<cv::Mat>(frame->clone());
}

// 显式实例化模板类
template class FrameQueue<AVFrame>;
template class FrameQueue<std::shared_ptr<cv::Mat>>; // 支持OpenCV