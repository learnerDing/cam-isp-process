// This is FrameQueue.cpp
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
void FrameQueue<T>::addFrame(std::shared_ptr<T> frame) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_frameQueue.push(frame); // 将帧推入队列
    }
    m_cond.notify_one(); // 唤醒等待的消费者
}

// 从队列获取帧
template <typename T>
bool FrameQueue<T>::getFrame(std::shared_ptr<T>& frame) {
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
template <typename T>
void FrameQueue<T>::stop() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running = false; // 停止流
    }
    m_cond.notify_all(); // 唤醒所有等待的消费者
}

// 显式实例化模板类
template class FrameQueue<cv::Mat>; // 支持 OpenCV
template class FrameQueue<AVFrame>; // 支持 FFmpeg
/*使用说明：
当队列里面为cv::Mat
FrameQueue<std::shared_ptr<cv::Mat>> matQueue;
// 线程1：入队
auto src = std::make_shared<cv::Mat>(...); // 从 v4l2 获取的图像
matQueue.addFrame(src); // 入队

// 线程2：出队
std::shared_ptr<cv::Mat> dst;
if (matQueue.getFrame(dst)) {
    // 处理 dst...
}
=================================================================
当队列里面为AVFrame时
FrameQueue<std::shared_ptr<AVFrame>> frameQueue;

// 线程1：入队
auto src = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame* ptr) { av_frame_free(&ptr); });
// 填充 src 数据...
frameQueue.addFrame(src); // 入队

// 线程2：出队
std::shared_ptr<AVFrame> dst;
if (frameQueue.getFrame(dst)) {
    // 处理 dst...
}
*/